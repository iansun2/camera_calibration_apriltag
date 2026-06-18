# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
PySide6 GUI for car-eye calibration.

Shows a top-down (X-Y) coverage heatmap of where the base has been sampled in
odom, a yaw-coverage bar, a virtual joystick that publishes ``cmd_vel`` with an
adjustable maximum speed, and the calibration controls / result readout.
"""

import math

import numpy
import rclpy
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from camera_calibration_apriltag.careye_hand_eye import (
    MIN_SAMPLES, compute_residual, matrix_to_transform_tuple,
    rotation_axis_rank, solve_hand_eye)
# Reuse the colormap, heatmap cell renderer and image converter from the
# intrinsics GUI.
from camera_calibration_apriltag.qt_gui import (
    HeatmapWidget, heat_qcolor, numpy_to_qimage)


class JoystickWidget(QWidget):
    """
    Square pad with a draggable knob.  Emits a normalized (x, y) in [-1, 1]
    with y pointing up (forward).  Returns to centre on release.
    """
    moved = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._knob = QPointF(0.0, 0.0)   # normalized, y up
        self._active = False

    def _radius(self):
        return min(self.width(), self.height()) / 2.0 - 12.0

    def _set_from_pos(self, pos):
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = self._radius()
        dx = (pos.x() - cx) / r
        dy = -(pos.y() - cy) / r          # screen y is down; flip to y-up
        mag = math.hypot(dx, dy)
        if mag > 1.0:
            dx, dy = dx / mag, dy / mag
        self._knob = QPointF(dx, dy)
        self.moved.emit(dx, dy)
        self.update()

    def mousePressEvent(self, event):
        self._active = True
        self._set_from_pos(event.position())

    def mouseMoveEvent(self, event):
        if self._active:
            self._set_from_pos(event.position())

    def mouseReleaseEvent(self, _event):
        self._active = False
        self._knob = QPointF(0.0, 0.0)
        self.moved.emit(0.0, 0.0)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = self._radius()
        p.setBrush(QColor(30, 30, 36))
        p.setPen(QPen(QColor(70, 70, 80), 2))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setPen(QPen(QColor(60, 60, 70), 1, Qt.DashLine))
        p.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
        p.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))
        kx = cx + self._knob.x() * r
        ky = cy - self._knob.y() * r
        p.setBrush(QColor(80, 160, 240) if self._active else QColor(110, 110, 130))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(kx, ky), 14, 14)


class XYCoverageWidget(QWidget):
    """
    Top-down heatmap of sampled base (x, y) positions in odom.  Auto-fits the
    extent to the samples seen so far; cells colour blue -> red with density.
    """
    EMPTY = QColor(38, 38, 44)
    GRID = QColor(20, 20, 24)

    def __init__(self, cols=12, rows=12, cap=3, parent=None):
        super().__init__(parent)
        self.cols = cols
        self.rows = rows
        self.cap = cap
        self.counts = numpy.zeros((rows, cols))
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_positions(self, positions):
        """positions: list of (x, y, yaw); only x, y used here."""
        self.counts = numpy.zeros((self.rows, self.cols))
        if positions:
            xs = numpy.array([p[0] for p in positions])
            ys = numpy.array([p[1] for p in positions])
            # Pad the bounding box so points don't land on the very edge.
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            pad = max(0.25, 0.1 * max(xmax - xmin, ymax - ymin))
            xmin, xmax = xmin - pad, xmax + pad
            ymin, ymax = ymin - pad, ymax + pad
            for x, y in zip(xs, ys):
                c = int((x - xmin) / (xmax - xmin) * self.cols) if xmax > xmin else 0
                # y up: row 0 at top, so larger y -> smaller row index.
                r = int((ymax - y) / (ymax - ymin) * self.rows) if ymax > ymin else 0
                c = min(max(c, 0), self.cols - 1)
                r = min(max(r, 0), self.rows - 1)
                self.counts[r, c] += 1
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        cw, ch = w / self.cols, h / self.rows
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.counts[r, c]
                color = self.EMPTY if v <= 0 else heat_qcolor(min(1.0, v / self.cap))
                p.fillRect(QRectF(c * cw, r * ch, cw, ch), color)
        p.setPen(self.GRID)
        for c in range(self.cols + 1):
            p.drawLine(int(c * cw), 0, int(c * cw), h)
        for r in range(self.rows + 1):
            p.drawLine(0, int(r * ch), w, int(r * ch))


class RosBridge(QObject):
    """Marshals data from ROS worker threads onto the Qt GUI thread."""
    frame_ready = Signal(object)


class CarEyeGui(QMainWindow):
    YAW_BINS = 16

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.bridge = RosBridge()
        self._last_frame = None

        self.setWindowTitle("Car-Eye Calibration")
        self._build_ui()

        self.node.display_callback = self.bridge.frame_ready.emit
        self.bridge.frame_ready.connect(self.on_frame)

        # Joystick state + a steady publish timer (acts as a soft deadman:
        # released joystick re-centres and we keep publishing zeros).
        self._joy = (0.0, 0.0)
        self._cmd_timer = QTimer(self)
        self._cmd_timer.timeout.connect(self._send_cmd)
        self._cmd_timer.start(50)   # 20 Hz

    # -- UI construction ---------------------------------------------------- #
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # Left column: live detection view.
        view_box = QGroupBox("Live detection view")
        view_layout = QVBoxLayout(view_box)
        self.image_label = QLabel("waiting for image...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 360)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view_layout.addWidget(self.image_label)
        root.addWidget(view_box, 2)

        # Middle column: coverage + live status.
        left = QVBoxLayout()

        cov_box = QGroupBox("Coverage")
        cov_layout = QVBoxLayout(cov_box)
        cov_layout.addWidget(QLabel("Base position in odom (top-down X-Y)"))
        self.xy_cov = XYCoverageWidget()
        cov_layout.addWidget(self.xy_cov, 1)
        cov_layout.addWidget(QLabel("Heading (yaw) coverage"))
        self.yaw_cov = HeatmapWidget(self.YAW_BINS, 1, cap=3)
        cov_layout.addWidget(self.yaw_cov)
        left.addWidget(cov_box, 1)

        status_box = QGroupBox("Live")
        status_outer = QVBoxLayout(status_box)
        # Status sits outside the form grid with a fixed height and a width it
        # never asks to grow, so long/short messages can't reflow the panel.
        self.lbl_status = QLabel("waiting for detections...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.lbl_status.setFixedHeight(2 * self.lbl_status.fontMetrics().height() + 6)
        self.lbl_status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        status_outer.addWidget(self.lbl_status)

        status_form = QFormLayout()
        self.lbl_tags = QLabel("0")
        self.lbl_pose = QLabel("-")
        self.lbl_grid_t = QLabel("-")
        self.lbl_grid_rpy = QLabel("-")
        self.lbl_range = QLabel("-")
        self.lbl_reproj = QLabel("-")
        self.lbl_samples = QLabel("0")
        status_form.addRow("Tags in view:", self.lbl_tags)
        status_form.addRow("Base x,y,yaw:", self.lbl_pose)
        status_form.addRow("Grid xyz (cam, m):", self.lbl_grid_t)
        status_form.addRow("Grid rpy (cam, deg):", self.lbl_grid_rpy)
        status_form.addRow("Grid range (m):", self.lbl_range)
        status_form.addRow("Reproj RMS (px):", self.lbl_reproj)
        status_form.addRow("Samples:", self.lbl_samples)
        status_outer.addLayout(status_form)
        left.addWidget(status_box)

        root.addLayout(left, 1)

        # Right column: joystick + actions + result.
        right = QVBoxLayout()

        joy_box = QGroupBox("Drive (cmd_vel)")
        joy_layout = QVBoxLayout(joy_box)
        self.joystick = JoystickWidget()
        self.joystick.moved.connect(self._on_joystick)
        joy_layout.addWidget(self.joystick, 1)
        speed_form = QFormLayout()
        self.max_linear = QDoubleSpinBox()
        self.max_linear.setRange(0.0, 5.0)
        self.max_linear.setSingleStep(0.05)
        self.max_linear.setValue(0.3)
        self.max_linear.setSuffix(" m/s")
        self.max_angular = QDoubleSpinBox()
        self.max_angular.setRange(0.0, 6.28)
        self.max_angular.setSingleStep(0.1)
        self.max_angular.setValue(0.8)
        self.max_angular.setSuffix(" rad/s")
        speed_form.addRow("Max linear:", self.max_linear)
        speed_form.addRow("Max angular:", self.max_angular)
        joy_layout.addLayout(speed_form)
        right.addWidget(joy_box, 1)

        action_box = QGroupBox("Calibration")
        action_layout = QVBoxLayout(action_box)
        self.btn_sample = QPushButton("TAKE SAMPLE")
        self.btn_sample.clicked.connect(self.on_take_sample)
        self.btn_remove = QPushButton("Remove last")
        self.btn_remove.clicked.connect(self.on_remove_sample)
        self.btn_calibrate = QPushButton("CALIBRATE")
        self.btn_calibrate.clicked.connect(self.on_calibrate)
        self.btn_save = QPushButton("SAVE")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save.setEnabled(False)
        action_layout.addWidget(self.btn_sample)
        action_layout.addWidget(self.btn_remove)
        action_layout.addWidget(self.btn_calibrate)
        action_layout.addWidget(self.btn_save)
        self.lbl_result = QLabel("No calibration yet.")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_result.setFrameShape(QFrame.Box)
        action_layout.addWidget(self.lbl_result, 1)
        right.addWidget(action_box, 1)

        root.addLayout(right, 1)
        self.setCentralWidget(central)
        self.statusBar()
        self._result = None

    # -- joystick / cmd_vel ------------------------------------------------- #
    def _on_joystick(self, jx, jy):
        self._joy = (jx, jy)

    def _send_cmd(self):
        jx, jy = self._joy
        linear = jy * self.max_linear.value()
        angular = -jx * self.max_angular.value()   # left = positive yaw
        self.node.publish_cmd(linear, angular)

    # -- detection frames --------------------------------------------------- #
    def on_frame(self, frame):
        self._last_frame = frame
        if frame.image is not None:
            self._show_image(frame.image)
        if frame.status:
            self.lbl_status.setText(frame.status)
        else:
            self.lbl_status.setText("OK - ready to sample" if frame.have_pose
                                    else "tracking...")
        self.lbl_tags.setText(str(frame.num_tags))
        if frame.have_grid:
            self.lbl_grid_t.setText("%.3f, %.3f, %.3f" % frame.grid_t)
            self.lbl_grid_rpy.setText("%.1f, %.1f, %.1f"
                                      % tuple(math.degrees(a) for a in frame.grid_rpy))
            self.lbl_range.setText("%.2f" % frame.range)
            self.lbl_reproj.setText("%.2f" % frame.reproj)
        if frame.have_pose:
            self.lbl_pose.setText("%.2f, %.2f, %.0f deg"
                                  % (frame.base_x, frame.base_y,
                                     math.degrees(frame.base_yaw)))
        self.btn_sample.setEnabled(frame.have_pose)

    def _show_image(self, bgr):
        qimg = numpy_to_qimage(bgr)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    def _refresh_coverage(self):
        positions = self.node.sample_positions()
        self.xy_cov.set_positions(positions)
        yaw_counts = numpy.zeros((1, self.YAW_BINS))
        for _, _, yaw in positions:
            b = int((yaw + math.pi) / (2 * math.pi) * self.YAW_BINS)
            b = min(max(b, 0), self.YAW_BINS - 1)
            yaw_counts[0, b] += 1
        self.yaw_cov.set_counts(yaw_counts)
        self.lbl_samples.setText(str(len(positions)))

    # -- actions ------------------------------------------------------------ #
    def on_take_sample(self):
        n = self.node.take_sample()
        if n < 0:
            self.statusBar().showMessage("No sample-able pose right now.", 3000)
            return
        self._refresh_coverage()
        self.statusBar().showMessage("Took sample %d." % n, 2000)

    def on_remove_sample(self):
        self.node.remove_last_sample()
        self._refresh_coverage()

    def on_calibrate(self):
        n = len(self.node.samples)
        if n < MIN_SAMPLES:
            self.statusBar().showMessage(
                "Need at least %d samples (have %d)." % (MIN_SAMPLES, n), 4000)
            return
        try:
            X = solve_hand_eye(self.node.samples, algorithm='Park')
            res = compute_residual(self.node.samples, X)
        except Exception as e:   # noqa: BLE001 - surface any solver failure
            self.lbl_result.setText("Calibration failed: %s" % e)
            return
        self._result = X
        (tx, ty, tz), (qx, qy, qz, qw) = matrix_to_transform_tuple(X)
        rpy = numpy.degrees(_mat_to_rpy(X))
        warn = ""
        if rotation_axis_rank(self.node.samples) < 2:
            warn = ("\n⚠ motion rotates about one axis only (near-planar); "
                    "result is unreliable — add pitch/roll or tilt the grid.")
        self.lbl_result.setText(
            "base_link -> %s\n"
            "xyz:  %.4f  %.4f  %.4f  (m)\n"
            "rpy:  %.2f  %.2f  %.2f  (deg)\n"
            "quat: %.4f %.4f %.4f %.4f\n"
            "residual: %.4f m / %.2f deg  (%d samples)%s"
            % (self.node.camera_frame(), tx, ty, tz, rpy[0], rpy[1], rpy[2],
               qx, qy, qz, qw, res['translation_rms'], res['rotation_rms'],
               n, warn))
        self.btn_save.setEnabled(True)

    def on_save(self):
        if self._result is None:
            return
        path = save_calibration(self.node, self._result)
        self.statusBar().showMessage("Saved to %s" % path, 5000)

    def closeEvent(self, event):
        self._cmd_timer.stop()
        try:
            self.node.stop()
        except Exception:
            pass
        super().closeEvent(event)


def _mat_to_rpy(X):
    import transforms3d as tfs
    return numpy.array(tfs.euler.mat2euler(X[:3, :3], axes='sxyz'))


def save_calibration(node, X):
    """Write the extrinsic as a YAML file under /tmp and return the path."""
    import os
    (tx, ty, tz), (qx, qy, qz, qw) = matrix_to_transform_tuple(X)
    rpy = _mat_to_rpy(X).tolist()
    path = os.path.join('/tmp', 'careye_calibration.yaml')
    with open(path, 'w') as f:
        f.write("# car-eye calibration: base_link -> camera extrinsic\n")
        f.write("parent_frame: %s\n" % node._base_frame)
        f.write("child_frame: %s\n" % node.camera_frame())
        f.write("translation: {x: %.6f, y: %.6f, z: %.6f}\n" % (tx, ty, tz))
        f.write("rotation: {x: %.6f, y: %.6f, z: %.6f, w: %.6f}\n"
                % (qx, qy, qz, qw))
        f.write("rpy_deg: {roll: %.4f, pitch: %.4f, yaw: %.4f}\n"
                % (math.degrees(rpy[0]), math.degrees(rpy[1]),
                   math.degrees(rpy[2])))
        f.write("# static_transform_publisher args:\n")
        f.write("# %.6f %.6f %.6f %.6f %.6f %.6f %.6f %s %s\n"
                % (tx, ty, tz, qx, qy, qz, qw, node._base_frame,
                   node.camera_frame()))
    return path


def run_gui(node):
    """Create the Qt application, spin ROS in the background, and run the GUI."""
    from camera_calibration_apriltag.careye_calibrator import SpinThread

    app = QApplication.instance() or QApplication([])
    gui = CarEyeGui(node)
    gui.resize(820, 620)
    gui.show()

    spin = SpinThread(node)
    spin.daemon = True
    spin.start()

    try:
        app.exec()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
