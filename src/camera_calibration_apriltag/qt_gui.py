# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
PySide6 front-end for AprilTag-based camera calibration.

Replaces the original OpenCV/HighGUI window.  The ROS node runs in a background
spin thread and pushes :class:`MonoDrawable`/:class:`StereoDrawable` objects to
this GUI through Qt signals (which marshal safely onto the GUI thread).
"""

import threading
import time

import cv2
import numpy
import rclpy

from PySide6.QtCore import Qt, QObject, QRectF, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSlider, QHBoxLayout, QVBoxLayout, QGroupBox,
    QFrame, QMessageBox, QFileDialog, QSizePolicy)

from camera_calibration_apriltag.calibrator import CAMERA_MODEL, CalibrationException


def numpy_to_qimage(bgr):
    """Convert a BGR uint8 numpy image into a (copied) QImage."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = numpy.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def heat_qcolor(t):
    """Map t in [0,1] to a jet-like color (blue -> cyan -> green -> yellow -> red)."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        r, g, b = 0.0, t / 0.25, 1.0
    elif t < 0.5:
        r, g, b = 0.0, 1.0, 1.0 - (t - 0.25) / 0.25
    elif t < 0.75:
        r, g, b = (t - 0.5) / 0.25, 1.0, 0.0
    else:
        r, g, b = 1.0, 1.0 - (t - 0.75) / 0.25, 0.0
    return QColor(int(r * 255), int(g * 255), int(b * 255))


class HeatmapWidget(QWidget):
    """
    Grid of cells coloured by a sample count.  Works as a 2D map (rows x cols)
    or a 1D bar (rows == 1).  Empty cells are drawn dark so gaps in coverage
    stand out; covered cells run blue -> red with increasing sample density.
    """
    EMPTY = QColor(38, 38, 44)
    GRID = QColor(20, 20, 24)

    def __init__(self, cols, rows, cap=3, parent=None):
        super().__init__(parent)
        self.cols = cols
        self.rows = rows
        self.cap = cap          # count mapped to the "hot" end of the colormap
        self.counts = numpy.zeros((rows, cols))
        self.setMinimumSize(140, 90 if rows > 1 else 26)
        self.setSizePolicy(QSizePolicy.Expanding,
                           QSizePolicy.Expanding if rows > 1 else QSizePolicy.Fixed)

    def set_counts(self, counts):
        self.counts = counts
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        cw = w / self.cols
        ch = h / self.rows
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
    drawable_ready = Signal(object)
    calibration_finished = Signal(bool, str)
    images_loaded = Signal(int, str)
    status = Signal(str)


class CalibrationGui(QMainWindow):
    # Display ranges for the coverage heatmaps.
    SIZE_RANGE = (0.0, 0.8)     # p_size = sqrt(board area / image area)
    SKEW_RANGE = (0.0, 0.8)     # apparent skew magnitude, 0 = fronto-parallel
    XY_COLS, XY_ROWS = 8, 6     # image-position grid
    BAR_BINS = 14               # bins for the size / skew bars

    def __init__(self, node, stereo=False):
        super().__init__()
        self.node = node
        self.stereo = stereo
        self.bridge = RosBridge()

        # Live FPS estimate (exponential moving average of frame intervals).
        self._last_frame_time = None
        self._fps = 0.0

        self.setWindowTitle("AprilTag Camera Calibration" + (" (stereo)" if stereo else ""))
        self._build_ui()

        # Wire the ROS node's display hook to a Qt signal so updates happen
        # on the GUI thread.
        self.node.display_callback = self.bridge.drawable_ready.emit
        self.bridge.drawable_ready.connect(self.on_drawable)
        self.bridge.calibration_finished.connect(self.on_calibration_finished)
        self.bridge.images_loaded.connect(self.on_images_loaded)
        self.bridge.status.connect(self.statusBar().showMessage)

    # -- UI construction ---------------------------------------------------- #
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # Image display area
        img_box = QHBoxLayout()
        self.left_label = QLabel("Waiting for images...")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setMinimumSize(640, 480)
        self.left_label.setFrameShape(QFrame.Box)
        img_box.addWidget(self.left_label)
        if self.stereo:
            self.right_label = QLabel("Waiting for images...")
            self.right_label.setAlignment(Qt.AlignCenter)
            self.right_label.setMinimumSize(640, 480)
            self.right_label.setFrameShape(QFrame.Box)
            img_box.addWidget(self.right_label)
        root.addLayout(img_box, stretch=1)

        # Control panel
        panel = QVBoxLayout()

        self.fps_label = QLabel("FPS: --")
        self.samples_label = QLabel("Samples: 0")
        self.tags_label = QLabel("Tags in view: 0")
        panel.addWidget(self.fps_label)
        panel.addWidget(self.samples_label)
        panel.addWidget(self.tags_label)

        cov_box = QGroupBox("Coverage")
        cov_layout = QVBoxLayout(cov_box)
        cov_layout.addWidget(QLabel("Image position (X-Y)"))
        self.xy_heat = HeatmapWidget(self.XY_COLS, self.XY_ROWS, cap=3)
        cov_layout.addWidget(self.xy_heat)
        cov_layout.addWidget(QLabel("Board size (small/far ← → large/near)"))
        self.size_heat = HeatmapWidget(self.BAR_BINS, 1, cap=3)
        cov_layout.addWidget(self.size_heat)
        cov_layout.addWidget(QLabel("Skew / tilt (flat ← → steep)"))
        self.skew_heat = HeatmapWidget(self.BAR_BINS, 1, cap=3)
        cov_layout.addWidget(self.skew_heat)
        panel.addWidget(cov_box)

        self.error_label = QLabel("Reprojection error: --")
        panel.addWidget(self.error_label)

        self.rms_label = QLabel("Final calibration RMS: --")
        panel.addWidget(self.rms_label)

        model_box = QGroupBox("Camera model")
        model_layout = QVBoxLayout(model_box)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Pinhole", "Fisheye"])
        self.model_combo.currentIndexChanged.connect(self.on_model_change)
        model_layout.addWidget(self.model_combo)
        panel.addWidget(model_box)

        scale_box = QGroupBox("Scale (alpha)")
        scale_layout = QVBoxLayout(scale_box)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(0, 100)
        self.scale_slider.setValue(0)
        self.scale_slider.setEnabled(False)
        self.scale_slider.valueChanged.connect(self.on_scale)
        scale_layout.addWidget(self.scale_slider)
        panel.addWidget(scale_box)

        self.load_btn = QPushButton("LOAD IMAGES...")
        self.load_btn.clicked.connect(self.on_load_images)
        panel.addWidget(self.load_btn)

        self.calibrate_btn = QPushButton("CALIBRATE")
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        panel.addWidget(self.calibrate_btn)

        self.save_btn = QPushButton("SAVE")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.on_save)
        panel.addWidget(self.save_btn)

        self.commit_btn = QPushButton("COMMIT")
        self.commit_btn.setEnabled(False)
        self.commit_btn.clicked.connect(self.on_commit)
        panel.addWidget(self.commit_btn)

        panel.addStretch(1)
        root.addLayout(panel)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    # -- display update ----------------------------------------------------- #
    def _set_image(self, label, bgr):
        qimg = numpy_to_qimage(bgr)
        pix = QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def _update_fps(self):
        now = time.monotonic()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                inst = 1.0 / dt
                # Exponential moving average for a stable readout.
                self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_frame_time = now
        self.fps_label.setText("FPS: %.1f" % self._fps)

    @Slot(object)
    def on_drawable(self, drawable):
        self._update_fps()
        c = self.node.c
        if self.stereo:
            if drawable.lscrib is not None:
                self._set_image(self.left_label, drawable.lscrib)
            if drawable.rscrib is not None:
                self._set_image(self.right_label, drawable.rscrib)
        else:
            if drawable.scrib is not None:
                self._set_image(self.left_label, drawable.scrib)

        self.tags_label.setText("Tags in view: %d" % getattr(drawable, 'num_tags', 0))
        if c is not None:
            self.samples_label.setText("Samples: %d" % len(c.db))

        self.refresh_coverage()

        # Error readout
        if c is not None and c.calibrated:
            if self.stereo:
                err = drawable.epierror
                self.error_label.setText(
                    "Epipolar error: %.3f px" % err if err and err >= 0 else "Epipolar error: --")
            else:
                err = drawable.linear_error
                self.error_label.setText(
                    "Reprojection error: %.3f px" % err if err and err >= 0 else "Reprojection error: --")

        # Button / slider enablement
        if c is not None:
            self.calibrate_btn.setEnabled(c.goodenough and not self._busy())
            self.save_btn.setEnabled(c.calibrated)
            self.commit_btn.setEnabled(c.calibrated and not self._busy())
            self.scale_slider.setEnabled(c.calibrated)

    def refresh_coverage(self):
        """Rebuild the coverage heatmaps from the collected sample parameters."""
        c = self.node.c
        if c is None or not c.db:
            return
        params = numpy.array([s[0] for s in c.db])   # (N, 4): px, py, p_size, skew
        # 2D image-position heatmap
        xy = numpy.zeros((self.XY_ROWS, self.XY_COLS))
        gx = numpy.clip((params[:, 0] * self.XY_COLS).astype(int), 0, self.XY_COLS - 1)
        gy = numpy.clip((params[:, 1] * self.XY_ROWS).astype(int), 0, self.XY_ROWS - 1)
        for x, y in zip(gx, gy):
            xy[y, x] += 1
        self.xy_heat.set_counts(xy)
        # 1D size / skew bars
        self.size_heat.set_counts(self._bin1d(params[:, 2], self.SIZE_RANGE))
        self.skew_heat.set_counts(self._bin1d(params[:, 3], self.SKEW_RANGE))

    def _bin1d(self, values, value_range):
        lo, hi = value_range
        out = numpy.zeros((1, self.BAR_BINS))
        idx = numpy.clip(((values - lo) / (hi - lo) * self.BAR_BINS).astype(int),
                         0, self.BAR_BINS - 1)
        for i in idx:
            out[0, i] += 1
        return out

    def _busy(self):
        return getattr(self, '_worker', None) is not None and self._worker.is_alive()

    # -- button handlers ---------------------------------------------------- #
    def _run_async(self, fn, busy_msg):
        if self._busy():
            return
        self.calibrate_btn.setEnabled(False)
        self.commit_btn.setEnabled(False)
        self.bridge.status.emit(busy_msg)

        def worker():
            ok, msg = True, ""
            try:
                fn()
            except CalibrationException as e:
                ok, msg = False, str(e)
            except Exception as e:  # noqa: BLE001 - surface any solver failure
                ok, msg = False, repr(e)
            self.bridge.calibration_finished.emit(ok, msg)

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def on_calibrate(self):
        self.calibrate_btn.setText("Calibrating...")
        self._run_async(self.node.do_calibration, "Calibrating...")

    def on_commit(self):
        def commit():
            if self.node.do_upload():
                self.bridge.status.emit("Calibration committed to camera driver")
            else:
                raise CalibrationException("Failed to upload calibration (see log)")
        self._run_async(commit, "Uploading calibration...")

    def on_load_images(self):
        if self._busy():
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Select folder of calibration images")
        if not directory:
            return
        self.load_btn.setEnabled(False)
        self.bridge.status.emit("Loading images from %s ..." % directory)

        def worker():
            try:
                n = self.node.load_images(directory)
                self.bridge.images_loaded.emit(n, "")
            except Exception as e:  # noqa: BLE001
                self.bridge.images_loaded.emit(-1, repr(e))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    @Slot(int, str)
    def on_images_loaded(self, n, err):
        self.load_btn.setEnabled(True)
        if n < 0:
            self.bridge.status.emit("Load failed: " + err)
            QMessageBox.warning(self, "Load failed", err)
            return
        c = self.node.c
        self.samples_label.setText("Samples: %d" % (len(c.db) if c else 0))
        self.refresh_coverage()
        if c is not None:
            self.calibrate_btn.setEnabled(c.goodenough)
        self.bridge.status.emit("Loaded %d images" % n)
        QMessageBox.information(self, "Images loaded",
                               "Added %d images. Total samples: %d"
                               % (n, len(c.db) if c else 0))

    def on_save(self):
        try:
            self.node.do_save()
            QMessageBox.information(self, "Saved",
                                   "Calibration data written to /tmp/calibrationdata.tar.gz")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(e))

    @Slot(bool, str)
    def on_calibration_finished(self, ok, msg):
        self.calibrate_btn.setText("CALIBRATE")
        if ok:
            self.bridge.status.emit("Done")
            self.save_btn.setEnabled(True)
            c = self.node.c
            rms = getattr(c, 'calibration_rms', None) if c is not None else None
            if rms is not None:
                self.rms_label.setText("Final calibration RMS: %.4f px" % rms)
        else:
            self.bridge.status.emit("Error: " + msg)
            QMessageBox.warning(self, "Calibration error", msg)

    def on_model_change(self, index):
        model = CAMERA_MODEL.PINHOLE if index == 0 else CAMERA_MODEL.FISHEYE
        self.node.set_camera_model(model)

    def on_scale(self, value):
        self.node.set_scale(value / 100.0)

    def closeEvent(self, event):
        self.node.display_callback = None
        super().closeEvent(event)


def run_gui(node, stereo=False):
    """Create the Qt application, spin ROS in the background, and run the GUI."""
    from camera_calibration_apriltag.camera_calibrator import SpinThread

    app = QApplication.instance() or QApplication([])
    gui = CalibrationGui(node, stereo=stereo)
    gui.resize(1100 if stereo else 760, 620)
    gui.show()

    spin = SpinThread(node)
    spin.daemon = True
    spin.start()

    try:
        app.exec()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
