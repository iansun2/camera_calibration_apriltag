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

import cv2
import numpy
import rclpy

from PySide6.QtCore import Qt, QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QProgressBar,
    QComboBox, QSlider, QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox,
    QFrame, QMessageBox)

from camera_calibration_apriltag.calibrator import CAMERA_MODEL, CalibrationException


def numpy_to_qimage(bgr):
    """Convert a BGR uint8 numpy image into a (copied) QImage."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = numpy.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class RosBridge(QObject):
    """Marshals data from ROS worker threads onto the Qt GUI thread."""
    drawable_ready = Signal(object)
    calibration_finished = Signal(bool, str)
    status = Signal(str)


class CalibrationGui(QMainWindow):
    PARAM_NAMES = ["X", "Y", "Size", "Skew"]

    def __init__(self, node, stereo=False):
        super().__init__()
        self.node = node
        self.stereo = stereo
        self.bridge = RosBridge()

        self.setWindowTitle("AprilTag Camera Calibration" + (" (stereo)" if stereo else ""))
        self._build_ui()

        # Wire the ROS node's display hook to a Qt signal so updates happen
        # on the GUI thread.
        self.node.display_callback = self.bridge.drawable_ready.emit
        self.bridge.drawable_ready.connect(self.on_drawable)
        self.bridge.calibration_finished.connect(self.on_calibration_finished)
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

        self.samples_label = QLabel("Samples: 0")
        self.tags_label = QLabel("Tags in view: 0")
        panel.addWidget(self.samples_label)
        panel.addWidget(self.tags_label)

        cov_box = QGroupBox("Coverage")
        cov_grid = QGridLayout(cov_box)
        self.bars = {}
        for i, name in enumerate(self.PARAM_NAMES):
            cov_grid.addWidget(QLabel(name), i, 0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            self.bars[name] = bar
            cov_grid.addWidget(bar, i, 1)
        panel.addWidget(cov_box)

        self.error_label = QLabel("Reprojection error: --")
        panel.addWidget(self.error_label)

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

    @Slot(object)
    def on_drawable(self, drawable):
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

        # Coverage bars
        if drawable.params:
            for (name, _lo, _hi, progress) in drawable.params:
                if name in self.bars:
                    self.bars[name].setValue(int(progress * 100))

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
        self._run_async(self.node.do_calibration, "Calibrating...")

    def on_commit(self):
        def commit():
            if self.node.do_upload():
                self.bridge.status.emit("Calibration committed to camera driver")
            else:
                raise CalibrationException("Failed to upload calibration (see log)")
        self._run_async(commit, "Uploading calibration...")

    def on_save(self):
        try:
            self.node.do_save()
            QMessageBox.information(self, "Saved",
                                   "Calibration data written to /tmp/calibrationdata.tar.gz")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(e))

    @Slot(bool, str)
    def on_calibration_finished(self, ok, msg):
        if ok:
            self.bridge.status.emit("Done")
            self.save_btn.setEnabled(True)
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
