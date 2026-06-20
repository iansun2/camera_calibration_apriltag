# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
A lightweight, dependency-free 3D viewer for the car-eye manual-calibration
page.  It draws an X-Y ground grid and a set of named coordinate frames
(triads) given as 4x4 homogeneous poses in the world frame (``odom``).

Rendering uses plain ``QPainter`` with a hand-rolled look-at + perspective
projection, so it needs no OpenGL / pyqtgraph.  Left-drag orbits, right/middle
drag pans, the wheel zooms.
"""

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


# X red, Y green, Z blue.
AXIS_COLORS = (QColor(235, 70, 70), QColor(70, 205, 90), QColor(80, 130, 245))
_BG = QColor(26, 27, 32)
_GRID = QColor(60, 62, 70)
_GRID_AXIS = QColor(110, 113, 125)
_NEAR = 0.05   # near-plane (m); geometry closer than this is skipped


class Viewer3D(QWidget):
    """Orbit viewer drawing an X-Y grid and named coordinate frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

        # Orbit camera state.
        self.az = 50.0           # azimuth (deg)
        self.el = 28.0           # elevation (deg)
        self.dist = 3.0          # distance to target (m)
        self.target = np.zeros(3)

        # Scene parameters.
        self.grid_half = 2.0     # half-extent of the ground grid (m)
        self.grid_step = 0.1     # grid cell size (m) = 10 cm
        self.axis_len = 0.25     # triad axis length (m)
        self.fov_deg = 50.0

        # live: list of (pose4x4, label, scale); pins: list of same lists.
        self.live = []
        self.pins = []

        self._last_mouse = None
        self._mode = None
        # Filled per-paint.
        self._R = np.eye(3)
        self._t = np.zeros(3)
        self._focal = 500.0

    # -- scene API ---------------------------------------------------------- #
    def set_live(self, frames):
        """frames: list of (4x4 pose, label, scale)."""
        self.live = frames
        self.update()

    def set_pins(self, pins):
        """pins: list of frame-lists (each like ``set_live``'s argument)."""
        self.pins = pins
        self.update()

    # -- interaction -------------------------------------------------------- #
    def mousePressEvent(self, event):
        self._last_mouse = event.position()
        self._mode = 'rotate' if event.button() == Qt.LeftButton else 'pan'

    def mouseReleaseEvent(self, _event):
        self._last_mouse = None
        self._mode = None

    def mouseMoveEvent(self, event):
        if self._last_mouse is None:
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        if self._mode == 'rotate':
            self.az = (self.az - delta.x() * 0.4) % 360.0
            self.el = max(-89.0, min(89.0, self.el + delta.y() * 0.4))
        else:
            right, up = self._view_basis()
            scale = self.dist * 0.0015
            self.target += (-right * delta.x() + up * delta.y()) * scale
        self.update()

    def wheelEvent(self, event):
        step = 0.9 if event.angleDelta().y() > 0 else 1.0 / 0.9
        self.dist = max(0.2, min(50.0, self.dist * step))
        self.update()

    # -- camera math -------------------------------------------------------- #
    def _eye(self):
        elr, azr = math.radians(self.el), math.radians(self.az)
        d = np.array([math.cos(elr) * math.cos(azr),
                      math.cos(elr) * math.sin(azr),
                      math.sin(elr)])
        return self.target + self.dist * d

    def _view_basis(self):
        """Return (right, up) world vectors of the current view plane."""
        eye = self._eye()
        f = self.target - eye
        f /= np.linalg.norm(f)
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(s) < 1e-6:
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        return s, u

    def _setup_camera(self):
        eye = self._eye()
        f = self.target - eye
        f /= np.linalg.norm(f)
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(s) < 1e-6:
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        self._R = np.vstack([s, u, -f])     # world -> camera rotation
        self._t = -self._R @ eye
        h = min(self.width(), self.height())
        self._focal = (0.5 * h) / math.tan(math.radians(self.fov_deg) / 2.0)

    def _project(self, pts):
        """pts (N,3) world -> (proj (N,2) screen, depth (N,))."""
        cam = (self._R @ pts.T).T + self._t
        depth = -cam[:, 2]
        z = np.maximum(depth, 1e-3)
        x = self._focal * cam[:, 0] / z + self.width() / 2.0
        y = -self._focal * cam[:, 1] / z + self.height() / 2.0
        return np.column_stack([x, y]), depth

    # -- rendering ---------------------------------------------------------- #
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), _BG)
        self._setup_camera()
        self._draw_grid(p)
        for frames in self.pins:
            self._draw_frames(p, frames, alpha=110, width=2)
        self._draw_frames(p, self.live, alpha=255, width=3)
        self._draw_legend(p)

    def _draw_grid(self, p):
        n = int(round(self.grid_half / self.grid_step))
        ext = n * self.grid_step
        starts, ends, axis_flags = [], [], []
        for i in range(-n, n + 1):
            c = i * self.grid_step
            starts.append([c, -ext, 0.0])
            ends.append([c, ext, 0.0])
            axis_flags.append(i == 0)
            starts.append([-ext, c, 0.0])
            ends.append([ext, c, 0.0])
            axis_flags.append(i == 0)
        s2, sd = self._project(np.array(starts))
        e2, ed = self._project(np.array(ends))
        for k in range(len(starts)):
            if sd[k] <= _NEAR or ed[k] <= _NEAR:
                continue
            p.setPen(QPen(_GRID_AXIS if axis_flags[k] else _GRID, 1))
            p.drawLine(QPointF(s2[k, 0], s2[k, 1]), QPointF(e2[k, 0], e2[k, 1]))

    def _draw_frames(self, p, frames, alpha, width):
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        for pose, label, scale in frames:
            if pose is None:
                continue
            o = pose[:3, 3]
            length = self.axis_len * scale
            ends = np.array([o,
                             o + length * pose[:3, 0],
                             o + length * pose[:3, 1],
                             o + length * pose[:3, 2]])
            proj, depth = self._project(ends)
            if depth[0] <= _NEAR:
                continue
            for k in range(3):
                if depth[k + 1] <= _NEAR:
                    continue
                col = QColor(AXIS_COLORS[k])
                col.setAlpha(alpha)
                p.setPen(QPen(col, width))
                p.drawLine(QPointF(proj[0, 0], proj[0, 1]),
                           QPointF(proj[k + 1, 0], proj[k + 1, 1]))
            if label:
                p.setPen(QColor(225, 225, 230, alpha))
                p.drawText(QPointF(proj[0, 0] + 5, proj[0, 1] - 5), label)

    def _draw_legend(self, p):
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        p.setPen(QColor(150, 150, 160))
        p.drawText(8, self.height() - 8,
                   "grid 10 cm   |   drag: orbit   right-drag: pan   wheel: zoom")
