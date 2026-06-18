#!/usr/bin/env python
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2009, Willow Garage, Inc.
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

"""
Camera calibration from AprilTag detections.

Unlike the original ``camera_calibration`` package, this module does **not**
detect a calibration target in the image itself.  Detection is delegated to an
external AprilTag detector node which publishes
``apriltag_msgs/msg/AprilTagDetectionArray`` messages.  This module consumes
those detections, matches each tag to its known 3D position on an
:class:`ApriltagBoard`, and runs the OpenCV calibration solver.
"""

from io import BytesIO
import cv2
import cv_bridge
import glob
import math
import numpy
import numpy.linalg
import os
import sensor_msgs.msg
import tarfile
import time
from enum import Enum

# Minimum number of tags a loaded image must contain to be used as a sample.
MIN_LOAD_TAGS = 4


# Supported camera models
class CAMERA_MODEL(Enum):
    PINHOLE = 0
    FISHEYE = 1


class CalibrationException(Exception):
    pass


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _calculate_skew(corners):
    """
    Get skew for a quadrilateral defined by its outside corners.
    Scaled to [0, 1]: 0 = no skew, 1 = high skew.
    """
    up_left, up_right, down_right, _ = corners

    def angle(a, b, c):
        ab = a - b
        cb = c - b
        denom = numpy.linalg.norm(ab) * numpy.linalg.norm(cb)
        if denom == 0:
            return math.pi / 2.   # degenerate -> treat as no skew
        return math.acos(numpy.clip(numpy.dot(ab, cb) / denom, -1.0, 1.0))

    skew = min(1.0, 2. * abs((math.pi / 2.) - angle(up_left, up_right, down_right)))
    return skew


def _calculate_area(corners):
    """
    Get 2d image area of the detected target as a convex quadrilateral,
    computed as |p X q| / 2.
    """
    (up_left, up_right, down_right, down_left) = corners
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return abs(p[0] * q[1] - p[1] * q[0]) / 2.


def _get_outside_corners(points):
    """
    Given an (N, 2) array of image points, return four extreme corners
    approximating the enclosing quadrilateral as
    (up_left, up_right, down_right, down_left).
    """
    s = points[:, 0] + points[:, 1]
    d = points[:, 0] - points[:, 1]
    up_left = points[numpy.argmin(s)]
    down_right = points[numpy.argmax(s)]
    up_right = points[numpy.argmax(d)]
    down_left = points[numpy.argmin(d)]
    return (up_left, up_right, down_right, down_left)


def _get_dist_model(dist_params, cam_model):
    if CAMERA_MODEL.PINHOLE == cam_model:
        if dist_params.size > 5:
            dist_model = "rational_polynomial"
        else:
            dist_model = "plumb_bob"
    elif CAMERA_MODEL.FISHEYE == cam_model:
        dist_model = "equidistant"
    else:
        dist_model = "unknown"
    return dist_model


def lmin(seq1, seq2):
    return [min(a, b) for (a, b) in zip(seq1, seq2)]


def lmax(seq1, seq2):
    return [max(a, b) for (a, b) in zip(seq1, seq2)]


# --------------------------------------------------------------------------- #
# Detection parsing
# --------------------------------------------------------------------------- #
def tags_to_dict(tags_msg, board=None):
    """
    Convert an ``AprilTagDetectionArray`` message into a dict mapping
    ``tag id -> (4, 2) float64 array`` of image corners (in detector order).

    If ``board`` is given, detections whose id is not part of the board (or
    whose family does not match a non-empty ``board.family``) are dropped.
    """
    out = {}
    for det in tags_msg.detections:
        if board is not None:
            if not board.contains(det.id):
                continue
            if board.family and det.family and board.family != det.family:
                continue
        corners = numpy.array([[c.x, c.y] for c in det.corners], dtype=numpy.float64)
        out[det.id] = corners
    return out


# --------------------------------------------------------------------------- #
# Drawables passed back to the GUI
# --------------------------------------------------------------------------- #
class ImageDrawable():
    def __init__(self):
        self.params = None


class MonoDrawable(ImageDrawable):
    def __init__(self):
        ImageDrawable.__init__(self)
        self.scrib = None
        self.linear_error = -1.0
        self.num_tags = 0


class StereoDrawable(ImageDrawable):
    def __init__(self):
        ImageDrawable.__init__(self)
        self.lscrib = None
        self.rscrib = None
        self.epierror = -1
        self.num_tags = 0


# Candidate per-tag corner orderings, tried at calibration time so the result
# is independent of the detector's corner convention. The first 4 are the cyclic
# rotations, the last 4 add a flip (reflection). The correct one yields by far
# the lowest reprojection error; the wrong ones are non-rigid and fit badly.
CORNER_PERMS = [
    (0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
    (0, 3, 2, 1), (3, 2, 1, 0), (2, 1, 0, 3), (1, 0, 3, 2),
]


def permute_objpoints(opts, perm):
    """Reorder object points within each group of 4 (one tag) by ``perm``."""
    return opts.reshape(-1, 4, 3)[:, perm, :].reshape(-1, 1, 3)


# --------------------------------------------------------------------------- #
# Calibrator base class
# --------------------------------------------------------------------------- #
class Calibrator():
    """Base class for the AprilTag-based calibration system."""

    def __init__(self, board, flags=0, fisheye_flags=0, name='',
                 max_chessboard_speed=-1.0, min_tags=1, require_all_tags=True):
        self.board = board
        self.calibrated = False
        self.calib_flags = flags
        self.fisheye_calib_flags = fisheye_flags
        self.br = cv_bridge.CvBridge()
        self.camera_model = CAMERA_MODEL.PINHOLE
        # self.db holds (params, ...) samples; params = [X, Y, size, skew] in [0,1]
        self.db = []
        # For each sample we record the matched (image_points, object_points, ids)
        self.good_corners = []
        self.goodenough = False
        self.param_ranges = [0.7, 0.7, 0.4, 0.5]
        self.name = name
        self.last_frame_tags = None
        self.max_chessboard_speed = max_chessboard_speed
        self.min_tags = max(1, int(min_tags))
        # Only accept a view as a sample when the full board is detected, so
        # every collected sample contributes all tags to the solver.
        self.require_all_tags = require_all_tags
        self.expected_tags = board.num_tags
        # RMS reprojection error reported by the solver after calibration.
        self.calibration_rms = None
        # Per-tag corner permutation chosen automatically at calibration time.
        self.corner_perm = None
        self.size = None

    def select_corner_perm(self, opts, ipts):
        """
        Find the per-tag corner ordering that best matches the detector output.

        Runs a fast pinhole calibration for each candidate permutation and
        returns the one with the lowest RMS reprojection error.  This makes the
        result independent of whichever corner convention the AprilTag detector
        uses.
        """
        best_perm, best_rms = CORNER_PERMS[0], float('inf')
        for perm in CORNER_PERMS:
            opts_p = [permute_objpoints(o, perm) for o in opts]
            try:
                rms, _, _, _, _ = cv2.calibrateCamera(
                    opts_p, ipts, self.size, numpy.eye(3), None,
                    flags=cv2.CALIB_FIX_K3)
            except cv2.error:
                continue
            if rms < best_rms:
                best_perm, best_rms = perm, rms
        print("auto-selected corner order %s (RMS=%.4f px)" % (best_perm, best_rms))
        return best_perm

    def enough_tags_for_sample(self, num_tags):
        """True if a view with ``num_tags`` matched tags may become a sample."""
        if self.require_all_tags:
            return num_tags >= self.expected_tags
        return num_tags >= self.min_tags

    # -- image conversion --------------------------------------------------- #
    def mkgray(self, msg):
        """Convert a sensor_msgs/Image into an 8-bit single channel image."""
        if self.br.encoding_to_dtype_with_channels(msg.encoding)[0] in ['uint16', 'int16']:
            mono16 = self.br.imgmsg_to_cv2(msg, '16UC1')
            return numpy.array(mono16 / 256, dtype=numpy.uint8)
        elif 'FC1' in msg.encoding:
            img = self.br.imgmsg_to_cv2(msg, "passthrough")
            _, max_val, _, _ = cv2.minMaxLoc(img)
            if max_val > 0:
                return (img * (255.0 / max_val)).astype(numpy.uint8)
            return img.astype(numpy.uint8)
        else:
            return self.br.imgmsg_to_cv2(msg, "mono8")

    def set_cammodel(self, modeltype):
        self.camera_model = modeltype

    # -- correspondences ---------------------------------------------------- #
    def make_correspondences(self, tags):
        """
        Given a dict ``tag id -> (4, 2)`` corners, return
        ``(image_points, object_points, ids)`` where image_points is
        (4N, 1, 2) float32, object_points is (4N, 1, 3) float32 and ids is the
        sorted list of matched tag ids.  Returns ``(None, None, [])`` when fewer
        than ``min_tags`` tags match the board.
        """
        ipts = []
        opts = []
        ids = []
        for tag_id in sorted(tags.keys()):
            objp = self.board.object_points_for_tag(tag_id)
            if objp is None:
                continue
            ids.append(tag_id)
            for k in range(4):
                ipts.append(tags[tag_id][k])
                opts.append(objp[k])
        if len(ids) < self.min_tags:
            return (None, None, [])
        image_points = numpy.array(ipts, dtype=numpy.float32).reshape(-1, 1, 2)
        object_points = numpy.array(opts, dtype=numpy.float32).reshape(-1, 1, 3)
        return (image_points, object_points, ids)

    # -- sample selection heuristics ---------------------------------------- #
    def get_parameters(self, image_points, size):
        """Return [X, Y, size, skew] describing the board view, in [0, 1]."""
        (width, height) = size
        pts = image_points.reshape(-1, 2)
        Xs = pts[:, 0]
        Ys = pts[:, 1]
        outside_corners = _get_outside_corners(pts)
        area = _calculate_area(outside_corners)
        skew = _calculate_skew(outside_corners)
        border = math.sqrt(area)
        p_x = min(1.0, max(0.0, (numpy.mean(Xs) - border / 2) / (width - border)))
        p_y = min(1.0, max(0.0, (numpy.mean(Ys) - border / 2) / (height - border)))
        p_size = math.sqrt(area / (width * height))
        return [p_x, p_y, p_size, skew]

    def is_slow_moving(self, tags, last_frame_tags):
        """True if the tags moved little between this frame and the previous one."""
        if not last_frame_tags:
            return False
        deltas = []
        for tag_id, corners in tags.items():
            if tag_id in last_frame_tags:
                deltas.append(corners - last_frame_tags[tag_id])
        if not deltas:
            return False
        deltas = numpy.concatenate(deltas)
        average_motion = numpy.average(numpy.linalg.norm(deltas, axis=1))
        return average_motion <= self.max_chessboard_speed

    def is_good_sample(self, params, tags, last_frame_tags):
        """True if the view described by params should be added to the database."""
        if not self.db:
            return True

        def param_distance(p1, p2):
            return sum([abs(a - b) for (a, b) in zip(p1, p2)])

        db_params = [sample[0] for sample in self.db]
        d = min([param_distance(params, p) for p in db_params])
        if d <= 0.2:
            return False
        if self.max_chessboard_speed > 0:
            if not self.is_slow_moving(tags, last_frame_tags):
                return False
        return True

    _param_names = ["X", "Y", "Size", "Skew"]

    def compute_goodenough(self):
        if not self.db:
            return None
        all_params = [sample[0] for sample in self.db]
        min_params = all_params[0]
        max_params = all_params[0]
        for params in all_params[1:]:
            min_params = lmin(min_params, params)
            max_params = lmax(max_params, params)
        # Don't reward small size or skew
        min_params = [min_params[0], min_params[1], 0., 0.]
        progress = [min((hi - lo) / r, 1.0)
                    for (lo, hi, r) in zip(min_params, max_params, self.param_ranges)]
        self.goodenough = (len(self.db) >= 40) or all([p == 1.0 for p in progress])
        return list(zip(self._param_names, min_params, max_params, progress))

    # -- display helpers ---------------------------------------------------- #
    @staticmethod
    def downsample(img):
        """Scale ``img`` to roughly VGA. Returns (scrib, x_scale, y_scale)."""
        height, width = img.shape[0], img.shape[1]
        scale = math.sqrt((width * height) / (640. * 480.))
        if scale > 1.0:
            scrib = cv2.resize(img, (int(width / scale), int(height / scale)))
        else:
            scrib = img
        x_scale = float(width) / scrib.shape[1]
        y_scale = float(height) / scrib.shape[0]
        return (scrib, x_scale, y_scale)

    @staticmethod
    def draw_tags(scrib, tags, x_scale, y_scale, color=(0, 255, 0)):
        """Draw detected tag outlines and ids on a (downsampled) BGR image."""
        for tag_id, corners in tags.items():
            pts = corners.copy()
            pts[:, 0] /= x_scale
            pts[:, 1] /= y_scale
            ipts = numpy.round(pts).astype(numpy.int32)
            cv2.polylines(scrib, [ipts.reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
            # mark corner 0 (lower-left) to show orientation
            cv2.circle(scrib, tuple(ipts[0]), 3, (0, 0, 255), -1)
            centre = tuple(numpy.round(pts.mean(axis=0)).astype(numpy.int32))
            cv2.putText(scrib, str(tag_id), centre, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # -- camera info (de)serialisation -------------------------------------- #
    @staticmethod
    def lrmsg(d, k, r, p, size, camera_model):
        msg = sensor_msgs.msg.CameraInfo()
        msg.width, msg.height = size
        msg.distortion_model = _get_dist_model(d, camera_model)
        msg.d = numpy.ravel(d).copy().tolist()
        msg.k = numpy.ravel(k).copy().tolist()
        msg.r = numpy.ravel(r).copy().tolist()
        msg.p = numpy.ravel(p).copy().tolist()
        return msg

    @staticmethod
    def lrreport(d, k, r, p):
        print("D =", numpy.ravel(d).tolist())
        print("K =", numpy.ravel(k).tolist())
        print("R =", numpy.ravel(r).tolist())
        print("P =", numpy.ravel(p).tolist())

    @staticmethod
    def lrost(name, d, k, r, p, size):
        assert k.shape == (3, 3)
        assert r.shape == (3, 3)
        assert p.shape == (3, 4)
        calmessage = "\n".join([
            "# oST version 5.0 parameters", "", "",
            "[image]", "",
            "width", "%d" % size[0], "",
            "height", "%d" % size[1], "",
            "[%s]" % name, "",
            "camera matrix",
            " ".join("%8f" % k[0, i] for i in range(3)),
            " ".join("%8f" % k[1, i] for i in range(3)),
            " ".join("%8f" % k[2, i] for i in range(3)), "",
            "distortion",
            " ".join("%8f" % x for x in d.flat), "",
            "rectification",
            " ".join("%8f" % r[0, i] for i in range(3)),
            " ".join("%8f" % r[1, i] for i in range(3)),
            " ".join("%8f" % r[2, i] for i in range(3)), "",
            "projection",
            " ".join("%8f" % p[0, i] for i in range(4)),
            " ".join("%8f" % p[1, i] for i in range(4)),
            " ".join("%8f" % p[2, i] for i in range(4)), ""
        ])
        assert len(calmessage) < 525, "Calibration info must be less than 525 bytes"
        return calmessage

    @staticmethod
    def lryaml(name, d, k, r, p, size, cam_model):
        def format_mat(x, precision):
            return ("[%s]" % (
                numpy.array2string(x, precision=precision, suppress_small=True, separator=", ")
                    .replace("[", "").replace("]", "").replace("\n", "\n        ")))

        dist_model = _get_dist_model(d, cam_model)
        assert k.shape == (3, 3)
        assert r.shape == (3, 3)
        assert p.shape == (3, 4)
        calmessage = "\n".join([
            "image_width: %d" % size[0],
            "image_height: %d" % size[1],
            "camera_name: " + name,
            "camera_matrix:",
            "  rows: 3", "  cols: 3",
            "  data: " + format_mat(k, 5),
            "distortion_model: " + dist_model,
            "distortion_coefficients:",
            "  rows: 1", "  cols: %d" % d.size,
            "  data: [%s]" % ", ".join("%8f" % x for x in d.flat),
            "rectification_matrix:",
            "  rows: 3", "  cols: 3",
            "  data: " + format_mat(r, 8),
            "projection_matrix:",
            "  rows: 3", "  cols: 4",
            "  data: " + format_mat(p, 5),
            ""
        ])
        return calmessage

    def do_save(self):
        filename = '/tmp/calibrationdata.tar.gz'
        tf = tarfile.open(filename, 'w:gz')
        self.do_tarfile_save(tf)  # Must be overridden in subclasses
        tf.close()
        print("Wrote calibration data to", filename)


# --------------------------------------------------------------------------- #
# Monocular calibrator
# --------------------------------------------------------------------------- #
class MonoCalibrator(Calibrator):
    """Calibration class for monocular cameras using AprilTag detections."""

    is_mono = True

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'narrow_stereo/left'
        super(MonoCalibrator, self).__init__(*args, **kwargs)

    def cal_fromcorners(self, good):
        """:param good: list of (image_points, object_points, ids)."""
        (ipts, opts, _ids) = zip(*good)
        intrinsics_in = numpy.eye(3, dtype=numpy.float64)
        opts = list(opts)
        ipts = list(ipts)

        # Pick the corner ordering that matches the detector, then apply it.
        self.corner_perm = self.select_corner_perm(opts, ipts)
        opts = [permute_objpoints(o, self.corner_perm) for o in opts]

        if self.camera_model == CAMERA_MODEL.PINHOLE:
            print("mono pinhole calibration...")
            reproj_err, self.intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                opts, ipts, self.size, intrinsics_in, None, flags=self.calib_flags)
            if self.calib_flags & cv2.CALIB_RATIONAL_MODEL:
                self.distortion = dist_coeffs.flat[:8].reshape(-1, 1)
            else:
                self.distortion = dist_coeffs.flat[:5].reshape(-1, 1)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            print("mono fisheye calibration...")
            opts = numpy.asarray(opts, dtype=numpy.float64)
            ipts = numpy.asarray(ipts, dtype=numpy.float64)
            reproj_err, self.intrinsics, self.distortion, rvecs, tvecs = cv2.fisheye.calibrate(
                opts, ipts, self.size, intrinsics_in, None, flags=self.fisheye_calib_flags)

        self.calibration_rms = float(reproj_err)
        print("mono calibration RMS reprojection error: %.4f px" % self.calibration_rms)

        # R is identity for monocular calibration
        self.R = numpy.eye(3, dtype=numpy.float64)
        self.P = numpy.zeros((3, 4), dtype=numpy.float64)
        self.set_alpha(0.0)

    def set_alpha(self, a):
        """Set the zoom (alpha) for the rectified output, 0 = cropped, 1 = full."""
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            ncm, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics, self.distortion, self.size, a)
            for j in range(3):
                for i in range(3):
                    self.P[j, i] = ncm[j, i]
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.intrinsics, self.distortion, self.R, ncm, self.size, cv2.CV_32FC1)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            self.P[:3, :3] = self.intrinsics[:3, :3]
            self.P[0, 0] /= (1. + a)
            self.P[1, 1] /= (1. + a)
            self.mapx, self.mapy = cv2.fisheye.initUndistortRectifyMap(
                self.intrinsics, self.distortion, self.R, self.P, self.size, cv2.CV_32FC1)

    def remap(self, src):
        return cv2.remap(src, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_points(self, src):
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            return cv2.undistortPoints(src, self.intrinsics, self.distortion, R=self.R, P=self.P)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            return cv2.fisheye.undistortPoints(src, self.intrinsics, self.distortion, R=self.R, P=self.P)

    def as_message(self):
        return self.lrmsg(self.distortion, self.intrinsics, self.R, self.P, self.size, self.camera_model)

    def from_message(self, msg, alpha=0.0):
        self.size = (msg.width, msg.height)
        self.intrinsics = numpy.array(msg.k, dtype=numpy.float64, copy=True).reshape((3, 3))
        self.distortion = numpy.array(msg.d, dtype=numpy.float64, copy=True).reshape((len(msg.d), 1))
        self.R = numpy.array(msg.r, dtype=numpy.float64, copy=True).reshape((3, 3))
        self.P = numpy.array(msg.p, dtype=numpy.float64, copy=True).reshape((3, 4))
        self.set_alpha(0.0)

    def report(self):
        self.lrreport(self.distortion, self.intrinsics, self.R, self.P)

    def ost(self):
        return self.lrost(self.name, self.distortion, self.intrinsics, self.R, self.P, self.size)

    def yaml(self):
        return self.lryaml(self.name, self.distortion, self.intrinsics, self.R, self.P, self.size, self.camera_model)

    def reprojection_error(self, image_points, object_points):
        """RMS reprojection error in pixels for a single view, using solvePnP."""
        if image_points is None or len(image_points) < 4:
            return None
        if self.corner_perm is not None:
            object_points = permute_objpoints(object_points, self.corner_perm)
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                          self.intrinsics, self.distortion)
            if not ok:
                return None
            projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                             self.intrinsics, self.distortion)
            err = numpy.linalg.norm(
                projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1)
            return float(numpy.sqrt(numpy.mean(err ** 2)))
        except cv2.error:
            return None

    def handle_msg(self, msgs):
        """
        :param msgs: tuple of (image_msg, tags_msg)

        Process one synchronized image/detection pair, adding it to the sample
        database if useful, and return a :class:`MonoDrawable` for display.
        """
        (img_msg, tags_msg) = msgs
        gray = self.mkgray(img_msg)
        if self.size is None:
            self.size = (gray.shape[1], gray.shape[0])
        linear_error = -1

        tags = tags_to_dict(tags_msg, self.board)
        image_points, object_points, ids = self.make_correspondences(tags)
        scrib_mono, x_scale, y_scale = self.downsample(gray)

        if self.calibrated:
            gray_remap = self.remap(gray)
            gray_rect = gray_remap
            if x_scale != 1.0 or y_scale != 1.0:
                gray_rect = cv2.resize(gray_remap, (scrib_mono.shape[1], scrib_mono.shape[0]))
            scrib = cv2.cvtColor(gray_rect, cv2.COLOR_GRAY2BGR)

            if image_points is not None:
                linear_error = self.reprojection_error(image_points, object_points)
                undistorted = self.undistort_points(image_points)
                # Re-pack undistorted corners into a per-tag dict for drawing
                undist_tags = {}
                for i, tag_id in enumerate(ids):
                    undist_tags[tag_id] = undistorted[4 * i:4 * i + 4].reshape(4, 2)
                self.draw_tags(scrib, undist_tags, x_scale, y_scale, color=(0, 255, 0))
        else:
            scrib = cv2.cvtColor(scrib_mono, cv2.COLOR_GRAY2BGR)
            if tags:
                self.draw_tags(scrib, tags, x_scale, y_scale, color=(0, 255, 0))
            if image_points is not None and self.enough_tags_for_sample(len(ids)):
                params = self.get_parameters(image_points, (gray.shape[1], gray.shape[0]))
                if self.is_good_sample(params, tags, self.last_frame_tags):
                    self.db.append((params, gray))
                    self.good_corners.append((image_points, object_points, ids))
                    print("*** Added sample %d, p_x = %.3f, p_y = %.3f, "
                          "p_size = %.3f, skew = %.3f"
                          % tuple([len(self.db)] + params))

        self.last_frame_tags = tags
        rv = MonoDrawable()
        rv.scrib = scrib
        rv.params = self.compute_goodenough()
        rv.linear_error = linear_error
        rv.num_tags = len(ids)
        return rv

    def do_calibration(self, dump=False):
        if not self.good_corners:
            raise CalibrationException("No samples collected, cannot calibrate")
        if self.db:
            self.size = (self.db[0][1].shape[1], self.db[0][1].shape[0])
        self.cal_fromcorners(self.good_corners)
        self.calibrated = True
        self.report()
        print(self.ost())

    def add_images(self, directory):
        """
        Detect tags in every image in ``directory`` and add them as samples.

        Tags are detected with OpenCV's AprilTag detector (the live detector
        node is not involved here).  Returns the number of images added.
        """
        from camera_calibration_apriltag.tag_detection import detect_tags, make_detector
        files = sorted(glob.glob(os.path.join(directory, '*.png')) +
                       glob.glob(os.path.join(directory, '*.jpg')))
        det = make_detector(self.board.family or '36h11')
        added = 0
        for f in files:
            gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            if self.size is None:
                self.size = (gray.shape[1], gray.shape[0])
            tags = {i: c for i, c in detect_tags(gray, detector=det).items()
                    if self.board.contains(i)}
            image_points, object_points, ids = self.make_correspondences(tags)
            if image_points is None or len(ids) < MIN_LOAD_TAGS:
                continue
            params = self.get_parameters(image_points, (gray.shape[1], gray.shape[0]))
            self.db.append((params, gray))
            self.good_corners.append((image_points, object_points, ids))
            added += 1
        self.compute_goodenough()
        print("Loaded %d images from %s (%d total samples)" % (added, directory, len(self.db)))
        return added

    def do_tarfile_save(self, tf):
        def taradd(name, buf):
            s = BytesIO(buf.encode('utf-8') if isinstance(buf, str) else buf)
            ti = tarfile.TarInfo(name)
            ti.size = len(s.getvalue())
            ti.uname = 'calibrator'
            ti.mtime = int(time.time())
            tf.addfile(tarinfo=ti, fileobj=s)

        for i, (_, im) in enumerate(self.db):
            taradd("left-%04d.png" % i, cv2.imencode(".png", im)[1].tobytes())
        taradd('ost.yaml', self.yaml())
        taradd('ost.txt', self.ost())


# --------------------------------------------------------------------------- #
# Stereo calibrator
# --------------------------------------------------------------------------- #
class StereoCalibrator(Calibrator):
    """Calibration class for stereo cameras using AprilTag detections."""

    is_mono = False

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'narrow_stereo'
        super(StereoCalibrator, self).__init__(*args, **kwargs)
        self.l = MonoCalibrator(*args, **kwargs)
        self.r = MonoCalibrator(*args, **kwargs)
        # Horizontal stereo rig can't get full X range in the left camera.
        self.param_ranges[0] = 0.4

    def set_cammodel(self, modeltype):
        super(StereoCalibrator, self).set_cammodel(modeltype)
        self.l.set_cammodel(modeltype)
        self.r.set_cammodel(modeltype)

    def match_stereo(self, ltags, rtags):
        """
        Build matched object/image points for tags seen by *both* cameras.

        Returns (lipts, ripts, opts, ids) as float32 arrays, or
        (None, None, None, []) if too few tags are shared.
        """
        common = sorted(set(ltags.keys()) & set(rtags.keys()))
        common = [tid for tid in common if self.board.contains(tid)]
        if len(common) < self.min_tags:
            return (None, None, None, [])
        lipts, ripts, opts = [], [], []
        for tid in common:
            objp = self.board.object_points_for_tag(tid)
            for k in range(4):
                lipts.append(ltags[tid][k])
                ripts.append(rtags[tid][k])
                opts.append(objp[k])
        return (numpy.array(lipts, dtype=numpy.float32).reshape(-1, 1, 2),
                numpy.array(ripts, dtype=numpy.float32).reshape(-1, 1, 2),
                numpy.array(opts, dtype=numpy.float32).reshape(-1, 1, 3),
                common)

    def cal_fromcorners(self, good):
        """:param good: list of (lipts, ripts, opts, ids)."""
        lcorners = [(li, op, ids) for (li, ri, op, ids) in good]
        rcorners = [(ri, op, ids) for (li, ri, op, ids) in good]
        self.l.size = self.size
        self.r.size = self.size
        self.l.cal_fromcorners(lcorners)
        self.r.cal_fromcorners(rcorners)

        (lipts, ripts, opts, _ids) = zip(*good)
        opts = list(opts)
        lipts = list(lipts)
        ripts = list(ripts)

        # Match the detector's corner ordering for the stereo solve too.
        self.corner_perm = self.select_corner_perm(opts, lipts)
        opts = [permute_objpoints(o, self.corner_perm) for o in opts]

        self.T = numpy.zeros((3, 1), dtype=numpy.float64)
        self.R = numpy.eye(3, dtype=numpy.float64)
        flags = cv2.CALIB_FIX_INTRINSIC

        if self.camera_model == CAMERA_MODEL.PINHOLE:
            print("stereo pinhole calibration...")
            result = cv2.stereoCalibrate(opts, lipts, ripts,
                                         self.l.intrinsics, self.l.distortion,
                                         self.r.intrinsics, self.r.distortion,
                                         self.size, self.R, self.T,
                                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5),
                                         flags=flags)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            print("stereo fisheye calibration...")
            lipts = numpy.asarray(lipts, dtype=numpy.float64)
            ripts = numpy.asarray(ripts, dtype=numpy.float64)
            opts = numpy.asarray(opts, dtype=numpy.float64)
            result = cv2.fisheye.stereoCalibrate(opts, lipts, ripts,
                                                 self.l.intrinsics, self.l.distortion,
                                                 self.r.intrinsics, self.r.distortion,
                                                 self.size, self.R, self.T,
                                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5),
                                                 flags=flags)
        # The first return value of (fisheye.)stereoCalibrate is the RMS error.
        self.calibration_rms = float(result[0])
        print("stereo calibration RMS reprojection error: %.4f px" % self.calibration_rms)
        self.set_alpha(0.0)

    def set_alpha(self, a):
        if self.camera_model == CAMERA_MODEL.PINHOLE:
            cv2.stereoRectify(self.l.intrinsics, self.l.distortion,
                              self.r.intrinsics, self.r.distortion,
                              self.size, self.R, self.T,
                              self.l.R, self.r.R, self.l.P, self.r.P,
                              alpha=a)
            self.l.mapx, self.l.mapy = cv2.initUndistortRectifyMap(
                self.l.intrinsics, self.l.distortion, self.l.R, self.l.P, self.size, cv2.CV_32FC1)
            self.r.mapx, self.r.mapy = cv2.initUndistortRectifyMap(
                self.r.intrinsics, self.r.distortion, self.r.R, self.r.P, self.size, cv2.CV_32FC1)
        elif self.camera_model == CAMERA_MODEL.FISHEYE:
            self.Q = numpy.zeros((4, 4), dtype=numpy.float64)
            flags = cv2.CALIB_ZERO_DISPARITY
            cv2.fisheye.stereoRectify(self.l.intrinsics, self.l.distortion,
                                      self.r.intrinsics, self.r.distortion,
                                      self.size, self.R, self.T, flags,
                                      self.l.R, self.r.R, self.l.P, self.r.P,
                                      self.Q, self.size, a, 1.0)
            self.l.P[:3, :3] = numpy.dot(self.l.intrinsics, self.l.R)
            self.r.P[:3, :3] = numpy.dot(self.r.intrinsics, self.r.R)
            self.l.mapx, self.l.mapy = cv2.fisheye.initUndistortRectifyMap(
                self.l.intrinsics, self.l.distortion, self.l.R, self.l.intrinsics, self.size, cv2.CV_32FC1)
            self.r.mapx, self.r.mapy = cv2.fisheye.initUndistortRectifyMap(
                self.r.intrinsics, self.r.distortion, self.r.R, self.r.intrinsics, self.size, cv2.CV_32FC1)

    def as_message(self):
        return (self.lrmsg(self.l.distortion, self.l.intrinsics, self.l.R, self.l.P, self.size, self.l.camera_model),
                self.lrmsg(self.r.distortion, self.r.intrinsics, self.r.R, self.r.P, self.size, self.r.camera_model))

    def from_message(self, msgs, alpha=0.0):
        self.size = (msgs[0].width, msgs[0].height)
        self.T = numpy.zeros((3, 1), dtype=numpy.float64)
        self.R = numpy.eye(3, dtype=numpy.float64)
        self.l.from_message(msgs[0])
        self.r.from_message(msgs[1])

    def report(self):
        print("\nLeft:")
        self.lrreport(self.l.distortion, self.l.intrinsics, self.l.R, self.l.P)
        print("\nRight:")
        self.lrreport(self.r.distortion, self.r.intrinsics, self.r.R, self.r.P)
        print("self.T =", numpy.ravel(self.T).tolist())
        print("self.R =", numpy.ravel(self.R).tolist())

    def ost(self):
        return (self.lrost(self.name + "/left", self.l.distortion, self.l.intrinsics, self.l.R, self.l.P, self.size) +
                self.lrost(self.name + "/right", self.r.distortion, self.r.intrinsics, self.r.R, self.r.P, self.size))

    def yaml(self, suffix, info):
        return self.lryaml(self.name + suffix, info.distortion, info.intrinsics, info.R, info.P, self.size, self.camera_model)

    def epipolar_error(self, lcorners, rcorners):
        d = lcorners[:, :, 1] - rcorners[:, :, 1]
        return numpy.sqrt(numpy.square(d).sum() / d.size)

    def handle_msg(self, msgs):
        """:param msgs: tuple of (limg_msg, ltags_msg, rimg_msg, rtags_msg)."""
        (limg_msg, ltags_msg, rimg_msg, rtags_msg) = msgs
        lgray = self.mkgray(limg_msg)
        rgray = self.mkgray(rimg_msg)
        if self.size is None:
            self.size = (lgray.shape[1], lgray.shape[0])
        epierror = -1

        ltags = tags_to_dict(ltags_msg, self.board)
        rtags = tags_to_dict(rtags_msg, self.board)
        lipts, ripts, opts, common = self.match_stereo(ltags, rtags)

        lscrib_mono, x_scale, y_scale = self.downsample(lgray)
        rscrib_mono, _, _ = self.downsample(rgray)

        if self.calibrated:
            lrect = self.l.remap(lgray)
            rrect = self.r.remap(rgray)
            if x_scale != 1.0 or y_scale != 1.0:
                lrect = cv2.resize(lrect, (lscrib_mono.shape[1], lscrib_mono.shape[0]))
                rrect = cv2.resize(rrect, (rscrib_mono.shape[1], rscrib_mono.shape[0]))
            lscrib = cv2.cvtColor(lrect, cv2.COLOR_GRAY2BGR)
            rscrib = cv2.cvtColor(rrect, cv2.COLOR_GRAY2BGR)

            if lipts is not None:
                lundist = self.l.undistort_points(lipts)
                rundist = self.r.undistort_points(ripts)
                ldict, rdict = {}, {}
                for i, tid in enumerate(common):
                    ldict[tid] = lundist[4 * i:4 * i + 4].reshape(4, 2)
                    rdict[tid] = rundist[4 * i:4 * i + 4].reshape(4, 2)
                self.draw_tags(lscrib, ldict, x_scale, y_scale)
                self.draw_tags(rscrib, rdict, x_scale, y_scale)
                epierror = self.epipolar_error(lundist, rundist)
        else:
            lscrib = cv2.cvtColor(lscrib_mono, cv2.COLOR_GRAY2BGR)
            rscrib = cv2.cvtColor(rscrib_mono, cv2.COLOR_GRAY2BGR)
            if ltags:
                self.draw_tags(lscrib, ltags, x_scale, y_scale)
            if rtags:
                self.draw_tags(rscrib, rtags, x_scale, y_scale)
            if lipts is not None and self.enough_tags_for_sample(len(common)):
                params = self.get_parameters(lipts, (lgray.shape[1], lgray.shape[0]))
                if self.is_good_sample(params, ltags, self.last_frame_tags):
                    self.db.append((params, lgray, rgray))
                    self.good_corners.append((lipts, ripts, opts, common))
                    print("*** Added sample %d, p_x = %.3f, p_y = %.3f, "
                          "p_size = %.3f, skew = %.3f"
                          % tuple([len(self.db)] + params))

        self.last_frame_tags = ltags
        rv = StereoDrawable()
        rv.lscrib = lscrib
        rv.rscrib = rscrib
        rv.params = self.compute_goodenough()
        rv.epierror = epierror
        rv.num_tags = len(common)
        return rv

    def do_calibration(self, dump=False):
        if not self.good_corners:
            raise CalibrationException("No samples collected, cannot calibrate")
        self.size = (self.db[0][1].shape[1], self.db[0][1].shape[0])
        self.cal_fromcorners(self.good_corners)
        self.calibrated = True
        self.report()
        print(self.ost())

    def add_images(self, directory):
        """
        Load left-*/right-* image pairs from ``directory`` as stereo samples.

        Pairs are matched by filename (``left-0007.png`` <-> ``right-0007.png``).
        Returns the number of pairs added.
        """
        from camera_calibration_apriltag.tag_detection import detect_tags, make_detector
        lefts = sorted(glob.glob(os.path.join(directory, 'left-*.png')) +
                       glob.glob(os.path.join(directory, 'left-*.jpg')))
        det = make_detector(self.board.family or '36h11')
        added = 0
        for lf in lefts:
            rf = lf.replace('left-', 'right-')
            if not os.path.exists(rf):
                continue
            lgray = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
            rgray = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
            if lgray is None or rgray is None:
                continue
            if self.size is None:
                self.size = (lgray.shape[1], lgray.shape[0])
            ltags = {i: c for i, c in detect_tags(lgray, detector=det).items()
                     if self.board.contains(i)}
            rtags = {i: c for i, c in detect_tags(rgray, detector=det).items()
                     if self.board.contains(i)}
            lipts, ripts, opts, common = self.match_stereo(ltags, rtags)
            if lipts is None or len(common) < MIN_LOAD_TAGS:
                continue
            params = self.get_parameters(lipts, (lgray.shape[1], lgray.shape[0]))
            self.db.append((params, lgray, rgray))
            self.good_corners.append((lipts, ripts, opts, common))
            added += 1
        self.compute_goodenough()
        print("Loaded %d stereo pairs from %s (%d total samples)" % (added, directory, len(self.db)))
        return added

    def do_tarfile_save(self, tf):
        def taradd(name, buf):
            s = BytesIO(buf.encode('utf-8') if isinstance(buf, str) else buf)
            ti = tarfile.TarInfo(name)
            ti.size = len(s.getvalue())
            ti.uname = 'calibrator'
            ti.mtime = int(time.time())
            tf.addfile(tarinfo=ti, fileobj=s)

        ims = ([("left-%04d.png" % i, im) for i, (_, im, _) in enumerate(self.db)] +
               [("right-%04d.png" % i, im) for i, (_, _, im) in enumerate(self.db)])
        for (name, im) in ims:
            taradd(name, cv2.imencode(".png", im)[1].tobytes())
        taradd('left.yaml', self.yaml("/left", self.l))
        taradd('right.yaml', self.yaml("/right", self.r))
        taradd('ost.txt', self.ost())
