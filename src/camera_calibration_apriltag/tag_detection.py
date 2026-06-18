# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
Offline AprilTag detection for the "load images" feature.

During live calibration the tags come from an external AprilTag detector node.
When loading saved images there is no detector running, so we detect the tags
here with OpenCV's built-in AprilTag (aruco) detector.  The corner ordering of
this detector may differ from the live detector, but that is handled
automatically by the corner-permutation search in
:meth:`Calibrator.select_corner_perm`.
"""

import cv2
import numpy


# Map common AprilTag family names (including the umich "tf" prefix) to the
# OpenCV predefined aruco dictionaries.
_FAMILY_DICTS = {
    '16h5': cv2.aruco.DICT_APRILTAG_16h5,
    '25h9': cv2.aruco.DICT_APRILTAG_25h9,
    '36h10': cv2.aruco.DICT_APRILTAG_36h10,
    '36h11': cv2.aruco.DICT_APRILTAG_36h11,
}


def _dict_id_for_family(family):
    f = (family or '36h11').lower()
    for suffix, dict_id in _FAMILY_DICTS.items():
        if f.endswith(suffix):
            return dict_id
    # Default to the most common family.
    return cv2.aruco.DICT_APRILTAG_36h11


def _make_detector(family):
    dictionary = cv2.aruco.getPredefinedDictionary(_dict_id_for_family(family))
    try:
        return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    except AttributeError:  # older OpenCV
        params = cv2.aruco.DetectorParameters_create()

        class _Compat:
            def detectMarkers(self, img):
                return cv2.aruco.detectMarkers(img, dictionary, parameters=params)
        return _Compat()


def detect_tags(gray, family='36h11', detector=None):
    """
    Detect AprilTags in a grayscale image.

    Returns a dict mapping ``tag id -> (4, 2)`` float array of image corners,
    matching the format produced by
    :func:`camera_calibration_apriltag.calibrator.tags_to_dict`.
    """
    if detector is None:
        detector = _make_detector(family)
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = c.reshape(4, 2).astype(numpy.float64)
    return out


def make_detector(family='36h11'):
    """Return a reusable detector object for the given tag family."""
    return _make_detector(family)
