# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
Geometry of an AprilTag grid calibration target (a.k.a. "AprilGrid").

The target is a regular ``n_cols`` x ``n_rows`` grid of square AprilTags.
Tag ids increase in row-major order starting at ``start_id`` in the top-left
corner and increasing to the right, e.g. for an 8x6 board::

       col 0   1   2   3   4   5   6   7
    row 0   0   1   2   3   4   5   6   7
    row 1   8   9  10  11  12  13  14  15
    ...
    row 5  40  41  42  43  44  45  46  47

Each detected tag contributes its four corners as known 3D object points,
expressed in a right-handed board frame with X to the right, Y downwards and
Z out of the board (so Z = 0 for every corner).  The corner ordering matches
the order produced by the AprilTag detector (counter-clockwise, starting at
the lower-left corner of the tag):

    corner 0: lower-left
    corner 1: lower-right
    corner 2: upper-right
    corner 3: upper-left
"""

import numpy


# Per-tag corner offsets from the tag centre, in (X, Y) with Y pointing down,
# multiplied by half the tag size.  Order matches the AprilTag detector output.
_CORNER_OFFSETS = numpy.array([
    [-1.0,  1.0],   # 0: lower-left
    [ 1.0,  1.0],   # 1: lower-right
    [ 1.0, -1.0],   # 2: upper-right
    [-1.0, -1.0],   # 3: upper-left
], dtype=numpy.float64)


class ApriltagBoard():
    """
    Definition of an AprilTag grid calibration target.

    :param n_cols: number of tags per row
    :param n_rows: number of tag rows
    :param tag_size: edge length of a single tag, in meters
    :param tag_spacing: distance between the centres of two neighbouring tags,
        in meters (centre-to-centre)
    :param start_id: id of the top-left tag
    :param family: tag family string (used only to optionally filter detections)
    """

    def __init__(self, n_cols=8, n_rows=6, tag_size=0.030, tag_spacing=0.03375,
                 start_id=0, family=''):
        if n_cols <= 0 or n_rows <= 0:
            raise ValueError("AprilTag board must have positive dimensions")
        if tag_size <= 0.0:
            raise ValueError("AprilTag size must be positive")
        if tag_spacing < tag_size:
            # Not strictly fatal, but almost always a configuration mistake.
            print("WARNING: tag_spacing (%.4f) < tag_size (%.4f); "
                  "tags would overlap" % (tag_spacing, tag_size))
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.tag_size = float(tag_size)
        self.tag_spacing = float(tag_spacing)
        self.start_id = int(start_id)
        self.family = family

        # Pre-compute the object points for every tag id, keyed by id.
        half = self.tag_size / 2.0
        self._object_points = {}
        for idx in range(self.n_cols * self.n_rows):
            tag_id = self.start_id + idx
            col = idx % self.n_cols
            row = idx // self.n_cols
            cx = col * self.tag_spacing
            cy = row * self.tag_spacing
            pts = numpy.zeros((4, 3), dtype=numpy.float32)
            for k in range(4):
                pts[k, 0] = cx + _CORNER_OFFSETS[k, 0] * half
                pts[k, 1] = cy + _CORNER_OFFSETS[k, 1] * half
                pts[k, 2] = 0.0
            self._object_points[tag_id] = pts

    @property
    def num_tags(self):
        return self.n_cols * self.n_rows

    def contains(self, tag_id):
        """Return True if ``tag_id`` belongs to this board."""
        return tag_id in self._object_points

    def object_points_for_tag(self, tag_id):
        """
        Return the (4, 3) array of object points for ``tag_id`` in the order
        produced by the AprilTag detector, or ``None`` if the id is not part of
        this board.
        """
        return self._object_points.get(tag_id)

    def __repr__(self):
        return ("ApriltagBoard(%dx%d, tag_size=%.4f, tag_spacing=%.4f, "
                "start_id=%d, family=%r)" % (self.n_cols, self.n_rows,
                self.tag_size, self.tag_spacing, self.start_id, self.family))
