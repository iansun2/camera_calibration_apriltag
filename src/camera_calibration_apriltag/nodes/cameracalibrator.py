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

"""
Entry point for AprilTag-based camera calibration.

Detection is performed by a separate AprilTag detector node which publishes
``apriltag_msgs/msg/AprilTagDetectionArray`` on the ``tags`` topic (and
``left_tags`` / ``right_tags`` for stereo).  This node consumes those
detections together with the matching image topic(s) and runs the calibration
through a PySide6 GUI.

Example (monocular)::

    ros2 run camera_calibration_apriltag cameracalibrator \\
        --size 8x6 --tag-size 0.030 --tag-spacing 0.03375 \\
        --ros-args -r image:=/camera/image_raw -r tags:=/camera/tags
"""

import cv2
import functools
import message_filters
import rclpy

from message_filters import ApproximateTimeSynchronizer

from camera_calibration_apriltag.apriltag_board import ApriltagBoard
from camera_calibration_apriltag.camera_calibrator import CalibrationNode
from camera_calibration_apriltag.qt_gui import run_gui


def main():
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(
        "%prog --size COLSxROWS --tag-size METERS --tag-spacing METERS",
        description="AprilTag grid camera calibration.")
    parser.add_option("-c", "--camera_name", type="string", default='narrow_stereo',
                      help="name of the camera to appear in the calibration file")
    parser.add_option("--stereo", action="store_true", default=False,
                      help="calibrate a stereo pair (left/left_tags, right/right_tags)")

    group = OptionGroup(parser, "AprilTag Board Options")
    group.add_option("-s", "--size", type="string", default="8x6",
                     help="board size as COLSxROWS in tags (default %default)")
    group.add_option("--tag-size", type="float", default=0.030,
                     help="edge length of a tag in meters (default %default)")
    group.add_option("--tag-spacing", type="float", default=0.03375,
                     help="centre-to-centre distance between tags in meters (default %default)")
    group.add_option("--start-id", type="int", default=0,
                     help="id of the top-left tag (default %default)")
    group.add_option("--tag-family", type="string", default="",
                     help="tag family to accept; empty accepts any (default any)")
    group.add_option("--min-tags", type="int", default=1,
                     help="minimum number of tags required to use a view (default %default)")
    parser.add_option_group(group)

    group = OptionGroup(parser, "ROS Communication Options")
    group.add_option("--approximate", type="float", default=0.0,
                     help="slop (seconds) allowed when synchronizing image and tag topics")
    group.add_option("--no-service-check", action="store_false", dest="service_check",
                     default=True, help="disable check for set_camera_info services at startup")
    group.add_option("--queue-size", type="int", default=1,
                     help="input queue size (default %default, 0 for unlimited)")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Calibration Optimizer Options")
    group.add_option("--fix-principal-point", action="store_true", default=False,
                     help="for pinhole, fix the principal point at the image center")
    group.add_option("--fix-aspect-ratio", action="store_true", default=False,
                     help="for pinhole, enforce fx == fy")
    group.add_option("--zero-tangent-dist", action="store_true", default=False,
                     help="for pinhole, set tangential distortion (p1, p2) to zero")
    group.add_option("-k", "--k-coefficients", type="int", default=2, metavar="NUM_COEFFS",
                     help="for pinhole, number of radial distortion coefficients (up to 6, default %default)")
    group.add_option("--fisheye-recompute-extrinsicsts", action="store_true", default=False,
                     help="for fisheye, recompute extrinsics each intrinsic iteration")
    group.add_option("--fisheye-fix-skew", action="store_true", default=False,
                     help="for fisheye, fix skew (alpha) to zero")
    group.add_option("--fisheye-fix-principal-point", action="store_true", default=False,
                     help="for fisheye, fix the principal point at the image center")
    group.add_option("--fisheye-k-coefficients", type="int", default=4, metavar="NUM_COEFFS",
                     help="for fisheye, number of radial distortion coefficients (up to 4, default %default)")
    group.add_option("--fisheye-check-conditions", action="store_true", default=False,
                     help="for fisheye, check validity of condition number")
    group.add_option("--max-chessboard-speed", type="float", default=-1.0,
                     help="reject views where the board moves faster than this (px/frame)")
    parser.add_option_group(group)

    options, _ = parser.parse_args(rclpy.utilities.remove_ros_args())

    try:
        n_cols, n_rows = (int(c) for c in options.size.split('x'))
    except ValueError:
        parser.error("--size must be of the form COLSxROWS, e.g. 8x6")

    board = ApriltagBoard(n_cols=n_cols, n_rows=n_rows,
                          tag_size=options.tag_size, tag_spacing=options.tag_spacing,
                          start_id=options.start_id, family=options.tag_family)
    print("Calibrating against %r" % board)

    if options.approximate == 0.0:
        sync = message_filters.TimeSynchronizer
    else:
        sync = functools.partial(ApproximateTimeSynchronizer, slop=options.approximate)

    # Pinhole calibration flags
    num_ks = options.k_coefficients
    calib_flags = 0
    if options.fix_principal_point:
        calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if options.fix_aspect_ratio:
        calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if options.zero_tangent_dist:
        calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if num_ks > 3:
        calib_flags |= cv2.CALIB_RATIONAL_MODEL
    if num_ks < 6:
        calib_flags |= cv2.CALIB_FIX_K6
    if num_ks < 5:
        calib_flags |= cv2.CALIB_FIX_K5
    if num_ks < 4:
        calib_flags |= cv2.CALIB_FIX_K4
    if num_ks < 3:
        calib_flags |= cv2.CALIB_FIX_K3
    if num_ks < 2:
        calib_flags |= cv2.CALIB_FIX_K2
    if num_ks < 1:
        calib_flags |= cv2.CALIB_FIX_K1

    # Fisheye calibration flags
    num_ks = options.fisheye_k_coefficients
    fisheye_calib_flags = 0
    if options.fisheye_fix_principal_point:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
    if options.fisheye_fix_skew:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_SKEW
    if options.fisheye_recompute_extrinsicsts:
        fisheye_calib_flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    if options.fisheye_check_conditions:
        fisheye_calib_flags |= cv2.fisheye.CALIB_CHECK_COND
    if num_ks < 4:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K4
    if num_ks < 3:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K3
    if num_ks < 2:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K2
    if num_ks < 1:
        fisheye_calib_flags |= cv2.fisheye.CALIB_FIX_K1

    rclpy.init()
    node = CalibrationNode("cameracalibrator", board, stereo=options.stereo,
                           service_check=options.service_check, synchronizer=sync,
                           flags=calib_flags, fisheye_flags=fisheye_calib_flags,
                           camera_name=options.camera_name,
                           max_chessboard_speed=options.max_chessboard_speed,
                           queue_size=options.queue_size, min_tags=options.min_tags)
    run_gui(node, stereo=options.stereo)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
