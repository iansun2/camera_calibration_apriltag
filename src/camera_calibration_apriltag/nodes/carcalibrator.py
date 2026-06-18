#!/usr/bin/env python
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
Entry point for "car-eye" calibration: the camera-to-base extrinsic of a camera
mounted on a mobile base.

The car (``base_link``) drives around in a static ``odom`` frame while a rigidly
mounted camera observes a static AprilGrid.  Detections come from an external
AprilTag detector (``apriltag_msgs/msg/AprilTagDetectionArray`` on ``tags``);
the base pose is read from tf (``odom -> base_link``).  A PySide6 GUI shows
coverage and a joystick that drives the platform via ``cmd_vel``.

Example::

    ros2 run camera_calibration_apriltag carcalibrator \\
        --size 7x5 --tag-size 0.030 --tag-spacing 0.035 \\
        --odom-frame odom --base-frame base_link \\
        --ros-args -r image:=/camera/image_rect -r tags:=/camera/tags \\
                   -r camera_info:=/camera/camera_info -r cmd_vel:=/cmd_vel
"""

import rclpy

from camera_calibration_apriltag.apriltag_board import ApriltagBoard
from camera_calibration_apriltag.careye_calibrator import CarEyeCalibrationNode
from camera_calibration_apriltag.careye_gui import run_gui


def main():
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(
        "%prog --size COLSxROWS --tag-size METERS --tag-spacing METERS",
        description="AprilTag grid car-eye (camera-to-base) calibration.")

    group = OptionGroup(parser, "AprilTag Board Options")
    group.add_option("-s", "--size", type="string", default="7x5",
                     help="board size as COLSxROWS in tags (default %default)")
    group.add_option("--tag-size", type="float", default=0.030,
                     help="edge length of a tag in meters (default %default)")
    group.add_option("--tag-spacing", type="float", default=0.035,
                     help="centre-to-centre distance between tags in meters (default %default)")
    group.add_option("--start-id", type="int", default=0,
                     help="id of the top-left tag (default %default)")
    group.add_option("--tag-family", type="string", default="",
                     help="tag family to accept; empty accepts any (default any)")
    group.add_option("--min-tags", type="int", default=1,
                     help="minimum number of tags required for a pose (default %default)")
    group.add_option("--require-all-tags", action="store_true", default=False,
                     help="require every tag of the board to be detected for a pose")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Frame Options")
    group.add_option("--odom-frame", type="string", default="odom",
                     help="static world frame (default %default)")
    group.add_option("--base-frame", type="string", default="base_link",
                     help="moving car body frame (default %default)")
    group.add_option("--camera-frame", type="string", default="",
                     help="camera optical frame; empty = from camera_info header")
    group.add_option("--tf-timeout", type="float", default=0.2,
                     help="seconds to wait for the odom->base tf at a stamp (default %default)")
    parser.add_option_group(group)

    group = OptionGroup(parser, "ROS Communication Options")
    group.add_option("--approximate", type="float", default=0.05,
                     help="slop (seconds) when syncing image and tags; 0 = exact "
                     "(default %default)")
    parser.add_option_group(group)

    options, _ = parser.parse_args(rclpy.utilities.remove_ros_args())

    try:
        n_cols, n_rows = (int(c) for c in options.size.split('x'))
    except ValueError:
        parser.error("--size must be of the form COLSxROWS, e.g. 7x5")

    board = ApriltagBoard(n_cols=n_cols, n_rows=n_rows,
                          tag_size=options.tag_size, tag_spacing=options.tag_spacing,
                          start_id=options.start_id, family=options.tag_family)
    print("Car-eye calibrating against %r" % board)

    rclpy.init()
    node = CarEyeCalibrationNode(
        board, odom_frame=options.odom_frame, base_frame=options.base_frame,
        camera_frame=options.camera_frame, min_tags=options.min_tags,
        require_all_tags=options.require_all_tags, tf_timeout=options.tf_timeout,
        approximate=options.approximate)
    run_gui(node)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
