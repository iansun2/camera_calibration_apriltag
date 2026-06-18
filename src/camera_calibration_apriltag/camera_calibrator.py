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
ROS node that drives AprilTag-based camera calibration.

The node subscribes to one (monocular) or two (stereo) AprilTag detection
topics published by an external AprilTag detector node, together with the
corresponding image topics for display.  Detections are matched against an
:class:`ApriltagBoard` and accumulated by a :class:`MonoCalibrator` /
:class:`StereoCalibrator`.

This class contains no GUI code; it exposes a ``display_callback`` hook which
is invoked (from a worker thread) with a drawable every time a frame is
processed.  The PySide6 front-end in :mod:`camera_calibration_apriltag.qt_gui`
connects to it.
"""

import message_filters
import rclpy
import sensor_msgs.msg
import sensor_msgs.srv
import threading

from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, QoSProfile

from camera_calibration_apriltag.calibrator import (
    MonoCalibrator, StereoCalibrator)

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


class BufferQueue(Queue):
    """Queue that discards the oldest item when full instead of blocking."""

    def put(self, item, *args, **kwargs):
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class SpinThread(threading.Thread):
    """Thread that spins the ROS node so the GUI can own the main thread."""

    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node

    def run(self):
        rclpy.spin(self.node)


class ConsumerThread(threading.Thread):
    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function

    def run(self):
        while rclpy.ok():
            m = self.queue.get()
            self.function(m)


class CalibrationNode(Node):
    """ROS node accumulating AprilTag detections for calibration."""

    def __init__(self, name, board, stereo=False, service_check=True,
                 synchronizer=message_filters.TimeSynchronizer, flags=0,
                 fisheye_flags=0, camera_name='', max_chessboard_speed=-1,
                 queue_size=1, min_tags=1, require_all_tags=True, max_views=0):
        super().__init__(name)

        self._board = board
        self._stereo = stereo
        self._calib_flags = flags
        self._fisheye_calib_flags = fisheye_flags
        self._camera_name = camera_name
        self._max_chessboard_speed = max_chessboard_speed
        self._min_tags = min_tags
        self._require_all_tags = require_all_tags
        self._max_views = max_views

        # Hook set by the GUI; called with a drawable for every processed frame.
        self.display_callback = None

        self.set_camera_info_service = self.create_client(
            sensor_msgs.srv.SetCameraInfo, "camera/set_camera_info")
        self.set_left_camera_info_service = self.create_client(
            sensor_msgs.srv.SetCameraInfo, "left_camera/set_camera_info")
        self.set_right_camera_info_service = self.create_client(
            sensor_msgs.srv.SetCameraInfo, "right_camera/set_camera_info")

        if service_check:
            services = ([self.set_left_camera_info_service, self.set_right_camera_info_service]
                        if stereo else [self.set_camera_info_service])
            for cli in services:
                self.get_logger().info("Waiting for service %s ..." % cli.srv_name)
                try:
                    cli.wait_for_service(timeout_sec=5)
                except Exception as e:
                    self.get_logger().warn("Service not found: %s" % e)

        self.c = None
        self._last_display = None

        if stereo:
            limg = message_filters.Subscriber(
                self, sensor_msgs.msg.Image, 'left', qos_profile=self.get_topic_qos("left"))
            ltag = message_filters.Subscriber(
                self, AprilTagDetectionArray, 'left_tags', qos_profile=self.get_topic_qos("left_tags"))
            rimg = message_filters.Subscriber(
                self, sensor_msgs.msg.Image, 'right', qos_profile=self.get_topic_qos("right"))
            rtag = message_filters.Subscriber(
                self, AprilTagDetectionArray, 'right_tags', qos_profile=self.get_topic_qos("right_tags"))
            ts = synchronizer([limg, ltag, rimg, rtag], 4)
            ts.registerCallback(self.queue_stereo)
            self.q = BufferQueue(queue_size)
            cth = ConsumerThread(self.q, self.handle_stereo)
        else:
            img = message_filters.Subscriber(
                self, sensor_msgs.msg.Image, 'image', qos_profile=self.get_topic_qos("image"))
            tag = message_filters.Subscriber(
                self, AprilTagDetectionArray, 'tags', qos_profile=self.get_topic_qos("tags"))
            ts = synchronizer([img, tag], 4)
            ts.registerCallback(self.queue_monocular)
            self.q = BufferQueue(queue_size)
            cth = ConsumerThread(self.q, self.handle_monocular)

        self._ts = ts
        cth.daemon = True
        cth.start()

    # -- subscription callbacks --------------------------------------------- #
    def queue_monocular(self, img_msg, tags_msg):
        self.q.put((img_msg, tags_msg))

    def queue_stereo(self, limg, ltag, rimg, rtag):
        self.q.put((limg, ltag, rimg, rtag))

    def _make_calibrator(self, cls):
        kwargs = dict(flags=self._calib_flags, fisheye_flags=self._fisheye_calib_flags,
                      max_chessboard_speed=self._max_chessboard_speed, min_tags=self._min_tags,
                      require_all_tags=self._require_all_tags, max_views=self._max_views)
        if self._camera_name:
            kwargs['name'] = self._camera_name
        return cls(self._board, **kwargs)

    def handle_monocular(self, msg):
        if self.c is None:
            self.c = self._make_calibrator(MonoCalibrator)
        drawable = self.c.handle_msg(msg)
        self._last_display = drawable
        if self.display_callback:
            self.display_callback(drawable)

    def handle_stereo(self, msg):
        if self.c is None:
            self.c = self._make_calibrator(StereoCalibrator)
        drawable = self.c.handle_msg(msg)
        self._last_display = drawable
        if self.display_callback:
            self.display_callback(drawable)

    # -- calibration actions ------------------------------------------------ #
    def load_images(self, directory):
        """Load saved calibration images from a directory and add them as samples."""
        if self.c is None:
            self.c = self._make_calibrator(StereoCalibrator if self._stereo else MonoCalibrator)
        return self.c.add_images(directory)

    def do_calibration(self):
        self.c.do_calibration()

    def do_save(self):
        self.c.do_save()

    def check_set_camera_info(self, response):
        if response is not None and response.success:
            return True
        self.get_logger().error("Failed to set camera info: %s"
                                % (response.status_message if response else "no response"))
        return False

    def do_upload(self):
        self.c.report()
        print(self.c.ost())
        info = self.c.as_message()
        req = sensor_msgs.srv.SetCameraInfo.Request()
        rv = True
        if self.c.is_mono:
            req.camera_info = info
            rv = self.check_set_camera_info(self.set_camera_info_service.call(req))
        else:
            req.camera_info = info[0]
            rv = self.check_set_camera_info(self.set_left_camera_info_service.call(req))
            req.camera_info = info[1]
            rv = rv and self.check_set_camera_info(self.set_right_camera_info_service.call(req))
        return rv

    def set_camera_model(self, model):
        if self.c is not None:
            self.c.set_cammodel(model)

    def set_scale(self, alpha):
        if self.c is not None and self.c.calibrated:
            self.c.set_alpha(alpha)

    def get_topic_qos(self, topic_name: str) -> QoSProfile:
        """Return the QoS profile a topic is being published with, if any."""
        topic_name = self.resolve_topic_name(topic_name)
        topic_info = self.get_publishers_info_by_topic(topic_name=topic_name)
        if len(topic_info):
            qos_profile = topic_info[0].qos_profile
            qos_profile.history = qos_profile_system_default.history
            qos_profile.depth = qos_profile_system_default.depth
            return qos_profile
        self.get_logger().warn(
            "No publishers for topic %s. Using system default QoS." % topic_name)
        return qos_profile_system_default
