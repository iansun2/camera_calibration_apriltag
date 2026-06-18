# Software License Agreement (BSD License)
#
# Copyright (c) 2024, The camera_calibration_apriltag authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions of the BSD
# license are met.

"""
ROS node for "car-eye" calibration: the camera-to-base extrinsic of a camera
mounted on a mobile base.

The node consumes ``apriltag_msgs/msg/AprilTagDetectionArray`` detections of a
static AprilGrid, recovers the grid pose in the camera with a single
``solvePnP`` over every detected tag corner, and pairs it with the base pose
read from tf (``odom -> base_link``).  Accumulated samples are solved with
:mod:`camera_calibration_apriltag.careye_hand_eye`.

It also publishes ``geometry_msgs/msg/Twist`` on ``cmd_vel`` so the GUI joystick
can drive the platform around to collect varied views.

The class contains no GUI code; it exposes a ``display_callback`` hook invoked
(from a worker thread) with a :class:`CarEyeFrame` for every processed frame.
"""

import threading

import cv2
import message_filters
import numpy as np
import rclpy
import sensor_msgs.msg
import transforms3d as tfs

from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from rclpy.time import Duration, Time

import tf2_ros
from tf2_ros import Buffer, TransformListener

from camera_calibration_apriltag.careye_hand_eye import transform_to_Rt

try:
    from queue import Queue
except ImportError:  # pragma: no cover
    from Queue import Queue


class SpinThread(threading.Thread):
    """Thread that spins the ROS node so the GUI can own the main thread."""

    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node

    def run(self):
        rclpy.spin(self.node)


class ConsumerThread(threading.Thread):
    """Drains a queue and applies a function to each item until shutdown."""

    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function

    def run(self):
        while rclpy.ok():
            m = self.queue.get()
            self.function(m)


class BufferQueue(Queue):
    """Queue that discards the oldest item when full instead of blocking."""

    def put(self, item, *args, **kwargs):
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class CarEyeFrame:
    """Result of processing one detection message, handed to the GUI."""

    def __init__(self):
        self.num_tags = 0          # tags matched to the board this frame
        self.have_pose = False     # PnP + tf both succeeded -> sample-able
        self.have_grid = False     # grid pose (PnP) available this frame
        self.base_x = 0.0          # base_link position in odom (m)
        self.base_y = 0.0
        self.base_yaw = 0.0        # base_link yaw in odom (rad)
        self.range = 0.0           # camera-to-grid distance (m)
        self.reproj = 0.0          # PnP reprojection error (px)
        self.status = ""           # human-readable reason when not sample-able
        self.image = None          # BGR uint8 image with detection overlay
        # Grid pose in the camera optical frame (camera_T_grid).
        self.grid_t = (0.0, 0.0, 0.0)            # translation x,y,z (m)
        self.grid_rpy = (0.0, 0.0, 0.0)          # roll,pitch,yaw (rad)


class CarEyeCalibrationNode(Node):
    """Accumulates AprilGrid/odometry pairs for car-eye calibration."""

    def __init__(self, board, odom_frame='odom', base_frame='base_link',
                 camera_frame='', min_tags=1, require_all_tags=False,
                 cmd_vel_topic='cmd_vel', tf_timeout=0.2, queue_size=1,
                 approximate=0.05):
        super().__init__('carcalibrator')

        self._board = board
        self._odom_frame = odom_frame
        self._base_frame = base_frame
        # Empty -> taken from the detection/camera_info header frame_id.
        self._camera_frame = camera_frame
        self._min_tags = min_tags
        self._require_all_tags = require_all_tags
        self._tf_timeout = Duration(seconds=tf_timeout)
        self._bridge = CvBridge()
        # Length of the drawn pose axes (meters): roughly half the board.
        self._axis_len = max(board.tag_size,
                             0.5 * max(board.n_cols, board.n_rows) * board.tag_spacing)

        # Hook set by the GUI; called with a CarEyeFrame for every frame.
        self.display_callback = None

        # Camera intrinsics, filled from the first camera_info message.
        self._K = None
        self._D = None
        self._info_frame = ''
        self._info_lock = threading.Lock()

        # Latest sample-able candidate (guarded for the GUI's "take sample").
        self._sample_lock = threading.Lock()
        self._latest = None        # dict(g2b, t2c, x, y, yaw) or None

        # Accumulated samples.  Each is a dict with 'g2b', 't2c' (R, t tuples)
        # and 'x','y','yaw' for the coverage heatmap.
        self.samples = []

        # tf: odom -> base_link.  spin_thread=False so the listener's
        # subscriptions are serviced by the node's own SpinThread; spinning the
        # same node from a second executor would raise.
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10), node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        # Outputs: velocity command for the joystick.
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Inputs.  camera_info latched separately; synchronized image + tags
        # drive processing so the overlay matches the detections.
        self.create_subscription(
            sensor_msgs.msg.CameraInfo, 'camera_info', self._on_camera_info,
            self._qos_for('camera_info'))

        self.q = BufferQueue(queue_size)
        img_sub = message_filters.Subscriber(
            self, sensor_msgs.msg.Image, 'image', qos_profile=self._qos_for('image'))
        tag_sub = message_filters.Subscriber(
            self, AprilTagDetectionArray, 'tags', qos_profile=self._qos_for('tags'))
        if approximate > 0.0:
            self._ts = message_filters.ApproximateTimeSynchronizer(
                [img_sub, tag_sub], 4, approximate)
        else:
            self._ts = message_filters.TimeSynchronizer([img_sub, tag_sub], 4)
        self._ts.registerCallback(lambda img, tags: self.q.put((img, tags)))
        cth = ConsumerThread(self.q, self._handle)
        cth.daemon = True
        cth.start()

    # -- ROS plumbing ------------------------------------------------------- #
    def _qos_for(self, topic_name):
        """Match a topic's advertised QoS, falling back to sensor-data QoS."""
        resolved = self.resolve_topic_name(topic_name)
        info = self.get_publishers_info_by_topic(topic_name=resolved)
        if info:
            qos = info[0].qos_profile
            qos.history = qos_profile_system_default.history
            qos.depth = qos_profile_system_default.depth
            return qos
        return qos_profile_sensor_data

    def _on_camera_info(self, msg):
        with self._info_lock:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
            self._info_frame = msg.header.frame_id

    # -- detection handling ------------------------------------------------- #
    def _grid_pose(self, detections):
        """
        Recover ``camera_T_grid`` from a detection message via solvePnP.

        :returns: dict with ``R``, ``t``, ``rvec``, ``tvec``, ``num_tags``,
            ``reproj`` or None when a pose can't be made.
        """
        with self._info_lock:
            if self._K is None:
                return None
            K, D = self._K, self._D

        obj_pts = []
        img_pts = []
        n = 0
        for det in detections:
            if self._board.family and det.family != self._board.family:
                continue
            board_pts = self._board.object_points_for_tag(det.id)
            if board_pts is None:
                continue
            n += 1
            for k in range(4):
                obj_pts.append(board_pts[k])
                img_pts.append((det.corners[k].x, det.corners[k].y))
        if n < self._min_tags:
            return None
        if self._require_all_tags and n < self._board.num_tags:
            return None

        obj = np.array(obj_pts, dtype=np.float64).reshape(-1, 3)
        img = np.array(img_pts, dtype=np.float64).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        reproj = float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        R, _ = cv2.Rodrigues(rvec)
        return {'R': R, 't': tvec.reshape(3), 'rvec': rvec, 'tvec': tvec,
                'num_tags': n, 'reproj': reproj}

    def _draw_overlay(self, bgr, detections, pose):
        """Draw detected tag outlines/ids and the grid pose axes onto ``bgr``."""
        for det in detections:
            if self._board.family and det.family != self._board.family:
                continue
            if self._board.object_points_for_tag(det.id) is None:
                continue
            pts = np.array([(c.x, c.y) for c in det.corners], dtype=np.int32)
            cv2.polylines(bgr, [pts], True, (0, 255, 0), 2)
            cx = int(det.centre.x)
            cy = int(det.centre.y)
            cv2.putText(bgr, str(det.id), (cx - 6, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                        cv2.LINE_AA)
        if pose is not None:
            with self._info_lock:
                K, D = self._K, self._D
            # RGB axes: X red, Y green, Z blue (out of the board).
            cv2.drawFrameAxes(bgr, K, D, pose['rvec'], pose['tvec'],
                              self._axis_len, 2)

    def _base_pose(self, stamp):
        """Look up ``odom_T_base_link`` at ``stamp``; returns (R, t) or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self._odom_frame, self._base_frame, Time.from_msg(stamp),
                self._tf_timeout)
        except tf2_ros.TransformException:
            # Retry at latest available time; odom is usually high-rate.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self._odom_frame, self._base_frame, Time())
            except tf2_ros.TransformException:
                return None
        return transform_to_Rt(tf.transform)

    def _handle(self, item):
        img_msg, tags_msg = item
        frame = CarEyeFrame()

        try:
            bgr = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:   # noqa: BLE001 - bad encoding shouldn't kill the loop
            self.get_logger().warn('cv_bridge failed: %s' % e, once=True)
            bgr = None

        pose = self._grid_pose(tags_msg.detections)
        if bgr is not None:
            self._draw_overlay(bgr, tags_msg.detections, pose)
            frame.image = bgr

        if pose is None:
            with self._info_lock:
                have_info = self._K is not None
            frame.status = ("no camera_info yet" if not have_info
                            else "no grid (need ≥%d tags)" % self._min_tags)
            self._publish(frame, None)
            return

        R_t2c, t_t2c = pose['R'], pose['t']
        frame.num_tags = pose['num_tags']
        frame.range = float(np.linalg.norm(t_t2c))
        frame.reproj = pose['reproj']
        frame.have_grid = True
        frame.grid_t = (float(t_t2c[0]), float(t_t2c[1]), float(t_t2c[2]))
        frame.grid_rpy = tuple(float(a) for a in
                               tfs.euler.mat2euler(R_t2c, axes='sxyz'))

        base = self._base_pose(tags_msg.header.stamp)
        if base is None:
            frame.status = "no tf %s→%s" % (self._odom_frame, self._base_frame)
            self._publish(frame, None)
            return

        R_g2b, t_g2b = base
        yaw = tfs.euler.mat2euler(R_g2b, axes='sxyz')[2]
        frame.have_pose = True
        frame.base_x = float(t_g2b[0])
        frame.base_y = float(t_g2b[1])
        frame.base_yaw = float(yaw)

        candidate = {
            'g2b': (R_g2b, t_g2b),
            't2c': (R_t2c, t_t2c),
            'x': frame.base_x, 'y': frame.base_y, 'yaw': frame.base_yaw,
        }
        self._publish(frame, candidate)

    def _publish(self, frame, candidate):
        with self._sample_lock:
            self._latest = candidate
        if self.display_callback:
            self.display_callback(frame)

    # -- GUI-facing API ----------------------------------------------------- #
    def take_sample(self):
        """Append the latest sample-able candidate.  Returns the new count or -1."""
        with self._sample_lock:
            cand = self._latest
        if cand is None:
            return -1
        self.samples.append(cand)
        return len(self.samples)

    def remove_last_sample(self):
        if self.samples:
            self.samples.pop()
        return len(self.samples)

    def clear_samples(self):
        self.samples = []

    def sample_positions(self):
        """List of (x, y, yaw) for every accumulated sample (for coverage)."""
        return [(s['x'], s['y'], s['yaw']) for s in self.samples]

    def camera_frame(self):
        """Best guess at the camera optical frame name."""
        return self._camera_frame or self._info_frame or 'camera'

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_cmd(0.0, 0.0)
