camera_calibration_apriltag
===========================

The camera_calibration_apriltag package contains a user-friendly calibration
tool, cameracalibrator.  Unlike the original camera_calibration package, target
detection is delegated to an external AprilTag detector node which publishes
``apriltag_msgs/msg/AprilTagDetectionArray`` messages; this tool consumes those
detections, matches them to a known AprilTag grid (AprilGrid), and runs the
OpenCV calibration solver behind a PySide6 GUI.  The following Python classes
hide some of the complexity of OpenCV's calibration process and of constructing
a ROS CameraInfo message.  They are documented here for people who need to
extend or make a new calibration tool.

For details on the camera model and camera calibration process, see
http://docs.opencv.org/master/d9/d0c/group__calib3d.html

.. autoclass:: camera_calibration_apriltag.calibrator.MonoCalibrator
    :members:

.. autoclass:: camera_calibration_apriltag.calibrator.StereoCalibrator
    :members:
