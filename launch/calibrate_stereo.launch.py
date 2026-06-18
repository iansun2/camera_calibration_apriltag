# -----------------------------------------------------------------------------
# Copyright 2024 The camera_calibration_apriltag authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Brings up two AprilTag detectors (left/right) and the stereo AprilTag
# calibration GUI together.  Each detector subscribes to its camera image and
# publishes tag detections; the calibrator consumes both pairs and runs the
# stereo calibration.

import launch
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _detector(camera, image, tags):
    """Include the AprilTag detector launch in the given camera namespace."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('apriltag_detector'), '/launch/detect.launch.py']),
        launch_arguments={
            'camera': camera,
            'image': image,
            'tags': tags,
            'type': LaunchConfig('type'),
            'tag_family': LaunchConfig('tag_family'),
            'image_transport': LaunchConfig('image_transport'),
        }.items(),
    )


def launch_setup(context, *args, **kwargs):
    left_camera = LaunchConfig('left_camera').perform(context)
    right_camera = LaunchConfig('right_camera').perform(context)
    left_image = LaunchConfig('left_image').perform(context)
    right_image = LaunchConfig('right_image').perform(context)
    tags = LaunchConfig('tags').perform(context)

    # Each detector runs in its own camera namespace, so topics resolve under
    # /<camera>/...
    left_image_topic = '/%s/%s' % (left_camera, left_image)
    left_tags_topic = '/%s/%s' % (left_camera, tags)
    right_image_topic = '/%s/%s' % (right_camera, right_image)
    right_tags_topic = '/%s/%s' % (right_camera, tags)
    left_set_info = '/%s/set_camera_info' % left_camera
    right_set_info = '/%s/set_camera_info' % right_camera

    left_detector = _detector(left_camera, left_image, tags)
    right_detector = _detector(right_camera, right_image, tags)

    calibrator = Node(
        package='camera_calibration_apriltag',
        executable='cameracalibrator',
        name='cameracalibrator',
        output='screen',
        arguments=[
            '--stereo',
            '--size', LaunchConfig('size'),
            '--tag-size', LaunchConfig('tag_size'),
            '--tag-spacing', LaunchConfig('tag_spacing'),
            '--start-id', LaunchConfig('start_id'),
            '--camera_name', LaunchConfig('camera_name'),
            '--approximate', LaunchConfig('approximate'),
            '--queue-size', LaunchConfig('queue_size'),
        ],
        remappings=[
            ('left', left_image_topic),
            ('left_tags', left_tags_topic),
            ('right', right_image_topic),
            ('right_tags', right_tags_topic),
            ('left_camera/set_camera_info', left_set_info),
            ('right_camera/set_camera_info', right_set_info),
        ],
    )

    return [left_detector, right_detector, calibrator]


def generate_launch_description():
    return launch.LaunchDescription([
        # Detector / topic wiring
        LaunchArg('left_camera', default_value='left_camera',
                  description='left camera namespace'),
        LaunchArg('right_camera', default_value='right_camera',
                  description='right camera namespace'),
        LaunchArg('left_image', default_value='image_raw',
                  description='left image topic name (within its namespace)'),
        LaunchArg('right_image', default_value='image_raw',
                  description='right image topic name (within its namespace)'),
        LaunchArg('tags', default_value='tags',
                  description='tag detections topic name (within each namespace)'),
        LaunchArg('type', default_value='umich', description='detector type (umich, mit)'),
        LaunchArg('tag_family', default_value='tf36h11', description='tag family'),
        LaunchArg('image_transport', default_value='raw', description='input image transport'),
        # Calibration board geometry
        LaunchArg('size', default_value='8x6', description='board size as COLSxROWS in tags'),
        LaunchArg('tag_size', default_value='0.030', description='tag edge length (meters)'),
        LaunchArg('tag_spacing', default_value='0.03375',
                  description='tag centre-to-centre distance (meters)'),
        LaunchArg('start_id', default_value='0', description='id of the top-left tag'),
        LaunchArg('camera_name', default_value='narrow_stereo',
                  description='camera name written into the calibration file'),
        # Synchronization (left/right are usually not bit-identical in stamp,
        # so a small slop is enabled by default; set to 0.0 for hardware-synced
        # rigs that assign identical stamps to both images)
        LaunchArg('approximate', default_value='0.05',
                  description='image/tags sync slop in seconds (0 = exact)'),
        LaunchArg('queue_size', default_value='5', description='input queue size'),
        OpaqueFunction(function=launch_setup),
    ])
