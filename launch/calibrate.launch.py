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
# Brings up the AprilTag detector and the (monocular) AprilTag calibration GUI
# together.  The detector subscribes to the camera image and publishes tag
# detections; the calibrator consumes both and runs the calibration.

import launch
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    camera = LaunchConfig('camera').perform(context)
    image = LaunchConfig('image').perform(context)
    tags = LaunchConfig('tags').perform(context)

    # The detector node runs in the 'camera' namespace, so the fully resolved
    # topics live under /<camera>/...
    image_topic = '/%s/%s' % (camera, image)
    tags_topic = '/%s/%s' % (camera, tags)
    set_info_topic = '/%s/set_camera_info' % camera

    detector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('apriltag_detector'), '/launch/detect.launch.py']),
        launch_arguments={
            'camera': camera,
            'image': image,
            'tags': tags,
            'type': LaunchConfig('type'),
            'tag_family': LaunchConfig('tag_family'),
            'image_transport': LaunchConfig('image_transport'),
            'num_threads': '4',
            # 'max_allowed_hamming_distance': '2'
        }.items(),
    )

    calibrator = Node(
        package='camera_calibration_apriltag',
        executable='cameracalibrator',
        name='cameracalibrator',
        output='screen',
        arguments=[
            '--size', LaunchConfig('size'),
            '--tag-size', LaunchConfig('tag_size'),
            '--tag-spacing', LaunchConfig('tag_spacing'),
            '--start-id', LaunchConfig('start_id'),
            '--camera_name', LaunchConfig('camera_name'),
            '--approximate', LaunchConfig('approximate'),
            '--queue-size', LaunchConfig('queue_size'),
        ],
        remappings=[
            ('image', image_topic),
            ('tags', tags_topic),
            ('camera/set_camera_info', set_info_topic),
        ],
    )

    return [detector, calibrator]


def generate_launch_description():
    return launch.LaunchDescription([
        # Detector / topic wiring
        LaunchArg('camera', default_value='camera', description='camera namespace'),
        LaunchArg('image', default_value='image_raw', description='image topic name'),
        LaunchArg('tags', default_value='tags', description='tag detections topic name'),
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
        # Synchronization
        LaunchArg('approximate', default_value='0.0',
                  description='image/tags sync slop in seconds (0 = exact)'),
        LaunchArg('queue_size', default_value='1', description='input queue size'),
        OpaqueFunction(function=launch_setup),
    ])
