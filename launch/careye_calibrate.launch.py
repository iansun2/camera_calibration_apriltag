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
# Brings up the AprilTag detector (apriltag_ros) and the car-eye calibration GUI.
# The detector consumes rectified images + camera_info and publishes tag
# detections; the calibrator pairs the grid pose with the odom->base_link tf
# and solves for the base_link->camera extrinsic.

import launch
from launch.actions import DeclareLaunchArgument as LaunchArg
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LaunchConfig
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    image_rect = LaunchConfig('image_rect').perform(context)
    camera_info = LaunchConfig('camera_info').perform(context)
    tags = LaunchConfig('tags').perform(context)
    cmd_vel = LaunchConfig('cmd_vel').perform(context)
    tag_size = float(LaunchConfig('tag_size').perform(context))

    detector = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag',
        output='screen',
        condition=IfCondition(LaunchConfig('run_detector')),
        parameters=[{
            'family': LaunchConfig('detector_family'),
            'size': tag_size,
        }],
        remappings=[
            ('image_rect', image_rect),
            ('camera_info', camera_info),
            ('detections', tags),
        ],
    )

    calibrator = Node(
        package='camera_calibration_apriltag',
        executable='carcalibrator',
        name='carcalibrator',
        output='screen',
        arguments=[
            '--size', LaunchConfig('size'),
            '--tag-size', LaunchConfig('tag_size'),
            '--tag-spacing', LaunchConfig('tag_spacing'),
            '--start-id', LaunchConfig('start_id'),
            '--tag-family', LaunchConfig('tag_family'),
            '--min-tags', LaunchConfig('min_tags'),
            '--odom-frame', LaunchConfig('odom_frame'),
            '--base-frame', LaunchConfig('base_frame'),
            '--camera-frame', LaunchConfig('camera_frame'),
            '--tf-timeout', LaunchConfig('tf_timeout'),
        ],
        remappings=[
            ('image', image_rect),
            ('tags', tags),
            ('camera_info', camera_info),
            ('cmd_vel', cmd_vel),
        ],
    )

    return [detector, calibrator]


def generate_launch_description():
    return launch.LaunchDescription([
        # --- Topic wiring ---
        LaunchArg('image_rect', default_value='/camera/image_rect',
                  description='rectified image topic fed to the detector'),
        LaunchArg('camera_info', default_value='/camera/camera_info',
                  description='camera_info topic (intrinsics for solvePnP)'),
        LaunchArg('tags', default_value='/camera/tags',
                  description='tag detections topic'),
        LaunchArg('cmd_vel', default_value='/cmd_vel',
                  description='velocity command topic for the joystick'),
        LaunchArg('run_detector', default_value='true',
                  description='also launch the apriltag_ros detector node'),
        LaunchArg('detector_family', default_value='36h11',
                  description='tag family for the apriltag_ros detector'),
        # --- AprilTag board geometry ---
        LaunchArg('size', default_value='7x5', description='board size as COLSxROWS in tags'),
        LaunchArg('tag_size', default_value='0.030', description='tag edge length (meters)'),
        LaunchArg('tag_spacing', default_value='0.035',
                  description='tag centre-to-centre distance (meters)'),
        LaunchArg('start_id', default_value='0', description='id of the top-left tag'),
        LaunchArg('tag_family', default_value='',
                  description='family filter for the calibrator (empty = accept any)'),
        LaunchArg('min_tags', default_value='1',
                  description='minimum tags required for a grid pose'),
        # --- Frames ---
        LaunchArg('odom_frame', default_value='odom', description='static world frame'),
        LaunchArg('base_frame', default_value='base_link', description='moving car body frame'),
        LaunchArg('camera_frame', default_value='',
                  description='camera optical frame (empty = from camera_info header)'),
        LaunchArg('tf_timeout', default_value='0.2',
                  description='seconds to wait for the odom->base tf at a stamp'),
        OpaqueFunction(function=launch_setup),
    ])
