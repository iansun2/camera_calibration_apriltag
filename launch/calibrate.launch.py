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


def _bool(context, name):
    return LaunchConfig(name).perform(context).lower() in ('1', 'true', 'yes', 'on')


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
            'num_threads': LaunchConfig('num_threads'),
        }.items(),
    )

    # Value arguments (always passed)
    cal_args = [
        '--size', LaunchConfig('size'),
        '--tag-size', LaunchConfig('tag_size'),
        '--tag-spacing', LaunchConfig('tag_spacing'),
        '--start-id', LaunchConfig('start_id'),
        '--tag-family', LaunchConfig('tag_family'),
        '--min-tags', LaunchConfig('min_tags'),
        '--max-views', LaunchConfig('max_views'),
        '--camera_name', LaunchConfig('camera_name'),
        '--approximate', LaunchConfig('approximate'),
        '--queue-size', LaunchConfig('queue_size'),
        '-k', LaunchConfig('k_coefficients'),
        '--fisheye-k-coefficients', LaunchConfig('fisheye_k_coefficients'),
        '--max-chessboard-speed', LaunchConfig('max_chessboard_speed'),
    ]
    # Boolean flags (only passed when enabled)
    if _bool(context, 'allow_partial_board'):
        cal_args.append('--allow-partial-board')
    if not _bool(context, 'service_check'):
        cal_args.append('--no-service-check')
    if _bool(context, 'fix_principal_point'):
        cal_args.append('--fix-principal-point')
    if _bool(context, 'fix_aspect_ratio'):
        cal_args.append('--fix-aspect-ratio')
    if _bool(context, 'zero_tangent_dist'):
        cal_args.append('--zero-tangent-dist')
    if _bool(context, 'fisheye_recompute_extrinsics'):
        cal_args.append('--fisheye-recompute-extrinsicsts')
    if _bool(context, 'fisheye_fix_skew'):
        cal_args.append('--fisheye-fix-skew')
    if _bool(context, 'fisheye_fix_principal_point'):
        cal_args.append('--fisheye-fix-principal-point')
    if _bool(context, 'fisheye_check_conditions'):
        cal_args.append('--fisheye-check-conditions')

    calibrator = Node(
        package='camera_calibration_apriltag',
        executable='cameracalibrator',
        name='cameracalibrator',
        output='screen',
        arguments=cal_args,
        remappings=[
            ('image', image_topic),
            ('tags', tags_topic),
            ('camera/set_camera_info', set_info_topic),
        ],
    )

    return [detector, calibrator]


def generate_launch_description():
    return launch.LaunchDescription([
        # --- Detector / topic wiring ---
        LaunchArg('camera', default_value='camera', description='camera namespace'),
        LaunchArg('image', default_value='image_raw', description='image topic name'),
        LaunchArg('tags', default_value='tags', description='tag detections topic name'),
        LaunchArg('type', default_value='umich', description='detector type (umich, mit)'),
        LaunchArg('image_transport', default_value='raw', description='input image transport'),
        LaunchArg('num_threads', default_value='4', description='detector worker threads'),
        # --- AprilTag board geometry ---
        LaunchArg('size', default_value='7x5', description='board size as COLSxROWS in tags'),
        LaunchArg('tag_size', default_value='0.030', description='tag edge length (meters)'),
        LaunchArg('tag_spacing', default_value='0.035',
                  description='tag centre-to-centre distance (meters)'),
        LaunchArg('start_id', default_value='0', description='id of the top-left tag'),
        LaunchArg('tag_family', default_value='tf36h11',
                  description='tag family (used by both detector and calibrator)'),
        # --- Sampling ---
        LaunchArg('min_tags', default_value='1',
                  description='minimum tags required to use a view'),
        LaunchArg('allow_partial_board', default_value='false',
                  description='accept samples without every tag (default requires full board)'),
        LaunchArg('max_views', default_value='0',
                  description='cap views used by the solver (0 = all); fewer = faster'),
        LaunchArg('camera_name', default_value='narrow_stereo',
                  description='camera name written into the calibration file'),
        # --- ROS communication ---
        LaunchArg('approximate', default_value='0.0',
                  description='image/tags sync slop in seconds (0 = exact)'),
        LaunchArg('queue_size', default_value='1', description='input queue size'),
        LaunchArg('service_check', default_value='true',
                  description='wait for set_camera_info service at startup'),
        # --- Pinhole optimizer ---
        LaunchArg('k_coefficients', default_value='2',
                  description='pinhole radial distortion coefficients (up to 6)'),
        LaunchArg('fix_principal_point', default_value='false',
                  description='pinhole: fix principal point at image center'),
        LaunchArg('fix_aspect_ratio', default_value='false',
                  description='pinhole: enforce fx == fy'),
        LaunchArg('zero_tangent_dist', default_value='false',
                  description='pinhole: set tangential distortion (p1, p2) to zero'),
        # --- Fisheye optimizer ---
        LaunchArg('fisheye_k_coefficients', default_value='4',
                  description='fisheye radial distortion coefficients (up to 4)'),
        LaunchArg('fisheye_recompute_extrinsics', default_value='false',
                  description='fisheye: recompute extrinsics each intrinsic iteration'),
        LaunchArg('fisheye_fix_skew', default_value='false',
                  description='fisheye: fix skew (alpha) to zero'),
        LaunchArg('fisheye_fix_principal_point', default_value='false',
                  description='fisheye: fix principal point at image center'),
        LaunchArg('fisheye_check_conditions', default_value='false',
                  description='fisheye: check validity of condition number'),
        # --- Misc ---
        LaunchArg('max_chessboard_speed', default_value='-1.0',
                  description='reject views where the board moves faster than this (px/frame)'),
        OpaqueFunction(function=launch_setup),
    ])
