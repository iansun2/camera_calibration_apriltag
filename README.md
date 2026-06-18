# camera_calibration_apriltag

Monocular and stereo camera calibration from **AprilTag** detections, with a
**PySide6** GUI.

**ROS 2 version:** Humble

Derived from:
- [apriltag_detector](https://github.com/ros-misc-utilities/apriltag_detector)
- [image_pipeline](https://github.com/ros-perception/image_pipeline)

Unlike the original `camera_calibration` package, this package does **not**
detect the target itself. Detection is delegated to an external AprilTag
detector node which publishes `apriltag_msgs/msg/AprilTagDetectionArray`. This
package consumes those detections, matches each tag to its known 3D position on
an **AprilGrid** target, and runs the OpenCV calibration solver.

```
camera ──image──► apriltag_detector ──tags──►  camera_calibration_apriltag
   └──────────────────image───────────────────────────────┘   (GUI + solver)
```

## Features

- Monocular and stereo intrinsic/extrinsic calibration.
- AprilGrid target (regular grid of tags), fully configurable.
- PySide6 GUI: live image with tag overlay, **X-Y coverage heatmap**, **size**
  and **skew** coverage bars, live FPS, reprojection/epipolar error and final
  RMS readout.
- **Load Images** button to calibrate offline from saved images (tags are
  detected in-process with OpenCV, no detector node needed).
- Automatic corner-order detection (robust to whichever corner convention the
  detector uses).
- Fast on slow/ARM CPUs: cheap homography-based corner-order search, `CALIB_USE_LU`,
  and an optional `--max-views` cap.

## Build

```bash
cd <workspace>
colcon build --packages-up-to camera_calibration_apriltag
source install/setup.bash
```

Dependencies: `rclpy`, `cv_bridge`, `image_geometry`, `message_filters`,
`sensor_msgs`, `std_srvs`, `apriltag_msgs`, `apriltag_detector` (runtime, for the
launch files), `python3-pyside6`, OpenCV (with `aruco`/`apriltag` support for
offline image loading).

## Quick start

Monocular, against a running camera (`/camera/image_raw`) and detector:

```bash
ros2 launch camera_calibration_apriltag calibrate.launch.py \
    size:=7x5 tag_size:=0.035 tag_spacing:=0.040 \
    camera:=camera image:=image_raw
```

Stereo:

```bash
ros2 launch camera_calibration_apriltag calibrate_stereo.launch.py \
    size:=7x5 tag_size:=0.035 tag_spacing:=0.040 \
    left_camera:=left_camera right_camera:=right_camera
```

Run the node directly (you wire the topics yourself):

```bash
ros2 run camera_calibration_apriltag cameracalibrator \
    --size 7x5 --tag-size 0.035 --tag-spacing 0.040 \
    --ros-args -r image:=/camera/image_raw -r tags:=/camera/tags
```

## Topics and services

The node uses the names below; the launch files remap them under the camera
namespace (e.g. `image` → `/<camera>/image_raw`, `tags` → `/<camera>/tags`).

### Monocular

| Direction | Name | Type | Description |
|-----------|------|------|-------------|
| Subscribe | `image` | `sensor_msgs/msg/Image` | camera image, for display/overlay |
| Subscribe | `tags` | `apriltag_msgs/msg/AprilTagDetectionArray` | tag detections used for calibration |
| Service client | `camera/set_camera_info` | `sensor_msgs/srv/SetCameraInfo` | called on **COMMIT** to upload the result |

### Stereo

| Direction | Name | Type | Description |
|-----------|------|------|-------------|
| Subscribe | `left` | `sensor_msgs/msg/Image` | left image |
| Subscribe | `left_tags` | `apriltag_msgs/msg/AprilTagDetectionArray` | left tag detections |
| Subscribe | `right` | `sensor_msgs/msg/Image` | right image |
| Subscribe | `right_tags` | `apriltag_msgs/msg/AprilTagDetectionArray` | right tag detections |
| Service client | `left_camera/set_camera_info` | `sensor_msgs/srv/SetCameraInfo` | upload left result on COMMIT |
| Service client | `right_camera/set_camera_info` | `sensor_msgs/srv/SetCameraInfo` | upload right result on COMMIT |

`image`/`tags` (and left/right) are time-synchronized with a
`TimeSynchronizer` (exact) or `ApproximateTimeSynchronizer` (when
`approximate > 0`).

## AprilGrid target

A regular `COLS x ROWS` grid of square AprilTags. Tag ids increase **row-major**
starting at `start_id` in the **top-left** corner, e.g. for `7x5`:

```
col   0   1   2   3   4   5   6
row0  0   1   2   3   4   5   6
row1  7   8   9  10  11  12  13
row2 14  15  16  17  18  19  20
row3 21  22  23  24  25  26  27
row4 28  29  30  31  32  33  34
```

- `tag_size` — edge length of one tag, in meters.
- `tag_spacing` — **centre-to-centre** distance between neighbouring tags, in
  meters (not the gap between tags). For 35 mm tags with a 5 mm gap,
  `tag_spacing = 0.040`.

## GUI

- **Live image** with detected tags overlaid (rectified view once calibrated).
- **FPS** — processing rate of synchronized frames.
- **Samples / Tags in view** counters.
- **Coverage**
  - *Image position (X-Y)* — 2D heatmap of where the board has been seen; dark
    cells are uncovered image regions.
  - *Board size* — coverage across near/far (board large/small in the image).
  - *Skew / tilt* — coverage across flat → steeply tilted views.
- **Final calibration RMS** — solver RMS reprojection error after calibrating.
- **Camera model** — Pinhole or Fisheye (select before calibrating).
- **Scale (alpha)** — rectification zoom, 0 = cropped, 1 = full frame.
- **LOAD IMAGES…** — add saved images from a folder as samples (mono: any
  `*.png/*.jpg`; stereo: `left-*`/`right-*` pairs).
- **CALIBRATE / SAVE / COMMIT** — solve, write
  `/tmp/calibrationdata.tar.gz`, and upload via `set_camera_info`.

## Node parameters (`cameracalibrator` CLI)

Pass these as plain CLI args (ROS remaps go after `--ros-args`).

### General

| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--camera_name` | `narrow_stereo` | camera name written into the calibration file |
| `--stereo` | off | calibrate a stereo pair (`left`/`left_tags`, `right`/`right_tags`) |

### AprilTag board

| Option | Default | Description |
|--------|---------|-------------|
| `-s`, `--size` | `8x6` | board size as `COLSxROWS` in tags |
| `--tag-size` | `0.030` | tag edge length (meters) |
| `--tag-spacing` | `0.03375` | tag centre-to-centre distance (meters) |
| `--start-id` | `0` | id of the top-left tag |
| `--tag-family` | `""` (any) | accept only this tag family; empty accepts any |
| `--min-tags` | `1` | minimum tags required to use a view |
| `--allow-partial-board` | off | accept samples without every tag (default requires the full board) |
| `--max-views` | `0` (all) | cap the number of views fed to the solver; fewer = faster on slow/ARM CPUs |

### ROS communication

| Option | Default | Description |
|--------|---------|-------------|
| `--approximate` | `0.0` | sync slop (seconds) between image and tag topics; `0` = exact |
| `--no-service-check` | (check on) | do not wait for the `set_camera_info` service at startup |
| `--queue-size` | `1` | input queue size (`0` = unlimited) |

### Pinhole optimizer

| Option | Default | Description |
|--------|---------|-------------|
| `--fix-principal-point` | off | fix the principal point at the image center |
| `--fix-aspect-ratio` | off | enforce `fx == fy` |
| `--zero-tangent-dist` | off | set tangential distortion (`p1`, `p2`) to zero |
| `-k`, `--k-coefficients` | `2` | number of radial distortion coefficients (up to 6) |

### Fisheye optimizer

| Option | Default | Description |
|--------|---------|-------------|
| `--fisheye-k-coefficients` | `4` | radial distortion coefficients (up to 4) |
| `--fisheye-recompute-extrinsicsts` | off | recompute extrinsics each intrinsic iteration |
| `--fisheye-fix-skew` | off | fix skew (alpha) to zero |
| `--fisheye-fix-principal-point` | off | fix the principal point at the image center |
| `--fisheye-check-conditions` | off | check validity of the condition number |

### Misc

| Option | Default | Description |
|--------|---------|-------------|
| `--max-chessboard-speed` | `-1.0` | reject views where the board moves faster than this (px/frame); useful for rolling shutter |

## Launch parameters

### `calibrate.launch.py` (monocular)

| Argument | Default | Description |
|----------|---------|-------------|
| `camera` | `camera` | camera namespace |
| `image` | `image_raw` | image topic name |
| `tags` | `tags` | tag detections topic name |
| `type` | `umich` | detector type (`umich`, `mit`) |
| `image_transport` | `raw` | input image transport |
| `num_threads` | `4` | detector worker threads |
| `size` | `8x6` | board size as `COLSxROWS` in tags |
| `tag_size` | `0.030` | tag edge length (meters) |
| `tag_spacing` | `0.03375` | tag centre-to-centre distance (meters) |
| `start_id` | `0` | id of the top-left tag |
| `tag_family` | `tf36h11` | tag family (used by both detector and calibrator) |
| `min_tags` | `1` | minimum tags required to use a view |
| `allow_partial_board` | `false` | accept samples without every tag |
| `max_views` | `0` | cap views used by the solver (`0` = all); fewer = faster |
| `camera_name` | `narrow_stereo` | camera name written into the calibration file |
| `approximate` | `0.0` | image/tags sync slop (seconds); `0` = exact |
| `queue_size` | `1` | input queue size |
| `service_check` | `true` | wait for `set_camera_info` service at startup |
| `k_coefficients` | `2` | pinhole radial distortion coefficients (up to 6) |
| `fix_principal_point` | `false` | pinhole: fix principal point at image center |
| `fix_aspect_ratio` | `false` | pinhole: enforce `fx == fy` |
| `zero_tangent_dist` | `false` | pinhole: set tangential distortion to zero |
| `fisheye_k_coefficients` | `4` | fisheye radial distortion coefficients (up to 4) |
| `fisheye_recompute_extrinsics` | `false` | fisheye: recompute extrinsics each intrinsic iteration |
| `fisheye_fix_skew` | `false` | fisheye: fix skew (alpha) to zero |
| `fisheye_fix_principal_point` | `false` | fisheye: fix principal point at image center |
| `fisheye_check_conditions` | `false` | fisheye: check validity of condition number |
| `max_chessboard_speed` | `-1.0` | reject views moving faster than this (px/frame) |

### `calibrate_stereo.launch.py` (stereo)

Same as above **except** the camera/topic wiring is per-eye and the sync
defaults are looser (left/right stamps are rarely bit-identical):

| Argument | Default | Description |
|----------|---------|-------------|
| `left_camera` | `left_camera` | left camera namespace |
| `right_camera` | `right_camera` | right camera namespace |
| `left_image` | `image_raw` | left image topic name (within its namespace) |
| `right_image` | `image_raw` | right image topic name (within its namespace) |
| `tags` | `tags` | tag detections topic name (within each namespace) |
| `approximate` | `0.05` | image/tags sync slop (seconds); set `0.0` for hardware-synced rigs |
| `queue_size` | `5` | input queue size |

All other arguments (`type`, `image_transport`, `num_threads`, board geometry,
sampling, optimizer, misc) are identical to the monocular launch.

## Calibration workflow

1. Start the camera driver and launch the calibrator (above).
2. Move the board so the **X-Y heatmap** fills up (cover the corners/edges), the
   **size** bar covers near *and* far, and the **skew** bar covers flat *and*
   tilted views.
3. When **CALIBRATE** enables, click it. Check the **Final calibration RMS**
   (sub-pixel, e.g. < 1 px, is good).
4. **SAVE** writes `/tmp/calibrationdata.tar.gz` (images + `ost.yaml`/`ost.txt`).
5. **COMMIT** uploads the result to the camera driver via `set_camera_info`.

To calibrate offline, click **LOAD IMAGES…** and select a folder of saved
images instead of (or in addition to) live capture.

## Performance notes

- The OpenCV solver (`calibrateCamera`/`stereoCalibrate`) is single-threaded;
  there is no multicore drop-in replacement.
- Corner-order selection uses a cheap per-view homography fit (not a full
  calibration per candidate), and the pinhole solve uses `CALIB_USE_LU`.
- On slow/ARM machines set `--max-views 30` (or `max_views:=30`) — accuracy
  saturates well before that and the solve is much faster.

## Output files

`SAVE` writes `/tmp/calibrationdata.tar.gz` containing:

- `left-XXXX.png` (and `right-XXXX.png` for stereo) — the sample images.
- `ost.yaml` / `ost.txt` — the calibration in OpenCV/ROS formats.

## Troubleshooting

- **Huge distortion / wrong `fx`,`fy`:** usually wrong `tag_spacing` (must be
  *centre-to-centre*, not the gap) or a corner-order mismatch. The corner order
  is auto-selected and printed at calibrate time; check the **RMS** — a large
  RMS means a geometry/parameter mismatch, a small RMS with a bad rectified
  image means insufficient view variety.
- **No samples added:** by default a sample requires the *full* board in view;
  pass `--allow-partial-board` (or `allow_partial_board:=true`) to relax this.
- **GUI shows no image:** the `image` and `tags` topics must be time-synced;
  increase `approximate` if their stamps differ.
