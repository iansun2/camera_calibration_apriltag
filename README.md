### ROS2 Version: Humble

### FROM:
- [apriltag_detector](https://github.com/ros-misc-utilities/apriltag_detector)
- [image_pipeline](https://github.com/ros-perception/image_pipeline)


### RUN:

```
ros2 launch camera_calibration_apriltag calibrate.launch.py \
    camera:=camera image:=image_raw \
    size:=8x6 tag_size:=0.030 tag_spacing:=0.03375
```