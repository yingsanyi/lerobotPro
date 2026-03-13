# Songling ALOHA Hardware Mapping Reference

`examples/songling_aloha/teleop.yaml` is the only authoritative config file.

This note mirrors the values in that config so they are easier to scan during bring-up.

## CAN Mapping

| Role | Interface | Notes |
| --- | --- | --- |
| follower_left | `can1` | robot.left_arm_config.port |
| follower_right | `can0` | robot.right_arm_config.port |
| leader_left | `can1` | teleop.left_arm_config.port (integrated chain) |
| leader_right | `can0` | teleop.right_arm_config.port (integrated chain) |

## Camera Mapping (UVC)

| Observation key | Device index/path | Notes |
| --- | --- | --- |
| `left_high` | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1S74101GJ-video-index0` | top camera, mapped to left_arm_config.cameras.high.index_or_path |
| `left_elbow` | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_AU1GC32017T-video-index0` | left elbow camera, mapped to left_arm_config.cameras.elbow.index_or_path |
| `right_elbow` | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1S74101GY-video-index0` | right elbow camera, mapped to right_arm_config.cameras.elbow.index_or_path |

Notes:
- Prefer `/dev/v4l/by-id/...` over numeric indices to avoid drift across reboots/unplug/replug.
- If a camera stalls (`select() timeout`) or looks corrupted, try switching `video-index0` <-> `video-index1` for that device.

## Bus Parameters

| Parameter | Value |
| --- | --- |
| use_can_fd | `false` |
| can_bitrate | `1000000` |
| can_data_bitrate | `5000000` |
