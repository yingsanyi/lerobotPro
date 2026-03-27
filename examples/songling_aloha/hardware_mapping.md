# Songling ALOHA Hardware Mapping Reference

`examples/songling_aloha/teleop.yaml` is the authoritative Songling unified config.

This note mirrors the follower-direct runtime values so they are easier to scan during bring-up.
This profile now targets live teaching/monitoring only, not dataset recording.

## CAN Mapping

| Role | Interface | Notes |
| --- | --- | --- |
| follower_left_pair | `can0` | shared CAN for the whole left master/slave pair, mapped from `robot.left_arm_config.channel` |
| follower_right_pair | `can1` | shared CAN for the whole right master/slave pair, mapped from `robot.right_arm_config.channel` |

## Camera Mapping (UVC)

| Observation key | Device index/path | Notes |
| --- | --- | --- |
| `left_high` | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1S74101GJ-video-index0` | top camera, mapped to left_arm_config.cameras.high.index_or_path |
| `left_elbow` | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_AU1GC32017T-video-index0` | left elbow camera, mapped to left_arm_config.cameras.elbow.index_or_path |
| `right_elbow` | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Dabai_DC1_CC1S74101GY-video-index0` | right elbow camera, mapped to right_arm_config.cameras.elbow.index_or_path |

Notes:
- Songling ALOHA uses two CAN ports total: one for the left master/slave pair and one for the right master/slave pair.
- Prefer `/dev/v4l/by-id/...` over numeric indices to avoid drift across reboots/unplug/replug.
- If a camera stalls (`select() timeout`) or looks corrupted, try switching `video-index0` <-> `video-index1` for that device.

## Bus Parameters

| Parameter | Value |
| --- | --- |
| transport_backend | `piper_sdk` |
| use_can_fd | `false` |
| interface | `socketcan` |
| bitrate | `1000000` |
| can_data_bitrate | `5000000` |
| speed_percent | `20` |
| motion_mode | `js` |
| gripper_force | `1.0` |

Notes:
- Installation-position downlinks are intentionally disabled in this fork to avoid changing arm-side compensation on hardware.
- Leader/follower role reconfiguration and teach-parameter writes are also blocked for the same reason.
- Host-side demonstration recording should read both control echo and follower feedback from the same side CAN port.
