# Songling ALOHA Main Workflow

This directory has been trimmed to keep only the main Songling host-side workflow:

- upper-computer visualization and manual control
- teleoperation / teaching
- a small set of support modules used by those entrypoints

One-off bring-up, recovery, diagnosis, and protocol-reverse-engineering scripts were removed on purpose.

## Kept Files

- `teleop.yaml`: single source of truth for CAN, cameras, and live-monitor defaults
- `hardware_mapping.md`: readable mirror of the unified config
- `capture_three_cameras.py`: verify the three UVC cameras quickly
- `visualize_teleop_live.py`: live host-side visualization
- `manual_joint_control.py`: desktop upper-computer joint-control panel

## Typical Workflow

### 1. Prepare Config

Use `examples/songling_aloha/teleop.yaml` as the only maintained config file.

Write CAN and camera mappings back into that file with:

```bash
lerobot-find-port --write-songling-config examples/songling_aloha/teleop.yaml --songling-side left
lerobot-find-port --write-songling-config examples/songling_aloha/teleop.yaml --songling-side right
lerobot-find-cameras opencv --mode configure-songling --write-songling-config examples/songling_aloha/teleop.yaml ...
```

### 2. Verify Cameras

```bash
python examples/songling_aloha/capture_three_cameras.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --output-dir /tmp/songling_snapshots
```

### 3. Visualize Live System

```bash
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml
```

### 3.5 Optimize USB Camera Stability (Recommended)

When cameras are connected through cascaded USB hubs, Linux autosuspend can cause
intermittent disconnect/re-enumeration under load.

Inspect current camera/hub USB power state:

```bash
python examples/songling_aloha/optimize_usb_camera_stability.py \
  --config-path examples/songling_aloha/teleop.yaml
```

Apply runtime optimization (effective until reboot):

```bash
sudo python examples/songling_aloha/optimize_usb_camera_stability.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --apply-runtime
```

Generate persistent udev rules:

```bash
python examples/songling_aloha/optimize_usb_camera_stability.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --write-udev-rules /tmp/99-songling-usb-camera-stability.rules
sudo cp /tmp/99-songling-usb-camera-stability.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 4. Manual Upper-Computer Control

```bash
python examples/songling_aloha/manual_joint_control.py \
  --config-path examples/songling_aloha/teleop.yaml
```

## Notes

- The integrated Songling hardware handles the master/follower relationship on the hardware side after power-on.
- For Songling ALOHA, `can0` and `can1` mean one shared CAN bus per side, not one CAN bus per individual arm.
- `visualize_teleop_live.py` now follows `double_piper.MD`: it reads master control echo and slave feedback from the same CAN port on each side.
- Teaching is now a pure live workflow in this example directory; dataset recording / replay entrypoints were removed on purpose.
- This repo now keeps the Songling example directory focused on host workflows instead of protocol forensics and recovery utilities.
- If you need fresh diagnostics in the future, add them back as separate purpose-built tools instead of mixing them into the main workflow directory.
