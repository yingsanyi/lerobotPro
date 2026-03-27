---
name: songling-aloha-integrated-can
description: Adapt the main LeRobot codebase for Songling ALOHA integrated master/slave CAN chains. Use when working on Songling bring-up, CAN mapping, raw CAN sniffing, camera + CAN visualization, or protocol adapter development where each side is an integrated chain sharing one CAN interface. Always treat `mobile-aloha/` and `lerobot_piper/` as read-only references and apply code changes only in the main `lerobot` project.
---

# Songling Aloha Integrated Can

## Overview

Use this skill to keep Songling integration work aligned with real hardware behavior:
- one integrated chain per side
- one CAN interface per side
- hardware teleoperation available after power-on
- no separate software leader device to connect

## Non-negotiables

1. Treat `mobile-aloha/` and `lerobot_piper/` as read-only references.
2. Apply implementation changes only in the main `lerobot` project.
3. Prefer traffic-based interface detection and live raw CAN visualization for bring-up.
4. Avoid modeling Songling integrated chains as independent software `leader` + `follower` on the same bus unless there is an explicit adapter that supports it.

## Workflow

1. Confirm CAN mode and interface state.
2. Detect left/right CAN interfaces (traffic-drop method preferred for integrated chains).
3. Update the single Songling mapping config (`examples/songling_aloha/teleop.yaml`) with detected ports.
4. Run raw CAN + camera visualization (`--raw-can-mode`) to validate bus activity and camera stability.
5. Keep the example directory focused on the main workflow; add diagnostics only when they materially unblock adapter work.
6. Implement adapter changes in main `lerobot` only.

## Canonical Commands

```bash
conda activate lerobot_v050
lerobot-setup-can --mode=setup --interfaces=can0,can1 --use_fd=false --bitrate=1000000
```

```bash
lerobot-find-port \
  --kind can \
  --can-mode traffic \
  --interfaces can0,can1 \
  --traffic-direction rx \
  --traffic-window-s 3 \
  --min-msgs 5 \
  --settle-time-s 1.0 \
  --write-songling-config examples/songling_aloha/teleop.yaml \
  --songling-side left
```

Run again with `--songling-side right` for the right chain.

```bash
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode
```

## Validation Checklist

1. `teleop.left_arm_config.port == robot.left_arm_config.port`
2. `teleop.right_arm_config.port == robot.right_arm_config.port`
3. Left and right ports are distinct unless there is confirmed single-bus hardware wiring.
4. Live traffic visualization shows stable CAN activity and expected CAN 2.0 flags (`is_fd=0` if classic CAN).
5. Any proposed code edits do not touch `mobile-aloha/` or `lerobot_piper/`.

## Failure Handling

1. If `No difference was found` in interface mode, switch to `--can-mode traffic`.
2. If traffic baseline is too low, generate motion on the target chain and retry.
3. If CAN handshake fails with default OpenArm IDs, treat it as protocol mismatch and continue adapter mapping from sniffed IDs instead of forcing default IDs.
