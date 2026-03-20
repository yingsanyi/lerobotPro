# Songling ALOHA Profile (CAN + UVC)

This profile is for a Songling integrated master/slave chain over CAN plus 3 UVC cameras.

Important:
- one integrated chain per side
- one CAN interface per side
- hardware teleoperation is handled by the Songling chain itself after power-on
- there is no separate software leader device to connect in LeRobot today
- `mobile-aloha/` and `lerobot_piper/` are reference material only and are not modified here

It is designed for:

- one integrated left chain over CAN
- one integrated right chain over CAN
- three UVC cameras (`left_high`, `left_elbow`, `right_elbow`)
- no mobile base in v1

## 0) Custom MotorsBus adaptation route

Since this setup uses an integrated leader/follower chain and does not match default OpenArm
motor probing IDs, start from protocol sniffing first:

```bash
python examples/songling_aloha/sniff_can_frames.py \
  --interfaces can0 can1 \
  --duration-s 20 \
  --bitrate 1000000 \
  --output-dir /tmp/songling_can_sniff
```

Then analyze `/tmp/songling_can_sniff/can0.csv` and `can1.csv` to map:

- joint-related arbitration IDs
- payload fields for position/velocity/effort
- command frame format for motion control

Once those are confirmed, implement/extend a dedicated custom MotorsBus adapter.

The current repo now includes a heuristic Songling RX decoder for the observed
state frames, but it is still an evolving best-effort mapping, not a confirmed
vendor protocol specification.

For a more structured workflow, capture one idle segment and one motion segment,
then diff them:

```bash
python examples/songling_aloha/sniff_can_frames.py \
  --interfaces can0 can1 \
  --duration-s 20 \
  --bitrate 1000000 \
  --output-dir /tmp/songling_can_idle

python examples/songling_aloha/sniff_can_frames.py \
  --interfaces can0 can1 \
  --duration-s 20 \
  --bitrate 1000000 \
  --output-dir /tmp/songling_can_move

python examples/songling_aloha/analyze_can_sniff.py \
  --idle-dir /tmp/songling_can_idle \
  --active-dir /tmp/songling_can_move
```

Recommended capture procedure:

- For `idle`, keep both sides powered on but do not move the arms.
- For `move`, manually move only one side or one joint group at a time if possible.
- Save separate motion captures for left-only, right-only, gripper-only, etc. if you want cleaner protocol separation.

## 1) Environment and dependencies

Use the existing conda environment:

```bash
conda activate lerobot_v050
pip install -e ".[openarms]"
python -c "import lerobot; print(lerobot.__version__)"
pip check
lerobot-setup-can --help
```

Expected version: `0.5.1`.

## 2) Hardware probing and fixed mapping

### Bring up CAN interfaces

```bash
lerobot-setup-can --mode=setup --interfaces=can0,can1 --use_fd=false --bitrate=1000000
lerobot-setup-can --mode=test --interfaces=can0,can1 --use_fd=false --bitrate=1000000
```

If your host exposes fewer or different interfaces, replace the list accordingly.

To detect which integrated chain corresponds to each CAN interface and write it back into the single Songling config:

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

Run the same command again with `--songling-side right` for the other chain.

### Find UVC cameras

```bash
lerobot-find-cameras opencv
```

To map physical cameras to device ports, save one snapshot per detected OpenCV camera:

```bash
lerobot-find-cameras opencv \
  --mode snapshot \
  --output-dir /tmp/lerobot_camera_probe \
  --opencv-width 640 \
  --opencv-height 480 \
  --opencv-fourcc MJPG \
  --opencv-backend 200 \
  --snapshot-name-style short
```

Then open the saved `*.png` files and `/tmp/lerobot_camera_probe/manifest.json` and decide which port is:

- top camera (`left_high`)
- left elbow (`left_elbow`)
- right elbow (`right_elbow`)

Prefer the `suggested_index_or_path` values from `manifest.json` (usually `/dev/v4l/by-id/...`) over numeric indices.

`examples/songling_aloha/teleop.yaml` is the only maintained Songling config file. After selecting the three camera ports, write them back automatically:

```bash
lerobot-find-cameras opencv \
  --mode configure-songling \
  --write-songling-config examples/songling_aloha/teleop.yaml \
  --songling-left-high /dev/v4l/by-id/<TOP_CAMERA>-video-index0 \
  --songling-left-elbow /dev/v4l/by-id/<LEFT_ELBOW_CAMERA>-video-index0 \
  --songling-right-elbow /dev/v4l/by-id/<RIGHT_ELBOW_CAMERA>-video-index0 \
  --opencv-width 640 \
  --opencv-height 480 \
  --opencv-fps 30 \
  --opencv-fourcc MJPG \
  --opencv-backend 200
```

You can also write directly from `manifest.json` by camera index (no long path copy-paste):

```bash
lerobot-find-cameras opencv \
  --mode configure-songling \
  --write-songling-config examples/songling_aloha/teleop.yaml \
  --songling-manifest /tmp/lerobot_camera_probe/manifest.json \
  --songling-left-high-index 2 \
  --songling-left-elbow-index 0 \
  --songling-right-elbow-index 1 \
  --opencv-width 640 \
  --opencv-height 480 \
  --opencv-fps 30 \
  --opencv-fourcc MJPG \
  --opencv-backend 200
```

This updates:

- `robot.left_arm_config.cameras.high.index_or_path`
- `robot.left_arm_config.cameras.elbow.index_or_path`
- `robot.right_arm_config.cameras.elbow.index_or_path`
- `width`
- `height`
- `fps`
- `fourcc`
- `backend`

in `examples/songling_aloha/teleop.yaml`.

Lock one mapping and keep it stable:

| Device | Example |
| --- | --- |
| follower left | `can1` |
| follower right | `can0` |
| leader left | `can1` (integrated chain) |
| leader right | `can0` (integrated chain) |
| `left_high` | `/dev/video4` (top camera) |
| `left_elbow` | `/dev/video0` |
| `right_elbow` | `/dev/video2` |

Keep `examples/songling_aloha/teleop.yaml` as the single source of truth. `hardware_mapping.md` is only a readable mirror of that file.

### Capture one still image from all three cameras

Default to YAML mapping:

```bash
python examples/songling_aloha/capture_three_cameras.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --output-dir /tmp/songling_snapshots
```

Explicit camera overrides:

```bash
python examples/songling_aloha/capture_three_cameras.py \
  --left-high /dev/v4l/by-id/<TOP_CAMERA>-video-index0 \
  --left-elbow /dev/v4l/by-id/<LEFT_ELBOW_CAMERA>-video-index0 \
  --right-elbow /dev/v4l/by-id/<RIGHT_ELBOW_CAMERA>-video-index0 \
  --output-dir /tmp/songling_snapshots
```

LeRobot-style dotted overrides are supported too (`CLI > YAML`):

```bash
python examples/songling_aloha/capture_three_cameras.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --robot.left_arm_config.cameras.high.index_or_path=/dev/video4 \
  --robot.left_arm_config.cameras.elbow.index_or_path=/dev/video0 \
  --robot.right_arm_config.cameras.elbow.index_or_path=/dev/video2
```

Saved filenames include both logical camera name and device node, for example:

- `left_high__dev_video4__20260311_173000.jpg`
- `left_elbow__dev_video0__20260311_173000.jpg`
- `right_elbow__dev_video2__20260311_173000.jpg`

## 3) Run sequence

### A. Current supported path

At the current stage, the supported bring-up flow in this example is:

- sniff raw CAN
- visualize raw CAN + cameras in Rerun
- map protocol and implement a dedicated Songling MotorsBus / robot adapter in the main LeRobot codebase

The generic `bi_openarm_leader` + `bi_openarm_follower` software teleop path is not the correct runtime model
for this hardware, because both master and slave are already integrated on the same CAN chain.

### A1. Realtime visualization (3 cameras + 2 leader/follower arm states)

```bash
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode
```

This is the recommended mode right now. It only uses:

- 3 camera streams
- raw CAN traffic statistics
- Rerun visualization

`visualize_teleop_live.py` also accepts native `lerobot-teleoperate` style overrides
(for Songling bimanual fields), with CLI values taking precedence over YAML:

```bash
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode \
  --robot.type=bi_openarm_follower \
  --robot.id=songling_aloha_follower \
  --robot.left_arm_config.port=can1 \
  --robot.right_arm_config.port=can0 \
  --robot.left_arm_config.cameras.high.index_or_path=/dev/video4 \
  --robot.left_arm_config.cameras.elbow.index_or_path=/dev/video0 \
  --robot.right_arm_config.cameras.elbow.index_or_path=/dev/video2 \
  --teleop.type=bi_openarm_leader \
  --teleop.id=songling_aloha_leader \
  --teleop.left_arm_config.port=can1 \
  --teleop.right_arm_config.port=can0 \
  --display_data=true \
  --display_ip=127.0.0.1 \
  --display_port=9876
```

When a field is not overridden from CLI, it falls back to values from `examples/songling_aloha/teleop.yaml`.

To reduce viewer load, limit the logging frequency:

```bash
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode \
  --can-poll-max-msgs 64 \
  --rerun-log-fps 5
```

If running remotely with a separate Rerun server:

```bash
# terminal A
rerun --serve-web --port 9876 --web-viewer-port 9090

# terminal B
python examples/songling_aloha/visualize_teleop_live.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --raw-can-mode \
  --display_ip 127.0.0.1 \
  --display_port 9876
```

### B. Record / Replay / Software teleop

`examples/songling_aloha/teleop.yaml` now carries default dataset recording fields too, so you can use
LeRobot-native record flags while keeping one unified config.

For the integrated Songling chain used in this repo, the recommended entry point is still
`record_compat.py`. It accepts the same dotted `lerobot-record`-style overrides, auto-injects the
default Songling config path, and when it detects the integrated CAN layout it forwards to the
raw-CAN recorder automatically.

Typical fixed-root collection command:

```bash
python examples/songling_aloha/record_compat.py \
  --config_path=examples/songling_aloha/teleop.yaml \
  --dataset.repo_id=your_hf_username/songling_aloha_demo \
  --dataset.root=outputs/songling_aloha_run_001 \
  --dataset.single_task="Your operation instructions" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false
```

If you want to reuse an existing directory exactly instead of auto-incrementing to a new suffix,
add these flags:

```bash
--resume=false \
--dataset.auto_increment_root=false
```

If you only want the shortest local command and are happy with YAML defaults for anything you do not
override:

```bash
python examples/songling_aloha/record_compat.py \
  --dataset.repo_id=your_hf_username/songling_aloha_demo \
  --dataset.single_task="Bimanual teleoperation with Songling ALOHA profile"
```

`record_compat.py` also normalizes the Songling unified YAML to a strict `lerobot-record`
schema at runtime (dropping non-record keys such as top-level `fps`), while preserving
native CLI override behavior (`CLI > YAML`) for `--robot.*`, `--teleop.*`, `--display_*`,
and `--dataset.*`.

If `--dataset.root` is not provided, `record_compat.py` defaults to:
- `dataset.root` from `examples/songling_aloha/teleop.yaml` if present
- otherwise `<repo>/outputs/songling_aloha`

If Hugging Face auth is not configured but `push_to_hub=true` is requested, it automatically
forces local-only recording (`--dataset.push_to_hub=false`).
Voice settings (`play_sounds`, `voice_*`) continue to come from `examples/songling_aloha/teleop.yaml`
and can be overridden from CLI.
Choose a writable path under your user directory or repo workspace, not a protected mount point without write permission.

The underlying raw-CAN recorder writes `meta/songling_recording_summary.json` for each dataset root.
That sidecar records per-episode control-frame ratios for left and right arms, which lets later
sanity checks tell whether dataset actions came from real CAN control frames or from observation
fallback.

Important unit note:

- on disk, this repo currently stores arm joints in degrees and the gripper in mm
- when comparing against `mobile-aloha` conventions, project those values as:
  - arm: `degree -> rad`
  - gripper: `mm -> normalized [0, 1]` using your verified gripper max stroke (commonly 70 mm or 100 mm)

`lerobot-replay` is still not the right tool for the integrated Songling chain, but this example
now includes a dedicated replay script:

```bash
python examples/songling_aloha/replay_episodes.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --dataset.root outputs/songling_aloha \
  --dataset.repo_id your_hf_username/songling_aloha_demo \
  --dataset.episode 0 \
  --display_data=true
```

This default mode is an offline episode replay: it walks the dataset at the original FPS and
re-logs recorded states, actions and camera frames to Rerun.

For the current integrated Songling CAN chain, use raw-CAN shadow replay instead of live command send:

```bash
python examples/songling_aloha/replay_episodes.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --dataset.root outputs/songling_aloha \
  --dataset.repo_id your_hf_username/songling_aloha_demo \
  --dataset.episode 0 \
  --raw-can-mode \
  --display_data=true
```

That mode replays the recorded episode timeline while simultaneously reading live CAN + cameras from
the current Songling setup, so you can compare:

- recorded dataset action/state
- live decoded observation
- live candidate action frames seen on the bus

If you later switch to a non-integrated OpenArm-compatible runtime config, the same script can send
recorded actions back to the robot:

```bash
python examples/songling_aloha/replay_episodes.py \
  --config-path path/to/non_integrated_openarm.yaml \
  --dataset.root outputs/songling_aloha \
  --dataset.repo_id your_hf_username/songling_aloha_demo \
  --dataset.episode 0 \
  --connect-live
```

An experimental Piper-V2-like Songling runtime config is now available at:

```bash
examples/songling_aloha/runtime_songling.yaml
```

Important:

- it is opt-in only and keeps `allow_unverified_commanding: false` by default
- to actually send real commands, set `robot.left_arm_config.allow_unverified_commanding=true`
  and `robot.right_arm_config.allow_unverified_commanding=true` after verifying the protocol on your hardware
- this new runtime assumes a Piper-like control path:
  - mode control: `0x151`
  - joint control: `0x155`, `0x156`, `0x157`
  - gripper control: `0x159`
  - feedback: `0x2A1`, `0x2A5`, `0x2A6`, `0x2A7`, `0x2A8`
- the earlier raw-CAN dataset path in this example used heuristic field extraction and may not match
  this new runtime's Piper-style units yet, so do not assume an old checkpoint or old dataset can be
  replayed directly on hardware without checking semantics first

Example ACT deployment with the experimental runtime:

```bash
python examples/songling_aloha/run_act_inference.py \
  --config-path examples/songling_aloha/runtime_songling.yaml \
  --policy.path=outputs/train/act_songling_plate_stack_v1 \
  --policy.device=cuda \
  --display_data=true
```

### C. ACT real-robot inference

For learned policy deployment on the Songling follower pair, use the dedicated
ACT runtime instead of `lerobot-record`. This path does not create an eval
dataset and does not require `dataset.single_task`.

The script below reuses the CAN and camera mapping from
`examples/songling_aloha/teleop.yaml`, while accepting familiar dotted CLI
overrides such as `--policy.path=...` and `--robot.left_arm_config.port=...`.

Important: the stock `teleop.yaml` in this example models the Songling
integrated leader+follower chain, where teleop and robot share the same CAN
port on each side. That hardware layout is not compatible with the generic
OpenArm/Damiao runtime yet, so real closed-loop ACT deployment still requires
a dedicated Songling MotorsBus / robot adapter.

You can still run online ACT shadow inference on the integrated chain by
reading observations from raw CAN plus cameras and visualizing predicted
actions without sending them back to the robot:

```bash
python examples/songling_aloha/run_act_inference.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --policy.path=outputs/train/act_songling_plate_stack_v1 \
  --policy.device=cuda \
  --raw-can-mode \
  --display_data=true
```

```bash
python examples/songling_aloha/run_act_inference.py \
  --config-path examples/songling_aloha/teleop.yaml \
  --policy.path=outputs/train/act_songling_plate_stack_v1 \
  --policy.device=cuda \
  --display_data=true
```

If you want to run a specific saved checkpoint instead of the exported model
directory, point `--policy.path` at that checkpoint's `pretrained_model`
directory:

```bash
python examples/songling_aloha/run_act_inference.py \
  --policy.path=outputs/train/act_songling_plate_stack_v1/checkpoints/040000/pretrained_model \
  --policy.device=cuda \
  --display_data=true
```

Useful optional overrides:

- `--duration=120` to stop automatically after 120 seconds
- `--display_data=false` to disable Rerun
- `--robot.left_arm_config.port=can1` and `--robot.right_arm_config.port=can0` to override YAML ports from CLI

### D. Sanity-check a new dataset

Run this immediately after collection, before training or replay:

```bash
python examples/songling_aloha/check_dataset_sanity.py \
  --dataset.root outputs/songling_aloha_run_001 \
  --dataset.repo_id your_hf_username/songling_aloha_demo \
  --gripper-max-mm 70
```

This checks:

- the 14-D bimanual schema expected by the current Songling example
- protocol-native ranges on disk (`degree` for arm joints, `mm` for gripper)
- a `mobile-aloha`-style projection (`rad` for arm joints, normalized gripper)
- whether recorded dataset actions mostly came from real control frames or from
  observation fallback, using `meta/songling_recording_summary.json` when available

If your hardware is using the larger gripper travel variant, rerun with:

```bash
--gripper-max-mm 100
```

### E. Visualize dataset

```bash
lerobot-dataset-viz \
  --repo-id your_hf_username/songling_aloha_demo \
  --root outputs/songling_aloha \
  --episode-index 0
```

## 4) Notes and troubleshooting

- If CAN setup fails, run `ip link show can0` (replace with your interface), then rerun `lerobot-setup-can --mode=setup`.
- If your adapter does not support CAN FD (e.g. `RTNETLINK answers: Operation not supported`), use `--use_fd=false`.
- If camera order changes after reboot, pin devices with udev rules and update the YAML once.
- Use `lerobot-find-port --write-songling-config examples/songling_aloha/teleop.yaml --songling-side <left|right>` to sync detected CAN ports back into the single config file.
- If one camera freezes when the scene moves quickly, first treat it as USB bandwidth/topology instability:
  - keep at most 2 cameras per USB hub (mobile-aloha recommendation),
  - avoid mixing all cameras behind a single low-quality hub,
  - prefer stable `/dev/v4l/by-id/...` nodes.
- For this script, use conservative runtime parameters first, then scale up:
  - `--rerun-log-fps 12`
  - `--camera-max-age-ms 800`
  - `--camera-retry-timeout-ms 120`
  - `--camera-reconnect-stale-count 24`
  - `--camera-reconnect-cooldown-s 2.0`
  - `--can-poll-max-msgs 32`
- If motion-triggered freezes persist, lower the problematic camera FPS in YAML (e.g. from 30 to 20) before changing other cameras.

## 5) Safety checklist for HIL tests

- Start with small-amplitude single-joint motions.
- Verify emergency stop access before long runs.
- Keep `max_relative_target` conservative before scaling speed/range.
