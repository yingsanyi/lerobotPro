#!/usr/bin/env python

"""Replay a Songling ALOHA dataset episode.

This script supports:

- offline replay: iterate one recorded episode and visualize recorded state/action/media
- live replay: send recorded actions to the Songling follower pair while showing live cameras/state

For integrated master/slave Songling chains, live replay assumes the hardware leader/master
is no longer actively commanding the follower. The script checks for pre-existing control echo
traffic and aborts by default if it looks like an external leader is still active.
"""

from __future__ import annotations

import argparse
import logging
import math
import numbers
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.bi_songling_follower import BiSonglingFollower
from lerobot.robots.songling_follower.protocol import DEFAULT_SONGLING_JOINT_NAMES
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

try:
    import rerun as rr
except ModuleNotFoundError:
    rr = None

try:
    from record_raw_can_dataset import _build_runtime_robot_config, _patch_multiprocess_resource_tracker
except ModuleNotFoundError:
    from examples.songling_aloha.record_raw_can_dataset import _build_runtime_robot_config, _patch_multiprocess_resource_tracker


DEFAULT_CONFIG_PATH = Path("examples/songling_aloha/teleop.yaml")
DEFAULT_SESSION_NAME = "songling_aloha_replay"
ACTION_KEY = "action"
OBS_STATE_KEY = "observation.state"
MISSING_STATUS_OR_MODE_REASON = "未收到真实状态帧，也未观察到 `0x151` 模式命令"
MISSING_DRIVER_ENABLE_REASON = "未收到驱动使能反馈"
ENABLE_PREPARE_OUTER_RETRIES = 3
ENABLE_PREPARE_RETRY_SETTLE_S = 0.05
MODE_RECOVERY_ATTEMPTS = 8
MODE_RECOVERY_SETTLE_S = 0.03
ENABLE_PRELOAD_REPEAT = 5
ENABLE_PRELOAD_SETTLE_S = 0.02
STARTUP_HOLD_REPEAT = 20
STARTUP_HOLD_SETTLE_S = 0.01
POST_REPLAY_HOLD_INTERVAL_S = 0.01
ZERO_TRAJECTORY_INTERVAL_S = 0.02
POST_REPLAY_ZERO_JOINT_STEP = 2.0
POST_REPLAY_ZERO_GRIPPER_STEP = 4.0
POST_REPLAY_ZERO_MIN_SETTLE_S = 0.05
POST_REPLAY_ZERO_MIN_ATTEMPTS = 120
POST_REPLAY_TRANSITION_HOLD_REPEAT = 50
POST_REPLAY_TRANSITION_HOLD_INTERVAL_S = 0.01
_RERUN_DISPLAY_ACTIVE = False

logger = logging.getLogger(__name__)


@dataclass
class DatasetReplayOptions:
    repo_id: str
    root: Path | None
    episode: int


def _parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false.")


def _load_songling_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import draccus

    with config_path.open("r", encoding="utf-8") as f:
        cfg = draccus.load(dict[str, Any], f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} is not a valid mapping.")
    return cfg


def _resolve_dataset_options(raw_cfg: dict[str, Any], args: argparse.Namespace) -> DatasetReplayOptions:
    raw_dataset = raw_cfg.get("dataset") if isinstance(raw_cfg.get("dataset"), dict) else {}

    root_value = args.dataset_root if args.dataset_root is not None else raw_dataset.get("root")
    repo_id = args.dataset_repo_id if args.dataset_repo_id is not None else raw_dataset.get("repo_id")
    episode = args.dataset_episode if args.dataset_episode is not None else raw_dataset.get("episode", 0)

    root = None if root_value is None else Path(root_value).expanduser().resolve()
    if repo_id is None and root is not None:
        repo_id = f"local/{root.name}"
        logger.info("No --dataset.repo_id provided, using derived local repo id: %s", repo_id)
    if repo_id is None:
        raise ValueError(
            "Missing dataset repo id. Set --dataset.repo_id=..., or provide --dataset.root so a local id can be derived."
        )

    return DatasetReplayOptions(repo_id=str(repo_id), root=root, episode=int(episode))


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (float, numbers.Real, np.integer, np.floating)) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )


def _log_value(path: str, value: Any, *, compress_images: bool) -> None:
    if not _rerun_display_is_active() or value is None:
        return

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            value = value.item()
        else:
            value = value.numpy()

    if _is_scalar(value):
        _safe_rr_log(path, rr.Scalars(float(value)))
        return

    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 1:
            for i, item in enumerate(arr):
                _safe_rr_log(f"{path}/{i}", rr.Scalars(float(item)))
            return

        image = rr.Image(arr).compress() if compress_images else rr.Image(arr)
        _safe_rr_log(path, image)
        return

    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            if _is_scalar(item):
                _safe_rr_log(f"{path}/{i}", rr.Scalars(float(item)))


def _rerun_display_is_active() -> bool:
    return bool(rr is not None and _RERUN_DISPLAY_ACTIVE)


def _disable_rerun_display(reason: str) -> None:
    global _RERUN_DISPLAY_ACTIVE
    if not _RERUN_DISPLAY_ACTIVE:
        return
    logger.warning("Disable Rerun display for the rest of replay: %s", reason)
    _RERUN_DISPLAY_ACTIVE = False
    _shutdown_rerun(True)


def _safe_rr_log(path: str, *args: Any, **kwargs: Any) -> bool:
    if not _rerun_display_is_active():
        return False
    try:
        rr.log(path, *args, **kwargs)
        return True
    except Exception as exc:
        _disable_rerun_display(f"rr.log failed at {path}: {exc}")
        return False


def _safe_rr_set_time(timeline: str, *, sequence: int) -> bool:
    if not _rerun_display_is_active():
        return False
    try:
        rr.set_time(timeline, sequence=sequence)
        return True
    except Exception as exc:
        _disable_rerun_display(f"rr.set_time failed on timeline {timeline}: {exc}")
        return False


def _log_bundle(
    namespace: str,
    *,
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    compress_images: bool,
) -> None:
    if observation:
        for key, value in observation.items():
            _log_value(f"{namespace}/observation/{key}", value, compress_images=compress_images)
    if action:
        for key, value in action.items():
            _log_value(f"{namespace}/action/{key}", value, compress_images=compress_images)


def _named_vector(values: np.ndarray | torch.Tensor, names: list[str]) -> dict[str, float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {array.shape}.")
    if len(array) != len(names):
        raise ValueError(f"Vector length {len(array)} does not match feature names length {len(names)}.")
    return {name: float(array[i]) for i, name in enumerate(names)}


def _extract_action_dict(frame_row: dict[str, Any], action_names: list[str]) -> dict[str, float]:
    return _named_vector(frame_row[ACTION_KEY], action_names)


def _extract_replay_observation(
    frame_row: dict[str, Any],
    *,
    state_names: list[str],
    camera_keys: list[str],
) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    if OBS_STATE_KEY in frame_row:
        observation.update(_named_vector(frame_row[OBS_STATE_KEY], state_names))
    for camera_key in camera_keys:
        if camera_key in frame_row:
            observation[camera_key] = frame_row[camera_key]
    return observation


def _slice_episode_indices(*, total: int, start_frame: int, max_frames: int | None) -> tuple[list[int], int]:
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0.")
    if start_frame >= total:
        raise ValueError(f"--start-frame={start_frame} is out of range for episode length {total}.")

    stop_frame = total if max_frames is None else min(total, start_frame + max_frames)
    return list(range(start_frame, stop_frame)), stop_frame


def _get_replay_item(dataset: LeRobotDataset, idx: int, *, include_media: bool) -> dict[str, Any]:
    if include_media:
        return dataset[idx]
    return dataset.hf_dataset[idx]


def _maybe_init_rerun(
    *,
    display_data: bool,
    session_name: str,
    display_ip: str | None,
    display_port: int | None,
) -> None:
    global _RERUN_DISPLAY_ACTIVE
    _RERUN_DISPLAY_ACTIVE = False
    if not display_data:
        return
    if rr is None:
        raise ModuleNotFoundError("rerun-sdk is required for --display-data=true. Please install rerun-sdk.")
    if display_ip is not None and display_port is not None:
        try:
            with socket.create_connection((str(display_ip), int(display_port)), timeout=1.0):
                pass
            init_rerun(session_name=session_name, ip=display_ip, port=display_port)
            _RERUN_DISPLAY_ACTIVE = True
            return
        except OSError:
            logger.warning(
                "Rerun endpoint %s:%s is not reachable. Falling back to local viewer spawn.",
                display_ip,
                display_port,
            )
    init_rerun(session_name=session_name, ip=None, port=None)
    _RERUN_DISPLAY_ACTIVE = True


def _shutdown_rerun(display_data: bool) -> None:
    global _RERUN_DISPLAY_ACTIVE
    _RERUN_DISPLAY_ACTIVE = False
    if display_data and rr is not None:
        try:
            rr.rerun_shutdown()
        except Exception:
            pass


def _run_offline_replay(
    *,
    replay_dataset: LeRobotDataset,
    replay_indices: list[int],
    playback_fps: int,
    action_names: list[str],
    state_names: list[str],
    camera_keys: list[str],
    display_data: bool,
    compress_images: bool,
    play_sounds: bool,
    start_frame: int,
) -> None:
    last_log = 0.0

    if play_sounds:
        log_say("Replaying episode", play_sounds, blocking=True)

    logger.info("Starting offline replay. frames=%d fps=%d", len(replay_indices), playback_fps)
    for local_idx, replay_idx in enumerate(replay_indices):
        loop_start = time.perf_counter()
        global_frame = start_frame + local_idx
        replay_item = _get_replay_item(replay_dataset, replay_idx, include_media=display_data)
        action = _extract_action_dict(replay_item, action_names)

        if display_data and _safe_rr_set_time("frame", sequence=global_frame):
            replay_observation = _extract_replay_observation(
                replay_item,
                state_names=state_names,
                camera_keys=camera_keys,
            )
            _log_bundle(
                "replay",
                observation=replay_observation,
                action=action,
                compress_images=compress_images,
            )

        now = time.time()
        if now - last_log >= 1.0:
            logger.info(
                "frame=%d/%d recorded_left_joint_1=%.3f recorded_right_joint_1=%.3f",
                global_frame,
                start_frame + len(replay_indices) - 1,
                action.get("left_joint_1.pos", float("nan")),
                action.get("right_joint_1.pos", float("nan")),
            )
            last_log = now

        precise_sleep(max(1.0 / playback_fps - (time.perf_counter() - loop_start), 0.0))


def _make_live_replay_robot_config(raw_cfg: dict[str, Any], args: argparse.Namespace, playback_fps: int):
    cfg = _build_runtime_robot_config(raw_cfg, args, fallback_fps=playback_fps)
    # Live replay already applies its own per-frame step limiter. Disable the lower-level
    # relative target clamp by default to avoid replay path distortion from stacked limits.
    if args.max_relative_target is not None:
        max_relative_target = None if float(args.max_relative_target) < 0.0 else float(args.max_relative_target)
    else:
        max_relative_target = None
    cfg.left_arm_config.max_relative_target = max_relative_target
    cfg.right_arm_config.max_relative_target = max_relative_target
    if args.speed_percent is not None:
        speed_percent = int(args.speed_percent)
        cfg.left_arm_config.speed_percent = speed_percent
        cfg.right_arm_config.speed_percent = speed_percent
    command_repeat = 2 if args.command_repeat is None else max(int(args.command_repeat), 1)
    cfg.left_arm_config.command_repeat = command_repeat
    cfg.right_arm_config.command_repeat = command_repeat
    if args.command_interval_s is not None:
        command_interval_s = max(float(args.command_interval_s), 0.0)
        cfg.left_arm_config.command_interval_s = command_interval_s
        cfg.right_arm_config.command_interval_s = command_interval_s
    for arm_cfg in (cfg.left_arm_config, cfg.right_arm_config):
        arm_cfg.allow_unverified_commanding = True
        # Keep replay in pure motion-command mode only. These fields must never be allowed
        # to downlink persistent parameter-like settings to the arm.
        arm_cfg.installation_pos = None
        arm_cfg.gripper_set_zero = 0
        arm_cfg.leader_follower_role = None
        arm_cfg.leader_follower_feedback_offset = 0x00
        arm_cfg.leader_follower_ctrl_offset = 0x00
        arm_cfg.leader_follower_linkage_offset = 0x00
        # Delay any command-mode / motor-enable writes until follower-only safety check passes.
        arm_cfg.auto_enable_on_connect = False
        arm_cfg.auto_configure_mode_on_connect = False
        arm_cfg.auto_configure_master_slave_on_connect = False
        # Re-send mode command on every action send in live replay (safety against mode drop).
        arm_cfg.mode_keepalive_s = 0.0
        # Keep disconnect non-destructive in replay flow even if yaml was overridden elsewhere.
        arm_cfg.disable_torque_on_disconnect = False
    return cfg


def _log_parameter_write_safety_lock(robot_cfg: Any) -> None:
    for side, arm_cfg in (("left", robot_cfg.left_arm_config), ("right", robot_cfg.right_arm_config)):
        logger.info(
            "%s replay parameter-write safety lock: installation_pos=%r gripper_set_zero=%r "
            "auto_configure_master_slave_on_connect=%r leader_follower_role=%r",
            _side_label(side),
            getattr(arm_cfg, "installation_pos", None),
            getattr(arm_cfg, "gripper_set_zero", None),
            getattr(arm_cfg, "auto_configure_master_slave_on_connect", None),
            getattr(arm_cfg, "leader_follower_role", None),
        )


def _poll_robot_buses(robot: BiSonglingFollower) -> None:
    robot.left_arm.bus.poll(max_msgs=robot.left_arm.bus.config.poll_max_msgs)
    robot.right_arm.bus.poll(max_msgs=robot.right_arm.bus.config.poll_max_msgs)


def _detect_external_leader_activity(robot: BiSonglingFollower, *, observe_s: float) -> dict[str, bool]:
    _poll_robot_buses(robot)
    left_initial = float(getattr(robot.left_arm.bus, "command_feedback_timestamp", 0.0) or 0.0)
    right_initial = float(getattr(robot.right_arm.bus, "command_feedback_timestamp", 0.0) or 0.0)
    activity = {"left": False, "right": False}
    deadline = time.perf_counter() + max(observe_s, 0.0)

    while time.perf_counter() < deadline:
        _poll_robot_buses(robot)
        if float(getattr(robot.left_arm.bus, "command_feedback_timestamp", 0.0) or 0.0) > left_initial:
            activity["left"] = True
        if float(getattr(robot.right_arm.bus, "command_feedback_timestamp", 0.0) or 0.0) > right_initial:
            activity["right"] = True
        if activity["left"] or activity["right"]:
            break
        precise_sleep(0.02)

    return activity


def _show_big_safety_popup(title: str, message: str) -> None:
    shown = False
    has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    if has_display:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.title(title)
            root.configure(bg="#8B0000")
            root.attributes("-topmost", True)
            root.geometry("980x560")

            headline = tk.Label(
                root,
                text=title,
                font=("Noto Sans CJK SC", 34, "bold"),
                fg="white",
                bg="#8B0000",
                wraplength=920,
                justify="center",
            )
            headline.pack(pady=(36, 20), padx=20)

            body = tk.Message(
                root,
                text=message,
                width=920,
                font=("Noto Sans CJK SC", 20, "bold"),
                fg="white",
                bg="#8B0000",
                justify="center",
            )
            body.pack(pady=(0, 30), padx=20)

            def _close() -> None:
                root.destroy()

            btn = tk.Button(
                root,
                text="我已知晓，立即退出重播",
                command=_close,
                font=("Noto Sans CJK SC", 20, "bold"),
                bg="#FFD54F",
                fg="black",
                width=20,
                height=2,
            )
            btn.pack(pady=(0, 34))
            root.protocol("WM_DELETE_WINDOW", _close)
            root.mainloop()
            shown = True
        except Exception:
            shown = False

    if (not shown) and has_display and shutil.which("zenity"):
        try:
            subprocess.run(
                [
                    "zenity",
                    "--error",
                    "--title",
                    title,
                    "--width",
                    "920",
                    "--height",
                    "520",
                    "--text",
                    message,
                ],
                check=False,
            )
            shown = True
        except Exception:
            shown = False

    if not shown:
        banner = "=" * 90
        logger.error("\n%s\n%s\n%s\n%s", banner, title, message, banner)


def _side_label(side: str) -> str:
    if side == "left":
        return "左臂"
    if side == "right":
        return "右臂"
    return side


def _connected_bus_sides(robot: BiSonglingFollower) -> set[str]:
    sides: set[str] = set()
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        if bool(getattr(arm.bus, "is_connected", False)):
            sides.add(side)
    return sides


def _arm_ready_for_command(side: str, arm: Any) -> tuple[bool, str | None]:
    arm.bus.poll()
    expected_ctrl_mode = int(getattr(arm.config, "ctrl_mode", -1))
    status_valid = bool(getattr(arm.bus, "status_feedback_valid", False))
    mode_valid = bool(getattr(arm.bus, "mode_command_valid", False))
    status_ctrl_mode = int(getattr(arm.bus, "status", {}).get("ctrl_mode", -1)) if status_valid else None
    mode_ctrl_mode = int(getattr(arm.bus, "mode_command", {}).get("ctrl_mode", -1)) if mode_valid else None

    if expected_ctrl_mode >= 0:
        status_matches = bool(status_valid and status_ctrl_mode == expected_ctrl_mode)
        mode_matches = bool(mode_valid and mode_ctrl_mode == expected_ctrl_mode)
        if not (status_matches or mode_matches):
            if not status_valid and not mode_valid:
                return False, f"{_side_label(side)} {MISSING_STATUS_OR_MODE_REASON}"
            status_text = "--" if status_ctrl_mode is None else f"0x{status_ctrl_mode:x}"
            mode_text = "--" if mode_ctrl_mode is None else f"0x{mode_ctrl_mode:x}"
            return (
                False,
                f"{_side_label(side)} 控制模式未进入预期值（expected=0x{expected_ctrl_mode:x}, status={status_text}, mode151={mode_text}）",
            )

    enable_status = arm.bus.get_driver_enable_status()
    known_states = [state for state in enable_status.values() if state is not None]
    if not known_states:
        return False, f"{_side_label(side)} {MISSING_DRIVER_ENABLE_REASON}"
    if not all(bool(state) for state in known_states):
        return False, f"{_side_label(side)} 仍有驱动未使能"
    return True, None


def _reason_is_missing_status_or_mode(reason: str | None) -> bool:
    return bool(reason and MISSING_STATUS_OR_MODE_REASON in reason)


def _arm_ready_for_zero_pose(side: str, arm: Any) -> tuple[bool, str | None]:
    ready, reason = _arm_ready_for_command(side, arm)
    if ready:
        return True, None
    if not _reason_is_missing_status_or_mode(reason):
        return False, reason
    if not bool(getattr(arm.config, "allow_unverified_commanding", False)):
        return False, reason
    enable_status = arm.bus.get_driver_enable_status()
    known_states = [state for state in enable_status.values() if state is not None]
    if not known_states or not all(bool(state) for state in known_states):
        return False, reason
    if not arm.bus.has_reliable_position_feedback():
        return False, reason
    return True, f"{_side_label(side)} {MISSING_STATUS_OR_MODE_REASON}；驱动已使能且关节反馈可靠，按兜底策略继续。"


def _capture_best_effort_arm_pose(arm: Any) -> dict[str, float] | None:
    try:
        arm.bus.poll()
    except Exception:
        pass

    positions = arm.bus.get_positions(poll=False)
    commanded = arm.bus.get_commanded_positions(poll=False)
    hold_pose: dict[str, float] = {}
    seen_any = False
    for joint_name in DEFAULT_SONGLING_JOINT_NAMES:
        if bool(arm.bus.joint_position_seen.get(joint_name, False)):
            hold_pose[joint_name] = float(positions[joint_name])
            seen_any = True
        elif bool(arm.bus.commanded_position_seen.get(joint_name, False)):
            hold_pose[joint_name] = float(commanded[joint_name])
            seen_any = True
        else:
            hold_pose[joint_name] = 0.0
    return hold_pose if seen_any else None


def _send_arm_hold_pose(
    arm: Any,
    hold_pose: dict[str, float] | None,
    *,
    repeat: int,
    settle_s: float,
) -> None:
    if hold_pose is None:
        return
    for _ in range(max(int(repeat), 1)):
        try:
            arm.hold_position(hold_pose)
        except Exception:
            break
        if settle_s > 0:
            time.sleep(settle_s)


def _prepare_single_arm_for_motion(side: str, arm: Any) -> tuple[bool, str | None]:
    attempts = max(int(ENABLE_PREPARE_OUTER_RETRIES), 1)
    retry_settle_s = max(float(ENABLE_PREPARE_RETRY_SETTLE_S), 0.0)
    last_reason: str | None = None

    for attempt in range(1, attempts + 1):
        pre_enable_pose = _capture_best_effort_arm_pose(arm)
        # Do not send SDK "emergency resume" during startup recovery here.
        # On some Piper/Songling chains the underlying 0x150/0x02 path behaves like
        # a reset/drop-power transition, which can immediately disable joints.
        try:
            arm.bus.ensure_can_command_mode(force=True)
        except Exception:
            pass
        try:
            _ = arm.bus.wait_for_driver_enable_status(enabled=True)
        except Exception as exc:
            last_reason = f"{_side_label(side)} 使能握手异常: {exc}"

        try:
            arm.bus.ensure_can_command_mode(force=True)
        except Exception:
            pass

        _send_arm_hold_pose(arm, pre_enable_pose, repeat=STARTUP_HOLD_REPEAT, settle_s=STARTUP_HOLD_SETTLE_S)

        ready, reason = _arm_ready_for_command(side, arm)
        if ready:
            return True, None

        if (
            _reason_is_missing_status_or_mode(reason)
            and bool(getattr(arm.config, "allow_unverified_commanding", False))
            and arm.bus.has_reliable_position_feedback()
        ):
            enable_status = arm.bus.get_driver_enable_status()
            known_states = [state for state in enable_status.values() if state is not None]
            if known_states and all(bool(state) for state in known_states):
                logger.warning(
                    "%s 启动阶段未收到状态/0x151 回读，但驱动已使能且关节反馈可靠，按兜底策略放行。",
                    _side_label(side),
                )
                return True, None

        last_reason = reason or f"{_side_label(side)} 使能后仍未进入可控状态"
        if attempt < attempts and retry_settle_s > 0:
            time.sleep(retry_settle_s)

    return False, last_reason


def _preload_hold_current_pose(robot: BiSonglingFollower, *, prepared_sides: set[str]) -> None:
    if not prepared_sides:
        return
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        if side not in prepared_sides:
            continue
        hold_pose = _capture_best_effort_arm_pose(arm)
        if hold_pose is None:
            logger.warning("Skip startup hold preload on %s: neither position feedback nor command echo is ready yet.", _side_label(side))
            continue
        _send_arm_hold_pose(arm, hold_pose, repeat=ENABLE_PRELOAD_REPEAT, settle_s=ENABLE_PRELOAD_SETTLE_S)


def _prepare_live_robot(robot: BiSonglingFollower, *, strict: bool) -> list[str]:
    prepared_sides: list[str] = []
    failed_sides: dict[str, str] = {}

    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        ready, reason = _prepare_single_arm_for_motion(side, arm)
        if ready:
            prepared_sides.append(side)
        else:
            failed_sides[side] = reason or "unknown"

    _preload_hold_current_pose(robot, prepared_sides=set(prepared_sides))

    if failed_sides:
        summary = "; ".join(f"{_side_label(side)}: {reason}" for side, reason in failed_sides.items())
        if strict:
            raise RuntimeError(f"Live replay prepare failed: {summary}")
        logger.warning("Live replay prepare partially failed: %s", summary)
    return prepared_sides


def _force_mode_and_enable_for_side(side: str, arm: Any) -> tuple[bool, str | None]:
    attempts = max(int(MODE_RECOVERY_ATTEMPTS), 1)
    settle_s = max(float(MODE_RECOVERY_SETTLE_S), 0.0)
    last_reason: str | None = None

    for attempt in range(1, attempts + 1):
        # Keep runtime recovery non-destructive: only refresh mode/enable, never send
        # the SDK emergency-resume/reset-style 0x150 transition automatically.
        try:
            arm.bus.ensure_can_command_mode(force=True)
        except Exception:
            pass
        try:
            _ = arm.bus.wait_for_driver_enable_status(enabled=True)
        except Exception:
            pass
        try:
            arm.bus.ensure_can_command_mode(force=True)
        except Exception:
            pass

        ready, reason = _arm_ready_for_command(side, arm)
        if ready:
            return True, None
        last_reason = reason or f"{_side_label(side)} mode/enable recovery failed"
        if attempt < attempts and settle_s > 0:
            time.sleep(settle_s)

    return False, last_reason


def _ensure_runtime_motion_ready(
    robot: BiSonglingFollower,
    *,
    sides: set[str],
    phase: str,
) -> None:
    _disable_live_camera_streams_if_unhealthy(robot, phase=phase)
    failures: list[str] = []
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        if side not in sides:
            continue

        ready, reason = _arm_ready_for_command(side, arm)
        if ready:
            continue

        ready, reason = _force_mode_and_enable_for_side(side, arm)
        if ready:
            logger.warning("%s runtime guard recovered %s from transient not-ready state.", phase, _side_label(side))
            continue
        failures.append(reason or f"{_side_label(side)} unknown runtime not-ready")

    if failures:
        raise RuntimeError(f"{phase} aborted: {'; '.join(failures)}")


def _iter_live_cameras(robot: BiSonglingFollower) -> list[tuple[str, Any]]:
    cameras: list[tuple[str, Any]] = []
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        for camera_key, camera in arm.cameras.items():
            cameras.append((f"{side}_{camera_key}", camera))
    return cameras


def _safe_disconnect_camera(camera: Any) -> None:
    disconnect = getattr(camera.disconnect, "__wrapped__", None)
    try:
        if callable(disconnect):
            disconnect(camera)
        else:
            camera.disconnect()
    except Exception:
        pass


def _safe_connect_camera(camera: Any) -> bool:
    try:
        camera.connect(warmup=True)
        return True
    except Exception:
        return False


def _disable_live_camera_streams(robot: BiSonglingFollower, *, reason: str) -> bool:
    camera_items = _iter_live_cameras(robot)
    if not camera_items:
        return False

    logger.warning("Disable live camera streams for the rest of replay: %s", reason)
    for _camera_name, camera in camera_items:
        _safe_disconnect_camera(camera)
    robot.left_arm.cameras = {}
    robot.right_arm.cameras = {}
    robot.cameras = {}
    return True


def _defer_live_camera_connect(robot: BiSonglingFollower) -> tuple[dict[str, Any], dict[str, Any]]:
    left_cameras = dict(robot.left_arm.cameras)
    right_cameras = dict(robot.right_arm.cameras)
    if left_cameras or right_cameras:
        logger.info("Defer live camera connection until after startup enable/zero so arm stabilization starts first.")
    robot.left_arm.cameras = {}
    robot.right_arm.cameras = {}
    robot.cameras = {}
    return left_cameras, right_cameras


def _attach_and_connect_live_cameras(
    robot: BiSonglingFollower,
    *,
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
) -> bool:
    if not left_cameras and not right_cameras:
        return False

    connected_left: dict[str, Any] = {}
    connected_right: dict[str, Any] = {}
    failures: list[str] = []
    for side, camera_items, bucket in (
        ("left", left_cameras.items(), connected_left),
        ("right", right_cameras.items(), connected_right),
    ):
        for camera_key, camera in camera_items:
            if _safe_connect_camera(camera):
                bucket[camera_key] = camera
            else:
                failures.append(f"{side}_{camera_key}")

    robot.left_arm.cameras = connected_left
    robot.right_arm.cameras = connected_right
    robot.cameras = {**connected_left, **connected_right}

    if failures:
        logger.warning("Some deferred live cameras failed to connect and will stay disabled: %s", ",".join(failures))
    return bool(robot.cameras)


def _find_unhealthy_live_cameras(robot: BiSonglingFollower) -> list[str]:
    unhealthy: list[str] = []
    for camera_name, camera in _iter_live_cameras(robot):
        try:
            if not bool(getattr(camera, "is_connected", False)):
                unhealthy.append(f"{camera_name}: disconnected")
                continue
            thread = getattr(camera, "thread", None)
            if thread is not None and not thread.is_alive():
                unhealthy.append(f"{camera_name}: read thread stopped")
        except Exception as exc:
            unhealthy.append(f"{camera_name}: health check failed ({exc})")
    return unhealthy


def _disable_live_camera_streams_if_unhealthy(robot: BiSonglingFollower, *, phase: str) -> bool:
    unhealthy = _find_unhealthy_live_cameras(robot)
    if not unhealthy:
        return False
    return _disable_live_camera_streams(robot, reason=f"{phase}: {'; '.join(unhealthy)}")


def _send_action_even_if_camera_dropped(
    robot: BiSonglingFollower,
    action: dict[str, float],
) -> dict[str, float]:
    try:
        return robot.send_action(action)
    except DeviceNotConnectedError as exc:
        left_bus_connected = bool(getattr(robot.left_arm.bus, "is_connected", False))
        right_bus_connected = bool(getattr(robot.right_arm.bus, "is_connected", False))
        if not (left_bus_connected and right_bus_connected):
            raise

        _disable_live_camera_streams(robot, reason=f"composite robot connection dropped: {exc}")
        try:
            return robot.send_action(action)
        except DeviceNotConnectedError:
            logger.warning(
                "Robot composite connection remained unavailable after disabling cameras; fallback to direct arm send. detail=%s",
                exc,
            )
        left_action = {key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")}
        right_action = {key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")}

        left_send = getattr(robot.left_arm.send_action, "__wrapped__", None)
        right_send = getattr(robot.right_arm.send_action, "__wrapped__", None)
        if callable(left_send):
            sent_left = left_send(robot.left_arm, left_action)
        else:
            sent_left = robot.left_arm.send_action(left_action)
        if callable(right_send):
            sent_right = right_send(robot.right_arm, right_action)
        else:
            sent_right = robot.right_arm.send_action(right_action)
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }


def _safe_disconnect_robot(robot: BiSonglingFollower) -> None:
    for arm in (robot.left_arm, robot.right_arm):
        disconnect = getattr(arm.disconnect, "__wrapped__", None)
        try:
            if callable(disconnect):
                disconnect(arm)
            else:
                arm.disconnect()
        except Exception:
            pass


def _detach_robot_without_disconnect(robot: BiSonglingFollower, *, reason: str) -> None:
    logger.warning(
        "Leave live robot enabled on exit to avoid joint droop; skip explicit bus disconnect. reason=%s",
        reason,
    )
    _disable_live_camera_streams(robot, reason=f"detach without robot disconnect: {reason}")
    for arm in (robot.left_arm, robot.right_arm):
        try:
            arm.cameras = {}
        except Exception:
            pass
        bus = getattr(arm, "bus", None)
        if bus is None:
            continue
        try:
            setattr(bus, "_is_connected", False)
        except Exception:
            pass
    robot.cameras = {}


def _finalize_live_robot(
    robot: BiSonglingFollower,
    *,
    args: argparse.Namespace,
    action_names: list[str],
) -> None:
    if bool(getattr(args, "disconnect_live_robot_on_exit", False)):
        _safe_disconnect_robot(robot)
        return

    try:
        _send_transition_hold_current_pose(robot, action_names=action_names)
    except Exception as exc:
        logger.warning("Final transition hold before exit failed: %s", exc)
    _detach_robot_without_disconnect(robot, reason="live replay cleanup")


def _side_is_near_zero_pose(
    observation: dict[str, float],
    *,
    side: str,
    joint_tolerance: float,
    gripper_tolerance: float,
) -> bool:
    for joint_name in DEFAULT_SONGLING_JOINT_NAMES:
        key = f"{side}_{joint_name}.pos"
        if key not in observation:
            return False
        tolerance = gripper_tolerance if joint_name == "gripper" else joint_tolerance
        if not math.isclose(float(observation[key]), 0.0, abs_tol=tolerance):
            return False
    return True


def _run_pre_replay_enable_and_zero(
    robot: BiSonglingFollower,
    *,
    args: argparse.Namespace,
    action_names: list[str],
    prepared_sides: set[str] | None = None,
) -> None:
    if prepared_sides is None:
        prepared_sides = set(_prepare_live_robot(robot, strict=True))
    else:
        prepared_sides = set(prepared_sides)
    if len(prepared_sides) < 2:
        raise RuntimeError(
            "Live replay startup requires both arms enabled and stable before motion, "
            f"but only ready side(s)={','.join(sorted(prepared_sides)) or 'none'}."
        )
    if not args.pre_replay_enable_and_zero:
        logger.info("Pre-replay enable+zero is disabled by CLI; skipping zero-pose routine.")
        return

    ready_sides: list[str] = []
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        if side not in prepared_sides:
            continue
        ready, reason = _arm_ready_for_zero_pose(side, arm)
        if not ready:
            logger.warning("Skip pre-replay zero pose on %s: %s", _side_label(side), reason)
            continue
        if reason is not None:
            logger.warning("Pre-replay zero pose on %s uses fallback readiness: %s", _side_label(side), reason)
        if not arm.bus.has_reliable_position_feedback():
            logger.warning("Skip pre-replay zero pose on %s: joint position feedback is not reliable yet.", _side_label(side))
            continue
        ready_sides.append(side)

    if not ready_sides:
        logger.warning("No arm is ready for pre-replay zero pose; continue directly to replay.")
        return

    max_attempts = max(int(args.zero_max_attempts), 1)
    settle_s = max(float(args.zero_settle_s), 0.0)
    joint_tolerance = max(float(args.zero_joint_tolerance), 0.0)
    gripper_tolerance = max(float(args.zero_gripper_tolerance), 0.0)

    live_observation = _get_live_joint_observation(robot)
    zero_action = {name: float(live_observation.get(name, 0.0)) for name in action_names}
    for side in ready_sides:
        for joint_name in DEFAULT_SONGLING_JOINT_NAMES:
            key = f"{side}_{joint_name}.pos"
            if key in zero_action:
                zero_action[key] = 0.0

    logger.info(
        "Pre-replay routine: move to zero pose after enabling. sides=%s max_attempts=%d joint_tol=%.3f gripper_tol=%.3f",
        ",".join(ready_sides),
        max_attempts,
        joint_tolerance,
        gripper_tolerance,
    )

    pending_sides = set(ready_sides)
    for attempt in range(1, max_attempts + 1):
        _ensure_runtime_motion_ready(
            robot,
            sides=set(pending_sides),
            phase=f"pre-replay zero attempt {attempt}/{max_attempts}",
        )
        sent_action = _send_action_even_if_camera_dropped(robot, zero_action)
        for key, value in sent_action.items():
            if key in zero_action:
                zero_action[key] = float(value)
        if settle_s > 0:
            time.sleep(settle_s)
        live_observation = _get_live_joint_observation(robot)
        pending_sides = {
            side
            for side in pending_sides
            if not _side_is_near_zero_pose(
                live_observation,
                side=side,
                joint_tolerance=joint_tolerance,
                gripper_tolerance=gripper_tolerance,
            )
        }
        if not pending_sides:
            logger.info("Pre-replay zero pose reached for side(s): %s", ",".join(ready_sides))
            return
        if attempt == 1 or attempt % 5 == 0 or attempt == max_attempts:
            logger.info(
                "Pre-replay zero in progress: attempt=%d/%d pending=%s left_joint_1=%.3f right_joint_1=%.3f",
                attempt,
                max_attempts,
                ",".join(sorted(pending_sides)),
                float(live_observation.get("left_joint_1.pos", float("nan"))),
                float(live_observation.get("right_joint_1.pos", float("nan"))),
            )

    logger.warning(
        "Pre-replay zero pose did not fully converge within %d attempts. pending=%s. Continue to replay.",
        max_attempts,
        ",".join(sorted(pending_sides)),
    )


def _run_post_replay_return_to_zero(
    robot: BiSonglingFollower,
    *,
    args: argparse.Namespace,
    action_names: list[str],
) -> dict[str, float] | None:
    if not args.post_replay_return_to_zero:
        logger.info("Post-replay return-to-zero is disabled by CLI; skipping.")
        return None
    hold_action = _send_transition_hold_current_pose(robot, action_names=action_names)
    pending_sides = _connected_bus_sides(robot)
    if not pending_sides:
        logger.warning("No arm reached a stable enabled state after replay; skip post-replay return-to-zero.")
        return hold_action

    max_attempts = max(int(args.zero_max_attempts), POST_REPLAY_ZERO_MIN_ATTEMPTS)
    settle_s = max(float(args.zero_settle_s), POST_REPLAY_ZERO_MIN_SETTLE_S)
    joint_tolerance = max(float(args.zero_joint_tolerance), 0.0)
    gripper_tolerance = max(float(args.zero_gripper_tolerance), 0.0)

    logger.info(
        "Post-replay routine: return to zero pose slowly. sides=%s max_attempts=%d joint_tol=%.3f "
        "gripper_tol=%.3f joint_step=%.3f gripper_step=%.3f",
        ",".join(sorted(pending_sides)),
        max_attempts,
        joint_tolerance,
        gripper_tolerance,
        POST_REPLAY_ZERO_JOINT_STEP,
        POST_REPLAY_ZERO_GRIPPER_STEP,
    )

    target_sides = set(pending_sides)
    abandoned_sides: set[str] = set()
    for attempt in range(1, max_attempts + 1):
        connected_sides = _connected_bus_sides(robot)
        abandoned_sides.update(pending_sides - connected_sides)
        pending_sides &= connected_sides
        if not pending_sides:
            logger.warning(
                "All CAN buses disconnected during post-replay return-to-zero; stop zero routine. lost=%s",
                ",".join(sorted(abandoned_sides)) if abandoned_sides else "none",
            )
            return hold_action
        try:
            _ensure_runtime_motion_ready(
                robot,
                sides=set(pending_sides),
                phase=f"post-replay zero attempt {attempt}/{max_attempts}",
            )
        except RuntimeError as exc:
            logger.warning(
                "Post-replay zero recovery is still in progress on attempt %d/%d: %s",
                attempt,
                max_attempts,
                exc,
            )
            hold_action = _send_repeated_hold_action(robot, hold_action, repeat=1, interval_s=0.0)
            if settle_s > 0:
                precise_sleep(settle_s)
            continue

        live_observation = _get_live_joint_observation(robot)
        send_sides = {
            side
            for side, arm in (("left", robot.left_arm), ("right", robot.right_arm))
            if side in pending_sides and arm.bus.has_reliable_position_feedback()
        }
        if not send_sides:
            if attempt == 1 or attempt % 5 == 0 or attempt == max_attempts:
                logger.warning(
                    "Post-replay zero is waiting for reliable position feedback. pending=%s",
                    ",".join(sorted(pending_sides)),
                )
            hold_action = _send_repeated_hold_action(robot, hold_action, repeat=1, interval_s=0.0)
            if settle_s > 0:
                precise_sleep(settle_s)
            continue

        zero_action = _build_zero_return_action(
            action_names=action_names,
            live_observation=live_observation,
            sides=set(send_sides),
        )
        hold_action = _send_action_even_if_camera_dropped(robot, zero_action)
        if settle_s > 0:
            precise_sleep(settle_s)
        live_observation = _get_live_joint_observation(robot)
        pending_sides = {
            side
            for side in pending_sides
            if not _side_is_near_zero_pose(
                live_observation,
                side=side,
                joint_tolerance=joint_tolerance,
                gripper_tolerance=gripper_tolerance,
            )
        }
        if not pending_sides:
            completed_sides = sorted(target_sides - abandoned_sides)
            if abandoned_sides:
                logger.warning(
                    "Post-replay return-to-zero reached for completed side(s): %s; lost side(s): %s",
                    ",".join(completed_sides) if completed_sides else "none",
                    ",".join(sorted(abandoned_sides)),
                )
            else:
                logger.info(
                    "Post-replay return-to-zero reached for target side(s): %s",
                    ",".join(completed_sides),
                )
            return hold_action
        if attempt == 1 or attempt % 5 == 0 or attempt == max_attempts:
            logger.info(
                "Post-replay zero in progress: attempt=%d/%d pending=%s left_joint_1=%.3f right_joint_1=%.3f",
                attempt,
                max_attempts,
                ",".join(sorted(pending_sides)),
                float(live_observation.get("left_joint_1.pos", float("nan"))),
                float(live_observation.get("right_joint_1.pos", float("nan"))),
            )

    logger.warning(
        "Post-replay return-to-zero did not fully converge within %d attempts. pending=%s.",
        max_attempts,
        ",".join(sorted(pending_sides)),
    )
    return hold_action


def _run_post_replay_hold(
    robot: BiSonglingFollower,
    *,
    args: argparse.Namespace,
    action_names: list[str],
    hold_action: dict[str, float] | None = None,
) -> None:
    hold_s = float(getattr(args, "post_replay_hold_s", 0.0))
    if hold_s == 0.0:
        logger.info("Post-replay hold is disabled by CLI; skipping.")
        return

    hold_sides = _connected_bus_sides(robot)
    if not hold_sides:
        logger.warning("No arm reached stable enabled state for post-replay hold; skipping.")
        return

    if hold_s < 0.0:
        logger.warning(
            "Post-replay hold is active indefinitely to prevent joint droop. Press Ctrl+C when you are ready to exit."
        )
    else:
        logger.info("Post-replay hold is active for %.1f s to prevent joint droop.", hold_s)
    start_t = time.perf_counter()
    keepalive_idx = 0
    frozen_hold_action = dict(hold_action) if hold_action is not None else _capture_hold_action(
        robot,
        action_names=action_names,
        sides=set(hold_sides),
    )
    while True:
        keepalive_idx += 1
        hold_sides &= _connected_bus_sides(robot)
        if not hold_sides:
            logger.warning("All CAN buses disconnected during post-replay hold; stop hold loop.")
            return
        try:
            _ensure_runtime_motion_ready(
                robot,
                sides=set(hold_sides),
                phase="post-replay hold",
            )
        except RuntimeError as exc:
            logger.warning("Post-replay hold is still recovering motion readiness: %s", exc)
            precise_sleep(max(POST_REPLAY_HOLD_INTERVAL_S, MODE_RECOVERY_SETTLE_S))
            continue

        for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
            if side not in hold_sides:
                continue
            try:
                arm.bus.ensure_can_command_mode(force=(keepalive_idx % 20 == 1))
            except Exception:
                pass

        _poll_robot_buses(robot)
        for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
            if side not in hold_sides:
                continue
            status = arm.bus.get_driver_enable_status()
            known = [state for state in status.values() if state is not None]
            if known and not all(bool(state) for state in known):
                logger.warning("%s detected driver-disable during post-replay hold; trying to re-enable.", _side_label(side))
                arm.bus.wait_for_driver_enable_status(enabled=True)
                arm.bus.ensure_can_command_mode(force=True)

        sent_action = _send_action_even_if_camera_dropped(robot, frozen_hold_action)
        for key, value in sent_action.items():
            if key in frozen_hold_action:
                frozen_hold_action[key] = float(value)

        if hold_s >= 0.0 and (time.perf_counter() - start_t) >= hold_s:
            logger.info("Post-replay hold window finished.")
            return
        precise_sleep(POST_REPLAY_HOLD_INTERVAL_S)


def _sanitize_optional_positive(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return None
    return float(value)


def _limit_action_step(
    target_action: dict[str, float],
    *,
    reference_action: dict[str, float],
    max_joint_step: float | None,
    max_gripper_step: float | None,
) -> dict[str, float]:
    if max_joint_step is None and max_gripper_step is None:
        return dict(target_action)

    limited: dict[str, float] = {}
    for name, target_value in target_action.items():
        max_step = max_gripper_step if "gripper" in name else max_joint_step
        if max_step is None:
            limited[name] = float(target_value)
            continue

        ref_value = float(reference_action.get(name, target_value))
        lower = ref_value - max_step
        upper = ref_value + max_step
        limited[name] = float(min(max(float(target_value), lower), upper))
    return limited


def _build_action_reference(
    *,
    action_names: list[str],
    live_observation: dict[str, float],
    fallback_action: dict[str, float] | None = None,
) -> dict[str, float]:
    reference: dict[str, float] = {}
    for name in action_names:
        if name in live_observation:
            reference[name] = float(live_observation[name])
        elif fallback_action is not None and name in fallback_action:
            reference[name] = float(fallback_action[name])
        else:
            reference[name] = 0.0
    return reference


def _capture_hold_action(
    robot: BiSonglingFollower,
    *,
    action_names: list[str],
    sides: set[str] | None = None,
    fallback_action: dict[str, float] | None = None,
) -> dict[str, float]:
    selected_sides = None if sides is None else set(sides)
    best_effort = _get_best_effort_joint_observation(robot)
    hold_action: dict[str, float] = {}
    for name in action_names:
        side = "left" if name.startswith("left_") else "right" if name.startswith("right_") else None
        if selected_sides is not None and side not in selected_sides:
            if fallback_action is not None and name in fallback_action:
                hold_action[name] = float(fallback_action[name])
            else:
                hold_action[name] = float(best_effort.get(name, 0.0))
            continue
        if name in best_effort:
            hold_action[name] = float(best_effort[name])
        elif fallback_action is not None and name in fallback_action:
            hold_action[name] = float(fallback_action[name])
        else:
            hold_action[name] = 0.0
    return hold_action


def _send_repeated_hold_action(
    robot: BiSonglingFollower,
    hold_action: dict[str, float] | None,
    *,
    repeat: int,
    interval_s: float,
) -> dict[str, float] | None:
    if hold_action is None:
        return None
    last_sent = dict(hold_action)
    for _ in range(max(int(repeat), 1)):
        sent_action = _send_action_even_if_camera_dropped(robot, last_sent)
        for key, value in sent_action.items():
            if key in last_sent:
                last_sent[key] = float(value)
        if interval_s > 0:
            precise_sleep(interval_s)
    return last_sent


def _build_zero_return_action(
    *,
    action_names: list[str],
    live_observation: dict[str, float],
    sides: set[str],
) -> dict[str, float]:
    zero_target = {name: float(live_observation.get(name, 0.0)) for name in action_names}
    for side in sides:
        for joint_name in DEFAULT_SONGLING_JOINT_NAMES:
            key = f"{side}_{joint_name}.pos"
            if key in zero_target:
                zero_target[key] = 0.0
    return _limit_action_step(
        zero_target,
        reference_action=_build_action_reference(
            action_names=action_names,
            live_observation=live_observation,
        ),
        max_joint_step=POST_REPLAY_ZERO_JOINT_STEP,
        max_gripper_step=POST_REPLAY_ZERO_GRIPPER_STEP,
    )


def _get_live_joint_observation(robot: BiSonglingFollower) -> dict[str, float]:
    _poll_robot_buses(robot)
    left_positions = robot.left_arm.bus.get_positions(poll=False)
    right_positions = robot.right_arm.bus.get_positions(poll=False)
    observation: dict[str, float] = {}
    for joint_name in DEFAULT_SONGLING_JOINT_NAMES:
        observation[f"left_{joint_name}.pos"] = float(left_positions[joint_name])
        observation[f"right_{joint_name}.pos"] = float(right_positions[joint_name])
    return observation


def _get_best_effort_joint_observation(robot: BiSonglingFollower) -> dict[str, float]:
    _poll_robot_buses(robot)
    observation: dict[str, float] = {}
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        live_positions = arm.bus.get_positions(poll=False)
        commanded_positions = arm.bus.get_commanded_positions(poll=False)
        for joint_name in DEFAULT_SONGLING_JOINT_NAMES:
            key = f"{side}_{joint_name}.pos"
            if bool(arm.bus.joint_position_seen.get(joint_name, False)):
                observation[key] = float(live_positions[joint_name])
            elif bool(arm.bus.commanded_position_seen.get(joint_name, False)):
                observation[key] = float(commanded_positions[joint_name])
            else:
                observation[key] = 0.0
    return observation


def _send_transition_hold_current_pose(
    robot: BiSonglingFollower,
    *,
    action_names: list[str],
    sides: set[str] | None = None,
) -> dict[str, float] | None:
    hold_sides = _connected_bus_sides(robot) if sides is None else (_connected_bus_sides(robot) & set(sides))
    if not hold_sides:
        return None

    hold_action = _capture_hold_action(
        robot,
        action_names=action_names,
        sides=set(hold_sides),
    )
    return _send_repeated_hold_action(
        robot,
        hold_action,
        repeat=POST_REPLAY_TRANSITION_HOLD_REPEAT,
        interval_s=POST_REPLAY_TRANSITION_HOLD_INTERVAL_S,
    )


def _get_live_camera_observation(robot: BiSonglingFollower) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for camera_key, camera in robot.left_arm.cameras.items():
        observation[f"left_{camera_key}"] = camera.read_latest()
    for camera_key, camera in robot.right_arm.cameras.items():
        observation[f"right_{camera_key}"] = camera.read_latest()
    return observation


def _run_live_integrated_replay(
    *,
    raw_cfg: dict[str, Any],
    args: argparse.Namespace,
    replay_dataset: LeRobotDataset,
    replay_indices: list[int],
    playback_fps: int,
    action_names: list[str],
    state_names: list[str],
    camera_keys: list[str],
    display_data: bool,
    compress_images: bool,
    play_sounds: bool,
    live_include_recorded_media: bool,
    live_read_cameras: bool,
    live_max_joint_step: float | None,
    live_max_gripper_step: float | None,
    start_frame: int,
) -> None:
    robot_cfg = _make_live_replay_robot_config(raw_cfg, args, playback_fps)
    _log_parameter_write_safety_lock(robot_cfg)
    robot = BiSonglingFollower(robot_cfg)
    deferred_left_cameras, deferred_right_cameras = _defer_live_camera_connect(robot)
    last_log = 0.0
    camera_read_enabled = bool(live_read_cameras)

    if play_sounds:
        log_say("Replaying episode on live robot", play_sounds, blocking=True)

    robot.connect(calibrate=False)
    try:
        startup_stabilized_sides = set(_prepare_live_robot(robot, strict=False))
        if startup_stabilized_sides:
            _send_transition_hold_current_pose(
                robot,
                action_names=action_names,
                sides=startup_stabilized_sides,
            )
        leader_activity = _detect_external_leader_activity(
            robot,
            observe_s=args.leader_activity_check_s,
        )
        if leader_activity["left"] or leader_activity["right"]:
            warning_message = (
                "检测到主臂(leader/master)仍在共享 CAN 链路中发送控制流量。\n\n"
                f"left_active={leader_activity['left']}, right_active={leader_activity['right']}\n\n"
                "当前安全策略要求: 只能连接从臂(slave/follower)再执行重播。\n"
                "请断开主臂连接后重新执行。"
            )
            _show_big_safety_popup("安全拦截: 检测到主臂连接", warning_message)
            if not args.allow_active_leader:
                raise RuntimeError(
                    warning_message
                    + "\n如果你确认要强制继续(不推荐)，可显式添加 --allow-active-leader=true。"
                )

        _run_pre_replay_enable_and_zero(
            robot,
            args=args,
            action_names=action_names,
            prepared_sides=(
                set(startup_stabilized_sides)
                if len(startup_stabilized_sides) == 2
                else None
            ),
        )
        last_sent_action = _send_transition_hold_current_pose(robot, action_names=action_names)
        if live_read_cameras:
            camera_read_enabled = _attach_and_connect_live_cameras(
                robot,
                left_cameras=deferred_left_cameras,
                right_cameras=deferred_right_cameras,
            )
            post_camera_hold_action = _send_transition_hold_current_pose(robot, action_names=action_names)
            if post_camera_hold_action is not None:
                last_sent_action = post_camera_hold_action
        if last_sent_action is None:
            last_sent_action = _get_live_joint_observation(robot)

        logger.info(
            "Starting live integrated replay. frames=%d fps=%d left_can=%s right_can=%s",
            len(replay_indices),
            playback_fps,
            robot_cfg.left_arm_config.channel,
            robot_cfg.right_arm_config.channel,
        )
        if display_data and not live_read_cameras:
            logger.info(
                "Live camera visualization is disabled (--no-live-read-cameras). "
                "Set --live-read-cameras=true to display desktop camera streams."
            )
        if display_data and not live_include_recorded_media:
            logger.info(
                "Recorded episode camera replay is disabled (--no-live-include-recorded-media). "
                "Set --live-include-recorded-media=true to display recorded video streams."
            )

        for local_idx, replay_idx in enumerate(replay_indices):
            loop_start = time.perf_counter()
            global_frame = start_frame + local_idx
            _ensure_runtime_motion_ready(
                robot,
                sides={"left", "right"},
                phase=f"live replay frame {global_frame}",
            )
            live_reference = _get_live_joint_observation(robot)
            replay_item = _get_replay_item(replay_dataset, replay_idx, include_media=live_include_recorded_media)
            recorded_action = _extract_action_dict(replay_item, action_names)
            action_to_send = _limit_action_step(
                recorded_action,
                reference_action=_build_action_reference(
                    action_names=action_names,
                    live_observation=live_reference,
                    fallback_action=last_sent_action,
                ),
                max_joint_step=live_max_joint_step,
                max_gripper_step=live_max_gripper_step,
            )
            recorded_observation = (
                _extract_replay_observation(
                    replay_item,
                    state_names=state_names,
                    camera_keys=camera_keys if live_include_recorded_media else [],
                )
                if display_data
                else None
            )

            sent_action = _send_action_even_if_camera_dropped(robot, action_to_send)
            last_sent_action = dict(sent_action)
            live_observation = _get_live_joint_observation(robot)
            if camera_read_enabled and not robot.cameras:
                camera_read_enabled = False
            if display_data and camera_read_enabled:
                try:
                    live_observation.update(_get_live_camera_observation(robot))
                except Exception as exc:
                    _disable_live_camera_streams(robot, reason=f"live camera read error: {exc}")
                    camera_read_enabled = False
                    logger.warning(
                        "Disable live camera reads for the rest of replay due to camera read error: %s",
                        exc,
                    )

            if display_data and _safe_rr_set_time("frame", sequence=global_frame):
                _log_bundle(
                    "replay",
                    observation=recorded_observation,
                    action=recorded_action,
                    compress_images=compress_images,
                    )
                _log_bundle(
                    "live_robot",
                    observation=live_observation,
                    action=sent_action,
                    compress_images=compress_images,
                )
                for name in action_names:
                    live_value = live_observation.get(name)
                    if live_value is None:
                        continue
                    _safe_rr_log(
                        f"tracking/error_vs_recorded/{name}",
                        rr.Scalars(float(live_value - recorded_action[name])),
                    )
                _safe_rr_log("status/left_obs_hz", rr.Scalars(float(robot.left_arm.bus.joint_feedback_hz)))
                _safe_rr_log("status/right_obs_hz", rr.Scalars(float(robot.right_arm.bus.joint_feedback_hz)))
                _safe_rr_log("status/left_cmd_hz", rr.Scalars(float(robot.left_arm.bus.command_feedback_hz)))
                _safe_rr_log("status/right_cmd_hz", rr.Scalars(float(robot.right_arm.bus.command_feedback_hz)))

            now = time.time()
            if now - last_log >= 1.0:
                logger.info(
                    "frame=%d/%d sent_left_joint_1=%.3f sent_right_joint_1=%.3f live_left_joint_1=%.3f live_right_joint_1=%.3f",
                    global_frame,
                    start_frame + len(replay_indices) - 1,
                    sent_action.get("left_joint_1.pos", float("nan")),
                    sent_action.get("right_joint_1.pos", float("nan")),
                    float(live_observation.get("left_joint_1.pos", float("nan"))),
                    float(live_observation.get("right_joint_1.pos", float("nan"))),
                )
                last_log = now

            precise_sleep(max(1.0 / playback_fps - (time.perf_counter() - loop_start), 0.0))

        zero_hold_action: dict[str, float] | None = None
        try:
            zero_hold_action = _run_post_replay_return_to_zero(robot, args=args, action_names=action_names)
        except Exception as exc:
            logger.exception(
                "Post-replay return-to-zero failed unexpectedly; switch to hold-current-pose to avoid joint droop. detail=%s",
                exc,
            )
            zero_hold_action = _send_transition_hold_current_pose(robot, action_names=action_names)
        _run_post_replay_hold(robot, args=args, action_names=action_names, hold_action=zero_hold_action)
    finally:
        _finalize_live_robot(robot, args=args, action_names=action_names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one Songling ALOHA dataset episode.")
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to Songling unified YAML config.",
    )
    parser.add_argument("--dataset.repo_id", "--dataset-repo-id", dest="dataset_repo_id", default=None)
    parser.add_argument("--dataset.root", "--dataset-root", dest="dataset_root", default=None)
    parser.add_argument("--dataset.episode", "--dataset-episode", dest="dataset_episode", type=int, default=0)
    parser.add_argument("--fps", dest="fps", type=int, default=None, help="Playback FPS. Defaults to dataset fps.")
    parser.add_argument("--start-frame", dest="start_frame", type=int, default=0)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved replay settings and exit.")
    parser.add_argument(
        "--display-data",
        "--display_data",
        dest="display_data",
        nargs="?",
        const="true",
        default=None,
        help="Enable Rerun visualization. Accepts true/false.",
    )
    parser.add_argument(
        "--no-display-data",
        "--no_display_data",
        dest="display_data",
        action="store_const",
        const="false",
        help="Disable Rerun visualization.",
    )
    parser.add_argument("--display-ip", "--display_ip", dest="display_ip", default=None)
    parser.add_argument("--display-port", "--display_port", dest="display_port", type=int, default=None)
    parser.add_argument(
        "--display-compressed-images",
        "--display_compressed_images",
        dest="display_compressed_images",
        nargs="?",
        const="true",
        default=None,
        help="Compress images before sending to Rerun. Accepts true/false.",
    )
    parser.add_argument(
        "--no-display-compressed-images",
        "--no_display_compressed_images",
        dest="display_compressed_images",
        action="store_const",
        const="false",
        help="Disable image compression before Rerun logging.",
    )
    parser.add_argument("--session-name", dest="session_name", default=DEFAULT_SESSION_NAME)
    parser.add_argument(
        "--play-sounds",
        "--play_sounds",
        dest="play_sounds",
        nargs="?",
        const="true",
        default=None,
        help="Enable voice notifications. Accepts true/false.",
    )
    parser.add_argument(
        "--no-play-sounds",
        "--no_play_sounds",
        dest="play_sounds",
        action="store_const",
        const="false",
        help="Disable voice notifications.",
    )
    parser.add_argument(
        "--connect-live",
        action="store_true",
        help="Send recorded actions to the Songling follower pair and display live cameras/state.",
    )
    parser.add_argument(
        "--leader-activity-check-s",
        type=float,
        default=0.5,
        help="Observe the bus for this duration before live replay to detect pre-existing leader control traffic.",
    )
    parser.add_argument(
        "--allow-active-leader",
        type=_parse_cli_bool,
        default=False,
        help="Override the live replay guard if leader/master control traffic is already active.",
    )
    parser.add_argument(
        "--pre-replay-enable-and-zero",
        "--pre_replay_enable_and_zero",
        dest="pre_replay_enable_and_zero",
        nargs="?",
        const="true",
        default="true",
        help="Before live replay, run enable + zero-pose sequence (mirrors manual upper-computer flow). Accepts true/false.",
    )
    parser.add_argument(
        "--no-pre-replay-enable-and-zero",
        "--no_pre_replay_enable_and_zero",
        dest="pre_replay_enable_and_zero",
        action="store_const",
        const="false",
        help="Skip the pre-replay enable + zero-pose sequence.",
    )
    parser.add_argument(
        "--post-replay-return-to-zero",
        "--post_replay_return_to_zero",
        dest="post_replay_return_to_zero",
        nargs="?",
        const="true",
        default="true",
        help="After live replay completes, return both arms to zero pose. Accepts true/false.",
    )
    parser.add_argument(
        "--no-post-replay-return-to-zero",
        "--no_post_replay_return_to_zero",
        dest="post_replay_return_to_zero",
        action="store_const",
        const="false",
        help="Skip post-replay return-to-zero routine.",
    )
    parser.add_argument(
        "--post-replay-hold-s",
        "--post_replay_hold_s",
        dest="post_replay_hold_s",
        type=float,
        default=-1.0,
        help=(
            "After replay/return-to-zero, keep sending hold commands to avoid joint droop. "
            "Set <0 to hold until Ctrl+C, 0 to disable, >0 for fixed seconds."
        ),
    )
    parser.add_argument(
        "--disconnect-live-robot-on-exit",
        action="store_true",
        help=(
            "Restore the old cleanup behavior and explicitly disconnect the live robot on script exit. "
            "By default replay now keeps the robot enabled on exit to reduce droop risk."
        ),
    )
    parser.add_argument(
        "--zero-max-attempts",
        type=int,
        default=60,
        help="Max zero-pose send attempts during pre-replay routine.",
    )
    parser.add_argument(
        "--zero-settle-s",
        type=float,
        default=ZERO_TRAJECTORY_INTERVAL_S,
        help="Sleep seconds between zero-pose attempts during pre-replay routine.",
    )
    parser.add_argument(
        "--zero-joint-tolerance",
        type=float,
        default=0.5,
        help="Joint tolerance (deg) for zero-pose convergence check.",
    )
    parser.add_argument(
        "--zero-gripper-tolerance",
        type=float,
        default=0.5,
        help="Gripper tolerance (mm) for zero-pose convergence check.",
    )
    parser.add_argument(
        "--live-include-recorded-media",
        "--live_include_recorded_media",
        dest="live_include_recorded_media",
        nargs="?",
        const="true",
        default="false",
        help="In live replay, also decode/log recorded episode images. Accepts true/false.",
    )
    parser.add_argument(
        "--no-live-include-recorded-media",
        "--no_live_include_recorded_media",
        dest="live_include_recorded_media",
        action="store_const",
        const="false",
        help="In live replay, skip recorded episode image decoding (recommended for stable timing).",
    )
    parser.add_argument(
        "--live-read-cameras",
        "--live_read_cameras",
        dest="live_read_cameras",
        nargs="?",
        const="true",
        default="false",
        help="In live replay, capture/log live camera frames. Accepts true/false.",
    )
    parser.add_argument(
        "--no-live-read-cameras",
        "--no_live_read_cameras",
        dest="live_read_cameras",
        action="store_const",
        const="false",
        help="In live replay, skip live camera capture to keep command loop lighter.",
    )
    parser.add_argument(
        "--live-max-joint-step",
        type=float,
        default=1.5,
        help="Per-command joint step limit in deg for live replay; <=0 disables this limit.",
    )
    parser.add_argument(
        "--live-max-gripper-step",
        type=float,
        default=3.0,
        help="Per-command gripper step limit in mm for live replay; <=0 disables this limit.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Override max_relative_target for both arms in live replay. Use negative value to disable.",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=None,
        help="Override move speed percent for both arms in live replay.",
    )
    parser.add_argument(
        "--command-repeat",
        type=int,
        default=None,
        help="Override command repeat count for both arms in live replay.",
    )
    parser.add_argument(
        "--command-interval-s",
        type=float,
        default=None,
        help="Override command interval seconds for both arms in live replay.",
    )
    parser.add_argument("--left-interface", "--robot.left_arm_config.channel", dest="left_interface", default=None)
    parser.add_argument("--right-interface", "--robot.right_arm_config.channel", dest="right_interface", default=None)
    parser.add_argument("--left-bitrate", "--robot.left_arm_config.bitrate", dest="left_bitrate", type=int, default=None)
    parser.add_argument("--right-bitrate", "--robot.right_arm_config.bitrate", dest="right_bitrate", type=int, default=None)
    parser.add_argument(
        "--left-data-bitrate",
        "--robot.left_arm_config.can_data_bitrate",
        dest="left_data_bitrate",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--right-data-bitrate",
        "--robot.right_arm_config.can_data_bitrate",
        dest="right_data_bitrate",
        type=int,
        default=None,
    )
    parser.add_argument("--left-use-fd", "--robot.left_arm_config.use_can_fd", dest="left_use_fd", type=_parse_cli_bool, default=None)
    parser.add_argument("--right-use-fd", "--robot.right_arm_config.use_can_fd", dest="right_use_fd", type=_parse_cli_bool, default=None)
    parser.add_argument("--left-high", "--robot.left_arm_config.cameras.high.index_or_path", dest="left_high", default=None)
    parser.add_argument("--left-elbow", "--robot.left_arm_config.cameras.elbow.index_or_path", dest="left_elbow", default=None)
    parser.add_argument("--right-elbow", "--robot.right_arm_config.cameras.elbow.index_or_path", dest="right_elbow", default=None)
    return parser.parse_args()


def main() -> None:
    _patch_multiprocess_resource_tracker()
    register_third_party_plugins()
    args = _parse_args()
    init_logging()

    raw_cfg = _load_songling_yaml(args.config_path)
    dataset_opts = _resolve_dataset_options(raw_cfg, args)

    raw_dataset = raw_cfg.get("dataset") if isinstance(raw_cfg.get("dataset"), dict) else {}
    display_data = True if args.display_data is None else _parse_cli_bool(args.display_data)
    if args.display_data is None and "display_data" in raw_cfg:
        display_data = bool(raw_cfg.get("display_data"))
    live_include_recorded_media = _parse_cli_bool(args.live_include_recorded_media)
    live_read_cameras = _parse_cli_bool(args.live_read_cameras)
    pre_replay_enable_and_zero = _parse_cli_bool(args.pre_replay_enable_and_zero)
    post_replay_return_to_zero = _parse_cli_bool(args.post_replay_return_to_zero)
    live_max_joint_step = _sanitize_optional_positive(args.live_max_joint_step)
    live_max_gripper_step = _sanitize_optional_positive(args.live_max_gripper_step)
    display_ip = args.display_ip if args.display_ip is not None else raw_cfg.get("display_ip")
    display_port = args.display_port if args.display_port is not None else raw_cfg.get("display_port")
    compress_images = False if args.display_compressed_images is None else _parse_cli_bool(args.display_compressed_images)
    if args.display_compressed_images is None and "display_compressed_images" in raw_cfg:
        compress_images = bool(raw_cfg.get("display_compressed_images"))
    play_sounds = False if args.play_sounds is None else _parse_cli_bool(args.play_sounds)
    if args.play_sounds is None and "play_sounds" in raw_cfg:
        play_sounds = bool(raw_cfg.get("play_sounds"))

    if (display_ip is None) != (display_port is None):
        raise ValueError("Please set both display_ip and display_port together, or omit both.")

    if args.dry_run:
        playback_fps = args.fps if args.fps is not None else int(raw_dataset.get("fps", raw_cfg.get("fps", 30)))
        print("=" * 50)
        print("Songling ALOHA Replay")
        print("=" * 50)
        print(f"Config: {args.config_path}")
        print(f"Dataset repo_id: {dataset_opts.repo_id}")
        print(f"Dataset root: {dataset_opts.root}")
        print(f"Episode: {dataset_opts.episode}")
        print(f"Playback FPS: {playback_fps}")
        print(f"Mode: {'live replay' if args.connect_live else 'offline replay'}")
        if args.connect_live:
            print(f"Leader activity guard: check {args.leader_activity_check_s:.2f}s, allow_active={args.allow_active_leader}")
            print(
                "Pre-replay routine: "
                f"enable_and_zero={'on' if pre_replay_enable_and_zero else 'off'}, "
                f"zero_max_attempts={max(int(args.zero_max_attempts), 1)}, "
                f"zero_settle_s={max(float(args.zero_settle_s), 0.0):.3f}, "
                f"zero_joint_tol={max(float(args.zero_joint_tolerance), 0.0):.3f}, "
                f"zero_gripper_tol={max(float(args.zero_gripper_tolerance), 0.0):.3f}"
            )
            print(f"Post-replay routine: return_to_zero={'on' if post_replay_return_to_zero else 'off'}")
            print(
                "Post-replay hold: "
                + (
                    "until Ctrl+C"
                    if float(args.post_replay_hold_s) < 0.0
                    else f"{max(float(args.post_replay_hold_s), 0.0):.1f}s"
                )
            )
            print(
                "Exit cleanup: "
                + ("explicit disconnect" if args.disconnect_live_robot_on_exit else "keep robot enabled")
            )
            print(
                "Live replay extras: "
                f"recorded_media={'on' if live_include_recorded_media else 'off'}, "
                f"live_cameras={'on' if live_read_cameras else 'off'}, "
                f"joint_step_limit={live_max_joint_step if live_max_joint_step is not None else 'off'}, "
                f"gripper_step_limit={live_max_gripper_step if live_max_gripper_step is not None else 'off'}"
            )
        print(
            "Display: "
            + (
                f"on ({display_ip}:{display_port})"
                if (display_data and display_ip and display_port)
                else ("on (local viewer)" if display_data else "off")
            )
        )
        return

    download_videos = display_data and (not args.connect_live or live_include_recorded_media)
    dataset = LeRobotDataset(
        dataset_opts.repo_id,
        root=dataset_opts.root,
        episodes=[dataset_opts.episode],
        download_videos=download_videos,
    )
    if len(dataset) == 0:
        raise ValueError(f"No frames found for episode {dataset_opts.episode} in dataset {dataset_opts.repo_id!r}.")
    if ACTION_KEY not in dataset.features:
        raise ValueError(f"Dataset is missing required '{ACTION_KEY}' feature.")
    if OBS_STATE_KEY not in dataset.features:
        raise ValueError(f"Dataset is missing required '{OBS_STATE_KEY}' feature.")

    action_names = list(dataset.features[ACTION_KEY]["names"])
    state_names = list(dataset.features[OBS_STATE_KEY]["names"])
    camera_keys = list(dataset.meta.camera_keys)
    replay_indices, stop_frame = _slice_episode_indices(
        total=len(dataset),
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )

    playback_fps = args.fps if args.fps is not None else int(dataset.fps)
    if playback_fps <= 0:
        raise ValueError(f"Invalid playback fps: {playback_fps}.")

    logger.info(
        "Replay config: repo_id=%s root=%s episode=%d frames=%d start_frame=%d stop_frame=%d fps=%d connect_live=%s",
        dataset_opts.repo_id,
        str(dataset_opts.root) if dataset_opts.root is not None else "default",
        dataset_opts.episode,
        len(replay_indices),
        args.start_frame,
        stop_frame,
        playback_fps,
        args.connect_live,
    )
    args.pre_replay_enable_and_zero = pre_replay_enable_and_zero
    args.post_replay_return_to_zero = post_replay_return_to_zero

    _maybe_init_rerun(
        display_data=display_data,
        session_name=args.session_name,
        display_ip=display_ip,
        display_port=display_port,
    )
    try:
        if args.connect_live:
            _run_live_integrated_replay(
                raw_cfg=raw_cfg,
                args=args,
                replay_dataset=dataset,
                replay_indices=replay_indices,
                playback_fps=playback_fps,
                action_names=action_names,
                state_names=state_names,
                camera_keys=camera_keys,
                display_data=display_data,
                compress_images=compress_images,
                play_sounds=play_sounds,
                live_include_recorded_media=live_include_recorded_media,
                live_read_cameras=live_read_cameras,
                live_max_joint_step=live_max_joint_step,
                live_max_gripper_step=live_max_gripper_step,
                start_frame=args.start_frame,
            )
        else:
            _run_offline_replay(
                replay_dataset=dataset,
                replay_indices=replay_indices,
                playback_fps=playback_fps,
                action_names=action_names,
                state_names=state_names,
                camera_keys=camera_keys,
                display_data=display_data,
                compress_images=compress_images,
                play_sounds=play_sounds,
                start_frame=args.start_frame,
            )
    finally:
        _shutdown_rerun(display_data)


if __name__ == "__main__":
    main()
