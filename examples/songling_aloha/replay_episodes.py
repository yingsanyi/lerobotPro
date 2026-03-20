#!/usr/bin/env python

"""Replay a Songling ALOHA dataset episode.

This script is modeled after ACT's `replay_episodes.py`, but adapted to the
LeRobot Songling example workflow:

- offline replay: iterate one episode at dataset FPS and visualize it in Rerun
- raw CAN shadow replay: compare recorded actions against live Songling CAN +
  camera observations, without sending commands
- live robot replay: send recorded actions through the generic robot runtime
  when using a non-integrated runtime configuration

Examples:
    python examples/songling_aloha/replay_episodes.py \
      --dataset.root outputs/songling_aloha \
      --dataset.repo_id your_hf_username/songling_aloha_demo \
      --dataset.episode 0 \
      --display_data=true

    python examples/songling_aloha/replay_episodes.py \
      --config-path examples/songling_aloha/teleop.yaml \
      --dataset.root outputs/songling_aloha \
      --dataset.repo_id your_hf_username/songling_aloha_demo \
      --dataset.episode 0 \
      --raw-can-mode \
      --display_data=true

    python examples/songling_aloha/replay_episodes.py \
      --config-path path/to/non_integrated_openarm.yaml \
      --dataset.root outputs/songling_aloha \
      --dataset.repo_id your_hf_username/songling_aloha_demo \
      --dataset.episode 0 \
      --connect-live
"""

from __future__ import annotations

import argparse
import logging
import numbers
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots import bi_openarm_follower, openarm_follower  # noqa: F401
from lerobot.robots import bi_songling_follower, songling_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators import bi_openarm_leader, openarm_leader  # noqa: F401
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

try:
    import rerun as rr
except ModuleNotFoundError:
    rr = None

DEFAULT_CONFIG_PATH = Path("examples/songling_aloha/teleop.yaml")
DEFAULT_SESSION_NAME = "songling_aloha_replay"

logger = logging.getLogger(__name__)


@dataclass
class SonglingUnifiedConfig:
    """Superset config used to parse the unified Songling YAML."""

    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    policy: Any | None = None
    fps: int = 30
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    dataset: Any | None = None
    play_sounds: bool | None = None
    voice_lang: str | None = None
    voice_rate: int | None = None
    voice_engine: str | None = None
    voice_piper_model: str | None = None
    voice_piper_binary: str | None = None
    voice_piper_speaker: int | None = None
    resume: bool | None = None


def _patch_multiprocess_resource_tracker() -> None:
    """Work around multiprocess<->Python3.12 shutdown noise."""

    try:
        from multiprocess import resource_tracker as _rt  # type: ignore
    except Exception:
        return

    if getattr(_rt, "_songling_patch_applied", False):
        return

    def _safe_stop_locked(
        self,
        close=os.close,
        waitpid=os.waitpid,
    ):
        recursion = 0
        try:
            recursion_attr = getattr(self._lock, "_recursion_count", None)
            if callable(recursion_attr):
                recursion = int(recursion_attr())
            elif recursion_attr is not None:
                recursion = int(recursion_attr)
        except Exception:
            recursion = 0

        if recursion > 1:
            return self._reentrant_call_error()
        if self._fd is None or self._pid is None:
            return

        close(self._fd)
        self._fd = None
        waitpid(self._pid, 0)
        self._pid = None

    try:
        _rt.ResourceTracker._stop_locked = _safe_stop_locked
        _rt._songling_patch_applied = True
    except Exception:
        pass


def _parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false.")


def _parse_songling_config(config_path: Path, unknown_args: list[str]) -> SonglingUnifiedConfig:
    import draccus

    return draccus.parse(SonglingUnifiedConfig, config_path=config_path, args=unknown_args)


def _load_songling_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import draccus

    with config_path.open("r", encoding="utf-8") as f:
        cfg = draccus.load(dict[str, Any], f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} is not a valid mapping.")
    return cfg


def _is_bimanual_songling_integrated_chain(cfg: SonglingUnifiedConfig | None) -> bool:
    if cfg is None or cfg.robot is None or cfg.teleop is None:
        return False

    robot_type = getattr(cfg.robot, "type", None)
    teleop_type = getattr(cfg.teleop, "type", None)
    if robot_type != "bi_openarm_follower" or teleop_type != "bi_openarm_leader":
        return False

    same_left_port = getattr(cfg.teleop.left_arm_config, "port", None) == getattr(
        cfg.robot.left_arm_config, "port", None
    )
    same_right_port = getattr(cfg.teleop.right_arm_config, "port", None) == getattr(
        cfg.robot.right_arm_config, "port", None
    )
    return same_left_port and same_right_port


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (float, numbers.Real, np.integer, np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def _log_value(path: str, value: Any, *, compress_images: bool) -> None:
    if rr is None or value is None:
        return

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            value = value.item()
        else:
            value = value.numpy()

    if _is_scalar(value):
        rr.log(path, rr.Scalars(float(value)))
        return

    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 1:
            for i, item in enumerate(arr):
                rr.log(f"{path}/{i}", rr.Scalars(float(item)))
            return

        image = rr.Image(arr).compress() if compress_images else rr.Image(arr)
        rr.log(path, image)
        return

    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            if _is_scalar(item):
                rr.log(f"{path}/{i}", rr.Scalars(float(item)))


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


def _named_vector(values: np.ndarray, names: list[str]) -> dict[str, float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {array.shape}.")
    if len(array) != len(names):
        raise ValueError(f"Vector length {len(array)} does not match feature names length {len(names)}.")
    return {name: float(array[i]) for i, name in enumerate(names)}


def _extract_action_dict(frame_row: dict[str, Any], action_names: list[str]) -> dict[str, float]:
    return _named_vector(np.asarray(frame_row[ACTION], dtype=np.float32), action_names)


def _extract_replay_observation(
    frame_row: dict[str, Any],
    *,
    state_names: list[str],
    camera_keys: list[str],
) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    if "observation.state" in frame_row:
        observation.update(_named_vector(np.asarray(frame_row["observation.state"], dtype=np.float32), state_names))
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


def _resolve_dataset_value(raw_dataset: dict[str, Any], cli_value: Any, key: str) -> Any:
    if cli_value is not None:
        return cli_value
    return raw_dataset.get(key)


def _resolve_dataset_options(raw_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[str, Path | None, int]:
    raw_dataset = raw_cfg.get("dataset") if isinstance(raw_cfg.get("dataset"), dict) else {}

    root = _resolve_dataset_value(raw_dataset, args.dataset_root, "root")
    repo_id = _resolve_dataset_value(raw_dataset, args.dataset_repo_id, "repo_id")
    episode = _resolve_dataset_value(raw_dataset, args.dataset_episode, "episode")

    root_path = None if root is None else Path(root)
    if args.dataset_repo_id is None and args.dataset_root is not None and root_path is not None:
        repo_id = f"local/{root_path.name}"
        logger.info("Using local repo id from explicit --dataset.root: %s", repo_id)
    elif repo_id is None and root_path is not None:
        repo_id = f"local/{root_path.name}"
        logger.info("No --dataset.repo_id provided, using derived local repo id: %s", repo_id)

    if repo_id is None:
        raise ValueError(
            "Missing dataset repo id. Set --dataset.repo_id=..., or provide --dataset.root so a local id can be derived, "
            "or add dataset.repo_id in the Songling YAML."
        )
    if episode is None:
        episode = 0

    return str(repo_id), root_path, int(episode)


def _maybe_init_rerun(
    *,
    display_data: bool,
    session_name: str,
    display_ip: str | None,
    display_port: int | None,
) -> None:
    if not display_data:
        return
    if rr is None:
        raise ModuleNotFoundError("rerun is required for --display_data=true. Please install rerun-sdk.")
    init_rerun(session_name=session_name, ip=display_ip, port=display_port)


def _shutdown_rerun(display_data: bool) -> None:
    if display_data and rr is not None:
        try:
            rr.rerun_shutdown()
        except Exception:  # nosec B110
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

        if display_data:
            rr.set_time("frame", sequence=global_frame)
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
            left_joint_1 = action.get("left_joint_1.pos")
            right_joint_1 = action.get("right_joint_1.pos")
            logger.info(
                "frame=%d/%d recorded_left_joint_1=%s recorded_right_joint_1=%s",
                global_frame,
                start_frame + len(replay_indices) - 1,
                f"{left_joint_1:.3f}" if left_joint_1 is not None else "n/a",
                f"{right_joint_1:.3f}" if right_joint_1 is not None else "n/a",
            )
            last_log = now

        precise_sleep(max(1.0 / playback_fps - (time.perf_counter() - loop_start), 0.0))


def _run_shadow_replay(
    *,
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
    start_frame: int,
) -> None:
    try:
        from shadow_env import SonglingShadowEnv, build_position_feature_names
    except ModuleNotFoundError:
        from examples.songling_aloha.shadow_env import SonglingShadowEnv, build_position_feature_names

    env = SonglingShadowEnv.from_cli_args(args=args, fallback_fps=playback_fps)
    live_names = build_position_feature_names(env.joint_names)
    last_log = 0.0

    if play_sounds:
        log_say("Replaying episode", play_sounds, blocking=True)

    logger.info(
        "Starting Songling raw-CAN shadow replay. frames=%d fps=%d left_can=%s right_can=%s",
        len(replay_indices),
        playback_fps,
        env.left_can.interface,
        env.right_can.interface,
    )

    try:
        env.connect()
        for local_idx, replay_idx in enumerate(replay_indices):
            loop_start = time.perf_counter()
            global_frame = start_frame + local_idx
            replay_item = _get_replay_item(replay_dataset, replay_idx, include_media=display_data)
            recorded_action = _extract_action_dict(replay_item, action_names)
            recorded_state = _named_vector(replay_item["observation.state"], state_names)
            shadow_step = env.poll()

            live_observation = {
                **_named_vector(np.concatenate((shadow_step.left_obs, shadow_step.right_obs), axis=0), live_names),
                "left_high": shadow_step.observation.get("left_high"),
                "left_elbow": shadow_step.observation.get("left_elbow"),
                "right_elbow": shadow_step.observation.get("right_elbow"),
            }
            live_observation = {k: v for k, v in live_observation.items() if v is not None}
            live_candidate = {
                **_named_vector(
                    np.concatenate((shadow_step.left_candidate_action, shadow_step.right_candidate_action), axis=0),
                    live_names,
                )
            }
            live_bus = {
                **_named_vector(
                    np.concatenate((shadow_step.left_bus_action, shadow_step.right_bus_action), axis=0),
                    live_names,
                )
            }

            if display_data:
                rr.set_time("frame", sequence=global_frame)
                replay_observation = _extract_replay_observation(
                    replay_item,
                    state_names=state_names,
                    camera_keys=camera_keys,
                )
                _log_bundle(
                    "replay",
                    observation=replay_observation,
                    action=recorded_action,
                    compress_images=compress_images,
                )
                _log_bundle(
                    "shadow/live",
                    observation=live_observation,
                    action=live_candidate,
                    compress_images=compress_images,
                )
                _log_bundle(
                    "shadow/bus",
                    action=live_bus,
                    compress_images=compress_images,
                )
                for name in sorted(set(action_names) & set(live_names)):
                    rr.log(
                        f"shadow/error_vs_recorded/{name}",
                        rr.Scalars(float(live_candidate[name] - recorded_action[name])),
                    )
                    rr.log(
                        f"shadow/error_obs_vs_recorded_state/{name}",
                        rr.Scalars(float(live_observation[name] - recorded_state[name])),
                    )
                rr.log("shadow/can.left_rx_hz", rr.Scalars(float(shadow_step.left_rx_hz)))
                rr.log("shadow/can.right_rx_hz", rr.Scalars(float(shadow_step.right_rx_hz)))
                for cam_key, health in shadow_step.camera_health.items():
                    rr.log(f"shadow/cameras/{cam_key}/stale_count", rr.Scalars(float(health["stale_count"])))
                    rr.log(
                        f"shadow/cameras/{cam_key}/identical_count",
                        rr.Scalars(float(health["identical_count"])),
                    )
                    rr.log(f"shadow/cameras/{cam_key}/used_cached", rr.Scalars(float(health["used_cached"])))
                    rr.log(f"shadow/cameras/{cam_key}/reconnected", rr.Scalars(float(health["reconnected"])))

            now = time.time()
            if now - last_log >= 1.0:
                logger.info(
                    "frame=%d/%d left_rx_hz=%.1f right_rx_hz=%.1f "
                    "recorded_left_joint_1=%.3f live_left_joint_1=%.3f candidate_left_joint_1=%.3f",
                    global_frame,
                    start_frame + len(replay_indices) - 1,
                    shadow_step.left_rx_hz,
                    shadow_step.right_rx_hz,
                    recorded_action.get("left_joint_1.pos", float("nan")),
                    live_observation.get("left_joint_1.pos", float("nan")),
                    live_candidate.get("left_joint_1.pos", float("nan")),
                )
                last_log = now

            precise_sleep(max(1.0 / playback_fps - (time.perf_counter() - loop_start), 0.0))
    finally:
        env.disconnect()


def _run_live_robot_replay(
    *,
    cfg: SonglingUnifiedConfig,
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
    if cfg.robot is None:
        raise ValueError("Live replay requires a valid robot config in the Songling YAML.")

    robot = make_robot_from_config(cfg.robot)
    last_log = 0.0

    if play_sounds:
        log_say("Replaying episode", play_sounds, blocking=True)

    robot.connect()
    try:
        logger.info("Starting live robot replay. frames=%d fps=%d robot=%s", len(replay_indices), playback_fps, robot)
        for local_idx, replay_idx in enumerate(replay_indices):
            loop_start = time.perf_counter()
            global_frame = start_frame + local_idx
            replay_item = _get_replay_item(replay_dataset, replay_idx, include_media=display_data)
            action = _extract_action_dict(replay_item, action_names)
            observation_before = robot.get_observation() if display_data else None
            sent_action = robot.send_action(action)

            if display_data:
                rr.set_time("frame", sequence=global_frame)
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
                _log_bundle(
                    "live_robot",
                    observation=observation_before,
                    action=sent_action,
                    compress_images=compress_images,
                )

            now = time.time()
            if now - last_log >= 1.0:
                logger.info(
                    "frame=%d/%d sent_left_joint_1=%s sent_right_joint_1=%s",
                    global_frame,
                    start_frame + len(replay_indices) - 1,
                    f"{sent_action.get('left_joint_1.pos', float('nan')):.3f}"
                    if "left_joint_1.pos" in sent_action
                    else "n/a",
                    f"{sent_action.get('right_joint_1.pos', float('nan')):.3f}"
                    if "right_joint_1.pos" in sent_action
                    else "n/a",
                )
                last_log = now

            precise_sleep(max(1.0 / playback_fps - (time.perf_counter() - loop_start), 0.0))
    finally:
        robot.disconnect()


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
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
    parser.add_argument(
        "--fps",
        dest="fps",
        type=int,
        default=None,
        help="Playback FPS. Defaults to dataset fps.",
    )
    parser.add_argument(
        "--start-frame",
        dest="start_frame",
        type=int,
        default=0,
        help="Start replay from this frame index within the episode.",
    )
    parser.add_argument(
        "--max-frames",
        dest="max_frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to replay.",
    )
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
    parser.add_argument(
        "--display-ip",
        "--display_ip",
        dest="display_ip",
        default=None,
        help="Optional Rerun server IP.",
    )
    parser.add_argument(
        "--display-port",
        "--display_port",
        dest="display_port",
        type=int,
        default=None,
        help="Optional Rerun server port.",
    )
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
    parser.add_argument(
        "--session-name",
        dest="session_name",
        default=DEFAULT_SESSION_NAME,
        help="Rerun session name.",
    )
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
        help="Send recorded actions to the robot using the generic runtime from the config.",
    )
    parser.add_argument(
        "--raw-can-mode",
        action="store_true",
        help="Use live Songling CAN + camera shadow replay without sending commands.",
    )
    parser.add_argument(
        "--joint-names",
        default="joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,gripper",
        help="Comma-separated joint names per side for raw CAN mode.",
    )
    parser.add_argument(
        "--observation-ids",
        default="0x251,0x252,0x253,0x254,0x255,0x256,0x2A8",
        help="Default CAN ids mapped to observation joints in raw CAN mode.",
    )
    parser.add_argument(
        "--action-ids",
        default="0x2A1,0x2A2,0x2A3,0x2A4,0x2A5,0x2A6,0x2A7",
        help="Default CAN ids mapped to action joints in raw CAN mode.",
    )
    parser.add_argument("--left-observation-ids", default=None, help="Override left-side observation ids.")
    parser.add_argument("--right-observation-ids", default=None, help="Override right-side observation ids.")
    parser.add_argument("--left-action-ids", default=None, help="Override left-side action ids.")
    parser.add_argument("--right-action-ids", default=None, help="Override right-side action ids.")
    parser.add_argument("--decode-byte-offset", type=int, default=0, help="CAN payload byte offset for raw decode.")
    parser.add_argument(
        "--decode-byte-length",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Number of bytes to decode into one signal value in raw CAN mode.",
    )
    parser.add_argument(
        "--decode-signed",
        type=_parse_cli_bool,
        default=True,
        help="Whether decoded integer is signed in raw CAN mode.",
    )
    parser.add_argument(
        "--decode-endian",
        choices=["little", "big"],
        default="little",
        help="Endianness for raw CAN integer decoding.",
    )
    parser.add_argument(
        "--decode-scale",
        type=float,
        default=1e-4,
        help="Scale applied after raw CAN integer decode.",
    )
    parser.add_argument("--decode-bias", type=float, default=0.0, help="Bias applied after raw CAN decode.")
    parser.add_argument("--max-can-poll-msgs", type=int, default=128, help="Max CAN frames consumed per loop per side.")
    parser.add_argument("--camera-max-age-ms", type=int, default=500, help="Max age for camera read_latest in raw mode.")
    parser.add_argument(
        "--camera-retry-timeout-ms",
        type=int,
        default=60,
        help="Retry timeout for camera async_read when read_latest is stale in raw mode.",
    )
    parser.add_argument(
        "--camera-reconnect-stale-count",
        type=int,
        default=12,
        help="Reconnect camera after this many consecutive stale/cached frames in raw mode.",
    )
    parser.add_argument(
        "--camera-freeze-identical-count",
        type=int,
        default=120,
        help="Reconnect camera after this many consecutive identical frames in raw mode.",
    )
    parser.add_argument(
        "--camera-reconnect-cooldown-s",
        type=float,
        default=2.0,
        help="Minimum seconds between reconnect attempts for the same camera in raw mode.",
    )
    parser.add_argument(
        "--camera-fail-on-freeze",
        type=_parse_cli_bool,
        default=False,
        help="Abort raw shadow replay when camera reconnect cannot recover from freeze.",
    )
    parser.add_argument("--left-interface", "--robot.left_arm_config.port", dest="left_interface", default=None)
    parser.add_argument("--right-interface", "--robot.right_arm_config.port", dest="right_interface", default=None)
    parser.add_argument(
        "--left-bitrate", "--robot.left_arm_config.can_bitrate", dest="left_bitrate", type=int, default=None
    )
    parser.add_argument(
        "--right-bitrate", "--robot.right_arm_config.can_bitrate", dest="right_bitrate", type=int, default=None
    )
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
    parser.add_argument(
        "--left-use-fd",
        "--robot.left_arm_config.use_can_fd",
        dest="left_use_fd",
        type=_parse_cli_bool,
        default=None,
    )
    parser.add_argument(
        "--right-use-fd",
        "--robot.right_arm_config.use_can_fd",
        dest="right_use_fd",
        type=_parse_cli_bool,
        default=None,
    )
    parser.add_argument(
        "--left-high",
        "--robot.left_arm_config.cameras.high.index_or_path",
        dest="left_high",
        default=None,
        help="Override camera device for left_high in raw CAN mode.",
    )
    parser.add_argument(
        "--left-elbow",
        "--robot.left_arm_config.cameras.elbow.index_or_path",
        dest="left_elbow",
        default=None,
        help="Override camera device for left_elbow in raw CAN mode.",
    )
    parser.add_argument(
        "--right-elbow",
        "--robot.right_arm_config.cameras.elbow.index_or_path",
        dest="right_elbow",
        default=None,
        help="Override camera device for right_elbow in raw CAN mode.",
    )
    return parser.parse_known_args()


def main() -> None:
    _patch_multiprocess_resource_tracker()
    register_third_party_plugins()
    args, unknown_args = _parse_args()
    init_logging()

    raw_cfg = _load_songling_yaml(args.config_path)
    cfg = _parse_songling_config(args.config_path, unknown_args)
    repo_id, root, episode = _resolve_dataset_options(raw_cfg, args)

    integrated_chain = _is_bimanual_songling_integrated_chain(cfg)
    raw_can_mode = bool(args.raw_can_mode)
    connect_live = bool(args.connect_live)
    if raw_can_mode and connect_live:
        raise ValueError("--raw-can-mode and --connect-live are mutually exclusive.")

    if integrated_chain and connect_live:
        raise NotImplementedError(
            "The current Songling teleop.yaml models an integrated leader+follower CAN chain. "
            "That layout is not compatible with the generic live OpenArm replay runtime yet. "
            "Use --raw-can-mode for shadow replay instead."
        )

    display_data = cfg.display_data if args.display_data is None else _parse_cli_bool(args.display_data)
    display_ip = cfg.display_ip if args.display_ip is None else args.display_ip
    display_port = cfg.display_port if args.display_port is None else args.display_port
    compress_images = (
        cfg.display_compressed_images
        if args.display_compressed_images is None
        else _parse_cli_bool(args.display_compressed_images)
    )
    play_sounds = (
        bool(cfg.play_sounds)
        if args.play_sounds is None
        else _parse_cli_bool(args.play_sounds)
    )

    download_videos = display_data
    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode], download_videos=download_videos)
    if len(dataset) == 0:
        raise ValueError(f"No frames found for episode {episode} in dataset {repo_id!r}.")

    if ACTION not in dataset.features:
        raise ValueError(f"Dataset is missing required '{ACTION}' feature.")
    if "observation.state" not in dataset.features:
        raise ValueError("Dataset is missing required 'observation.state' feature.")

    state_names = list(dataset.features["observation.state"]["names"])
    action_names = list(dataset.features[ACTION]["names"])
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
        "Replay config: repo_id=%s root=%s episode=%d frames=%d start_frame=%d stop_frame=%d fps=%d "
        "display_data=%s raw_can_mode=%s connect_live=%s integrated_chain=%s",
        repo_id,
        str(root) if root is not None else "default",
        episode,
        len(replay_indices),
        args.start_frame,
        stop_frame,
        playback_fps,
        display_data,
        raw_can_mode,
        connect_live,
        integrated_chain,
    )

    _maybe_init_rerun(
        display_data=display_data,
        session_name=args.session_name,
        display_ip=display_ip,
        display_port=display_port,
    )
    try:
        if raw_can_mode:
            _run_shadow_replay(
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
                start_frame=args.start_frame,
            )
        elif connect_live:
            _run_live_robot_replay(
                cfg=cfg,
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
