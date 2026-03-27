#!/usr/bin/env python

"""Live Songling ALOHA teaching monitor for shared-CAN master/slave pairs.

This script is the single maintained Songling teaching entrypoint:

- one CAN interface per side
- each side is one master/slave pair sharing that CAN bus
- no dataset recording
- no action sending from the host
- only read leader/master control echo + follower/slave observation
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import bi_songling_follower, songling_follower  # noqa: F401
from lerobot.robots.bi_songling_follower import BiSonglingFollower
from lerobot.robots.bi_songling_follower.config_bi_songling_follower import BiSonglingFollowerConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators import bi_openarm_leader, openarm_leader  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class SonglingTeachConfig:
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    fps: int = 30
    display_data: bool = True
    display_compressed_images: bool = False
    dataset: Any | None = None
    play_sounds: bool | None = None
    voice_lang: str | None = None
    voice_rate: int | None = None
    voice_engine: str | None = None
    voice_piper_model: str | None = None
    voice_piper_binary: str | None = None
    voice_piper_speaker: int | None = None
    display_ip: str | None = None
    display_port: int | None = None
    resume: bool | None = None


def _parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false.")


def _load_config(config_path: Path, unknown_args: list[str]) -> SonglingTeachConfig:
    return draccus.parse(SonglingTeachConfig, config_path=config_path, args=unknown_args)


def _apply_robot_overrides(robot_cfg: BiSonglingFollowerConfig, args: argparse.Namespace) -> None:
    left = robot_cfg.left_arm_config
    right = robot_cfg.right_arm_config

    if args.left_interface is not None:
        left.channel = str(args.left_interface)
    if args.right_interface is not None:
        right.channel = str(args.right_interface)

    if args.left_bitrate is not None:
        left.bitrate = int(args.left_bitrate)
    if args.right_bitrate is not None:
        right.bitrate = int(args.right_bitrate)

    if args.left_data_bitrate is not None:
        left.can_data_bitrate = int(args.left_data_bitrate)
    if args.right_data_bitrate is not None:
        right.can_data_bitrate = int(args.right_data_bitrate)

    if args.left_use_fd is not None:
        left.use_can_fd = bool(args.left_use_fd)
    if args.right_use_fd is not None:
        right.use_can_fd = bool(args.right_use_fd)

    if args.left_high is not None:
        left.cameras["high"].index_or_path = args.left_high
    if args.left_elbow is not None:
        left.cameras["elbow"].index_or_path = args.left_elbow
    if args.right_elbow is not None:
        right.cameras["elbow"].index_or_path = args.right_elbow

    for arm_cfg in (left, right):
        arm_cfg.transport_backend = "piper_sdk"
        arm_cfg.allow_unverified_commanding = False
        arm_cfg.auto_enable_on_connect = False
        arm_cfg.auto_configure_mode_on_connect = False
        arm_cfg.auto_configure_master_slave_on_connect = False


def _side_has_command_echo(robot: BiSonglingFollower, side: str) -> bool:
    arm = robot.left_arm if side == "left" else robot.right_arm
    return bool(arm.bus.command_feedback_valid) and all(
        bool(arm.bus.commanded_position_seen.get(joint_name, False))
        for joint_name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
    )


def _build_passthrough_action(
    robot: BiSonglingFollower,
    obs: RobotObservation,
    commanded: RobotAction,
) -> tuple[RobotAction, bool, bool]:
    if not isinstance(obs, dict) or not isinstance(commanded, dict):
        raise TypeError("Expected dict observations/actions.")

    action: RobotAction = {}
    left_has_control = _side_has_command_echo(robot, "left")
    right_has_control = _side_has_command_echo(robot, "right")
    for side, has_control in (("left", left_has_control), ("right", right_has_control)):
        for joint_name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"):
            key = f"{side}_{joint_name}.pos"
            source = commanded if has_control and key in commanded else obs
            action[key] = float(source[key])
    return action, left_has_control, right_has_control


def _iter_camera_frames(obs: RobotObservation) -> list[tuple[str, Any]]:
    frames: list[tuple[str, Any]] = []
    for key, value in obs.items():
        if key.endswith(".pos"):
            continue
        if hasattr(value, "ndim") and getattr(value, "ndim", 0) >= 2:
            frames.append((key, value))
    return frames


def _log_scalar(rr, path: str, value: float) -> None:
    if value == value:
        rr.log(path, rr.Scalars(float(value)))


def _log_frame(rr, path: str, frame: Any, compress_images: bool) -> None:
    entity = rr.Image(frame).compress() if compress_images else rr.Image(frame)
    rr.log(path, entity=entity)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Songling ALOHA teaching monitor.")
    parser.add_argument(
        "--config-path",
        "--config_path",
        type=Path,
        default=Path("examples/songling_aloha/teleop.yaml"),
        help="Path to Songling config yaml.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override FPS from config.")
    parser.add_argument(
        "--display_data",
        "--display-data",
        dest="display_data",
        type=_parse_cli_bool,
        default=None,
        help="Enable Rerun visualization.",
    )
    parser.add_argument(
        "--display_ip",
        "--display-ip",
        dest="display_ip",
        default=None,
        help="Rerun server IP. If omitted, local viewer spawn is used.",
    )
    parser.add_argument(
        "--display_port",
        "--display-port",
        dest="display_port",
        type=int,
        default=None,
        help="Rerun server port (used with --display_ip).",
    )
    parser.add_argument(
        "--display_compressed_images",
        "--display-compressed-images",
        dest="display_compressed_images",
        type=_parse_cli_bool,
        default=None,
        help="Compress images before logging to Rerun.",
    )
    parser.add_argument("--left-interface", "--robot.left_arm_config.channel", dest="left_interface", default=None)
    parser.add_argument("--right-interface", "--robot.right_arm_config.channel", dest="right_interface", default=None)
    parser.add_argument(
        "--left-bitrate", "--robot.left_arm_config.bitrate", dest="left_bitrate", type=int, default=None
    )
    parser.add_argument(
        "--right-bitrate", "--robot.right_arm_config.bitrate", dest="right_bitrate", type=int, default=None
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
        help="Override left high camera device.",
    )
    parser.add_argument(
        "--left-elbow",
        "--robot.left_arm_config.cameras.elbow.index_or_path",
        dest="left_elbow",
        default=None,
        help="Override left elbow camera device.",
    )
    parser.add_argument(
        "--right-elbow",
        "--robot.right_arm_config.cameras.elbow.index_or_path",
        dest="right_elbow",
        default=None,
        help="Override right elbow camera device.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        quoted = " ".join(shlex.quote(item) for item in unknown)
        print(f"[WARN] Ignoring unsupported overrides: {quoted}")

    register_third_party_plugins()
    cfg = _load_config(args.config_path, [])
    if not isinstance(cfg.robot, BiSonglingFollowerConfig):
        raise ValueError(
            "This script currently supports only robot.type=bi_songling_follower, "
            f"but got {type(cfg.robot).__name__ if cfg.robot is not None else None}."
        )

    _apply_robot_overrides(cfg.robot, args)

    fps = args.fps if args.fps is not None else int(cfg.fps)
    display_data = args.display_data if args.display_data is not None else bool(cfg.display_data)
    display_ip = args.display_ip if args.display_ip is not None else cfg.display_ip
    display_port = args.display_port if args.display_port is not None else cfg.display_port
    display_compressed_images = (
        args.display_compressed_images
        if args.display_compressed_images is not None
        else bool(cfg.display_compressed_images)
    )

    print("=" * 50)
    print("Songling ALOHA Live Teaching")
    print("=" * 50)
    print(f"Config: {args.config_path}")
    print(f"FPS: {fps}")
    print(f"Left pair CAN: {cfg.robot.left_arm_config.channel}")
    print(f"Right pair CAN: {cfg.robot.right_arm_config.channel}")
    print("Mode: read-only shared-CAN teaching monitor")
    print("Host behavior: no dataset recording, no action sending, no save/replay")
    print()

    rr = None
    robot = make_robot_from_config(cfg.robot)
    if not isinstance(robot, BiSonglingFollower):
        raise TypeError(f"Expected BiSonglingFollower runtime, got {type(robot).__name__}.")

    try:
        if display_data:
            import rerun as rr_mod  # type: ignore

            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(session_name="songling_aloha_teach", ip=display_ip, port=display_port)
            rr = rr_mod

        robot.connect()
        frame_idx = 0
        last_log_s = 0.0
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            commanded = robot.get_commanded_action(poll=False)
            action, left_has_control, right_has_control = _build_passthrough_action(robot, obs, commanded)

            if rr is not None:
                rr.set_time("frame", sequence=frame_idx)
                for cam_key, frame in _iter_camera_frames(obs):
                    _log_frame(rr, f"cameras/{cam_key}", frame, display_compressed_images)

                for side in ("left", "right"):
                    for joint_name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"):
                        key = f"{side}_{joint_name}.pos"
                        obs_val = float(obs[key])
                        cmd_val = float(action[key])
                        _log_scalar(rr, f"follower/obs/{key}", obs_val)
                        _log_scalar(rr, f"leader/cmd/{key}", cmd_val)
                        _log_scalar(rr, f"tracking/err/{key}", obs_val - cmd_val)
                _log_scalar(rr, "status/left_cmd_echo", 1.0 if left_has_control else 0.0)
                _log_scalar(rr, "status/right_cmd_echo", 1.0 if right_has_control else 0.0)
                _log_scalar(rr, "status/left_obs_hz", float(robot.left_arm.bus.joint_feedback_hz))
                _log_scalar(rr, "status/right_obs_hz", float(robot.right_arm.bus.joint_feedback_hz))
                _log_scalar(rr, "status/left_cmd_hz", float(robot.left_arm.bus.command_feedback_hz))
                _log_scalar(rr, "status/right_cmd_hz", float(robot.right_arm.bus.command_feedback_hz))

            now_s = time.time()
            if now_s - last_log_s >= 1.0:
                print(
                    "[INFO] frame=%d left_cmd_echo=%s right_cmd_echo=%s "
                    "left_obs_hz=%.1f right_obs_hz=%.1f left_cmd_hz=%.1f right_cmd_hz=%.1f "
                    "left_joint_1(cmd/obs)=%.3f/%.3f right_joint_1(cmd/obs)=%.3f/%.3f"
                    % (
                        frame_idx,
                        left_has_control,
                        right_has_control,
                        robot.left_arm.bus.joint_feedback_hz,
                        robot.right_arm.bus.joint_feedback_hz,
                        robot.left_arm.bus.command_feedback_hz,
                        robot.right_arm.bus.command_feedback_hz,
                        float(action["left_joint_1.pos"]),
                        float(obs["left_joint_1.pos"]),
                        float(action["right_joint_1.pos"]),
                        float(obs["right_joint_1.pos"]),
                    ),
                    flush=True,
                )
                last_log_s = now_s

            frame_idx += 1
            dt = time.perf_counter() - t0
            precise_sleep(max((1.0 / fps) - dt, 0.0))
    except KeyboardInterrupt:
        print("[INFO] Received Ctrl+C, stopping live teaching monitor.", flush=True)
    finally:
        if robot.is_connected:
            robot.disconnect()
        if rr is not None:
            try:
                rr.rerun_shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
