#!/usr/bin/env python

"""Run ACT real-robot inference for the Songling ALOHA bimanual setup.

This script is intentionally closer to `lerobot-record` CLI ergonomics than to
the tutorial examples:

- it reuses `examples/songling_aloha/teleop.yaml` as the single source of truth
  for CAN ports and camera devices
- it accepts familiar dotted overrides like `--robot.left_arm_config.port=can1`
  and `--policy.path=...`
- it runs a pure policy control loop, so no dataset creation and no
  `dataset.single_task` are required

Examples:
    python examples/songling_aloha/run_act_inference.py \
      --config-path examples/songling_aloha/teleop.yaml \
      --policy.path=outputs/train/act_songling_plate_stack_v1 \
      --policy.device=cuda \
      --display_data=true

    python examples/songling_aloha/run_act_inference.py \
      --policy.path=outputs/train/act_songling_plate_stack_v1/checkpoints/040000/pretrained_model \
      --display_data=false \
      --duration=120
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import rerun as rr
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots import bi_openarm_follower, openarm_follower  # noqa: F401
from lerobot.robots import bi_songling_follower, songling_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators import bi_openarm_leader, openarm_leader  # noqa: F401
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

try:
    from shadow_env import (
        DEFAULT_ACTION_IDS,
        DEFAULT_JOINT_NAMES,
        DEFAULT_OBSERVATION_IDS,
        SonglingShadowEnv,
    )
except ModuleNotFoundError:
    from examples.songling_aloha.shadow_env import (
        DEFAULT_ACTION_IDS,
        DEFAULT_JOINT_NAMES,
        DEFAULT_OBSERVATION_IDS,
        SonglingShadowEnv,
    )

DEFAULT_CONFIG_PATH = Path("examples/songling_aloha/teleop.yaml")
DEFAULT_SESSION_NAME = "songling_aloha_act_inference"
CAMERA_INFO_NAMES = ["height", "width", "channels"]
SONGLING_POS_NAMES = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_gripper.pos",
]
SONGLING_CAMERA_MAP = {
    "observation.images.left_high": ("left_arm_config", "high"),
    "observation.images.left_elbow": ("left_arm_config", "elbow"),
    "observation.images.right_elbow": ("right_arm_config", "elbow"),
}

logger = logging.getLogger(__name__)


@dataclass
class SonglingUnifiedConfig:
    """Local superset config so `teleop.yaml` can be reused without rewriting it."""

    robot: RobotConfig
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


def _is_bimanual_songling_integrated_chain(cfg: SonglingUnifiedConfig) -> bool:
    """Detect the current Songling integrated-chain YAML shape.

    The maintained `teleop.yaml` models one integrated leader+follower chain per side.
    For this hardware, `teleop` and `robot` typically point to the same CAN interface
    on each side. That is not compatible with the generic OpenArm Damiao runtime.
    """

    if cfg.teleop is None:
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


def _parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false.")


def _parse_songling_config(config_path: Path, unknown_args: list[str]) -> SonglingUnifiedConfig:
    return draccus.parse(SonglingUnifiedConfig, config_path=config_path, args=unknown_args)


def _feature_shape(feature: Any) -> list[int]:
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if shape is None:
        raise ValueError(f"Could not read feature shape from {feature!r}.")
    return list(shape)


def _camera_cfg_from_robot_cfg(robot_cfg: RobotConfig, feature_key: str):
    arm_name, camera_name = SONGLING_CAMERA_MAP[feature_key]
    arm_cfg = getattr(robot_cfg, arm_name, None)
    if arm_cfg is None:
        raise ValueError(f"Robot config is missing {arm_name}.")
    camera_cfg = getattr(arm_cfg, "cameras", {}).get(camera_name)
    if camera_cfg is None:
        raise ValueError(
            f"Robot config is missing camera '{camera_name}' under '{arm_name}', "
            f"required by model feature '{feature_key}'."
        )
    return camera_cfg


def _build_dataset_features(policy: ACTPolicy, robot_cfg: RobotConfig) -> dict[str, dict[str, Any]]:
    input_features = policy.config.input_features
    output_features = policy.config.output_features

    state_feature = input_features.get("observation.state")
    if state_feature is None:
        raise ValueError("ACT policy is missing required input feature 'observation.state'.")
    if _feature_shape(state_feature) != [len(SONGLING_POS_NAMES)]:
        raise ValueError(
            "Songling ACT inference expects a 14-dim observation.state, "
            f"but got {_feature_shape(state_feature)}."
        )

    action_feature = output_features.get(ACTION)
    if action_feature is None:
        raise ValueError("ACT policy is missing required output feature 'action'.")
    if _feature_shape(action_feature) != [len(SONGLING_POS_NAMES)]:
        raise ValueError(
            "Songling ACT inference expects a 14-dim action output, "
            f"but got {_feature_shape(action_feature)}."
        )

    dataset_features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(SONGLING_POS_NAMES)],
            "names": SONGLING_POS_NAMES,
        },
        ACTION: {
            "dtype": "float32",
            "shape": [len(SONGLING_POS_NAMES)],
            "names": SONGLING_POS_NAMES,
        },
    }

    for feature_key, feature in input_features.items():
        if feature_key == "observation.state":
            continue

        if feature_key not in SONGLING_CAMERA_MAP:
            raise ValueError(
                "Songling ACT inference only supports the camera features "
                f"{sorted(SONGLING_CAMERA_MAP)}, but the model expects '{feature_key}'."
            )

        shape = _feature_shape(feature)
        if len(shape) != 3:
            raise ValueError(f"Expected 3D image feature for {feature_key}, got shape {shape}.")

        camera_cfg = _camera_cfg_from_robot_cfg(robot_cfg, feature_key)
        expected_hwc = [shape[1], shape[2], shape[0]]
        if camera_cfg.height != expected_hwc[0] or camera_cfg.width != expected_hwc[1]:
            raise ValueError(
                f"Camera '{feature_key}' expects resolution {expected_hwc[1]}x{expected_hwc[0]}, "
                f"but config provides {camera_cfg.width}x{camera_cfg.height}."
            )

        dataset_features[feature_key] = {
            "dtype": "video",
            "shape": expected_hwc,
            "names": CAMERA_INFO_NAMES,
        }

    return dataset_features


def _log_shadow_metrics(
    *,
    joint_names: list[str],
    left_obs: np.ndarray,
    right_obs: np.ndarray,
    left_bus_action: np.ndarray,
    right_bus_action: np.ndarray,
    left_candidate_action: np.ndarray,
    right_candidate_action: np.ndarray,
    predicted_action: dict[str, float],
    left_hz: float,
    right_hz: float,
) -> None:
    rr.log("can.left.rx_hz", rr.Scalars(float(left_hz)))
    rr.log("can.right.rx_hz", rr.Scalars(float(right_hz)))
    for idx, joint_name in enumerate(joint_names):
        obs_left_key = f"left_{joint_name}.pos"
        obs_right_key = f"right_{joint_name}.pos"
        pred_left = float(predicted_action[obs_left_key])
        pred_right = float(predicted_action[obs_right_key])
        bus_left = float(left_bus_action[idx])
        bus_right = float(right_bus_action[idx])
        cand_left = float(left_candidate_action[idx])
        cand_right = float(right_candidate_action[idx])
        rr.log(f"shadow/pred.left_{joint_name}.pos", rr.Scalars(pred_left))
        rr.log(f"shadow/pred.right_{joint_name}.pos", rr.Scalars(pred_right))
        rr.log(f"shadow/bus.left_{joint_name}.pos", rr.Scalars(bus_left))
        rr.log(f"shadow/bus.right_{joint_name}.pos", rr.Scalars(bus_right))
        rr.log(f"shadow/error.left_{joint_name}.pos", rr.Scalars(pred_left - bus_left))
        rr.log(f"shadow/error.right_{joint_name}.pos", rr.Scalars(pred_right - bus_right))
        rr.log(f"shadow/candidate.left_{joint_name}.pos", rr.Scalars(cand_left))
        rr.log(f"shadow/candidate.right_{joint_name}.pos", rr.Scalars(cand_right))
        rr.log(f"shadow/error_vs_candidate.left_{joint_name}.pos", rr.Scalars(pred_left - cand_left))
        rr.log(f"shadow/error_vs_candidate.right_{joint_name}.pos", rr.Scalars(pred_right - cand_right))
        rr.log(f"shadow/error_vs_obs.left_{joint_name}.pos", rr.Scalars(pred_left - float(left_obs[idx])))
        rr.log(f"shadow/error_vs_obs.right_{joint_name}.pos", rr.Scalars(pred_right - float(right_obs[idx])))
        rr.log(f"shadow/obs.left_{joint_name}.pos", rr.Scalars(float(left_obs[idx])))
        rr.log(f"shadow/obs.right_{joint_name}.pos", rr.Scalars(float(right_obs[idx])))


def _run_raw_can_shadow_inference(
    *,
    args: argparse.Namespace,
    policy: ACTPolicy,
    device: torch.device,
    dataset_features: dict[str, dict[str, Any]],
    preprocessor,
    postprocessor,
    fps: int,
    display_data: bool,
    display_ip: str | None,
    display_port: int | None,
    display_compressed_images: bool,
) -> None:
    env = SonglingShadowEnv.from_cli_args(args=args, fallback_fps=fps)
    if len(env.joint_names) * 2 != len(SONGLING_POS_NAMES):
        raise ValueError(
            f"Raw CAN mode currently expects {len(SONGLING_POS_NAMES) // 2} joints per side, "
            f"but got {len(env.joint_names)} from --joint-names."
        )
    last_log = 0.0

    logger.info(
        "Running Songling ACT shadow inference over raw CAN.\n"
        "Predicted actions will be computed and visualized, but not sent to the robot."
    )
    logger.info(
        "Left CAN: %s (%s, bitrate=%s), Right CAN: %s (%s, bitrate=%s)",
        env.left_can.interface,
        "FD" if env.left_can.use_fd else "CAN2.0",
        env.left_can.bitrate,
        env.right_can.interface,
        "FD" if env.right_can.use_fd else "CAN2.0",
        env.right_can.bitrate,
    )
    if args.action_ids == DEFAULT_ACTION_IDS and args.left_action_ids is None and args.right_action_ids is None:
        logger.warning(
            "Raw CAN mode is using Piper-style default action frame IDs (%s). "
            "If your integrated chain does not expose 0x155/0x156/0x157/0x159 on the bus, "
            "shadow mode will fall back to observation feedback as the action proxy.",
            DEFAULT_ACTION_IDS,
        )

    if display_data:
        init_rerun(session_name=args.session_name, ip=display_ip, port=display_port)

    try:
        env.connect()
        for cam_name in env.cameras:
            logger.info("Connected camera: %s", cam_name)
        policy.reset()
        start_time = time.perf_counter()
        with torch.inference_mode():
            while True:
                loop_start = time.perf_counter()
                if args.duration_s is not None and loop_start - start_time >= args.duration_s:
                    logger.info("Reached requested duration, stopping shadow inference loop.")
                    break

                step = env.poll()
                batch = build_inference_frame(
                    observation=step.observation,
                    device=device,
                    ds_features=dataset_features,
                    task=None,
                    robot_type="songling_aloha_raw_can",
                )
                batch = preprocessor(batch)

                action = policy.select_action(batch)
                action = postprocessor(action)
                predicted_action = make_robot_action(action, dataset_features)

                if display_data:
                    rr.set_time("frame", sequence=step.frame_index)
                    log_rerun_data(
                        observation=step.observation,
                        action=predicted_action,
                        compress_images=display_compressed_images,
                    )
                    _log_shadow_metrics(
                        joint_names=env.joint_names,
                        left_obs=step.left_obs,
                        right_obs=step.right_obs,
                        left_bus_action=step.left_bus_action,
                        right_bus_action=step.right_bus_action,
                        left_candidate_action=step.left_candidate_action,
                        right_candidate_action=step.right_candidate_action,
                        predicted_action=predicted_action,
                        left_hz=step.left_rx_hz,
                        right_hz=step.right_rx_hz,
                    )
                    for cam_key, health in step.camera_health.items():
                        rr.log(f"cameras.{cam_key}.stale_count", rr.Scalars(health["stale_count"]))
                        rr.log(f"cameras.{cam_key}.identical_count", rr.Scalars(health["identical_count"]))
                        rr.log(f"cameras.{cam_key}.used_cached", rr.Scalars(health["used_cached"]))
                        rr.log(f"cameras.{cam_key}.reconnected", rr.Scalars(health["reconnected"]))

                now = time.time()
                if now - last_log >= 1.0:
                    left_pred = predicted_action["left_joint_1.pos"]
                    right_pred = predicted_action["right_joint_1.pos"]
                    left_obs_joint1 = float(step.left_obs[0])
                    right_obs_joint1 = float(step.right_obs[0])
                    left_obs_gripper = float(step.left_obs[-1])
                    right_obs_gripper = float(step.right_obs[-1])
                    left_cand_joint1 = float(step.left_candidate_action[0])
                    right_cand_joint1 = float(step.right_candidate_action[0])
                    logger.info(
                        "frame=%d left_rx_hz=%.1f right_rx_hz=%.1f "
                        "obs_left_joint_1=%.3f obs_right_joint_1=%.3f "
                        "obs_left_gripper=%.3f obs_right_gripper=%.3f "
                        "cand_left_joint_1=%.3f cand_right_joint_1=%.3f "
                        "pred_left_joint_1=%.3f pred_right_joint_1=%.3f",
                        step.frame_index,
                        step.left_rx_hz,
                        step.right_rx_hz,
                        left_obs_joint1,
                        right_obs_joint1,
                        left_obs_gripper,
                        right_obs_gripper,
                        left_cand_joint1,
                        right_cand_joint1,
                        left_pred,
                        right_pred,
                    )
                    last_log = now

                precise_sleep(max(1.0 / fps - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping shadow inference loop.")
    finally:
        env.disconnect()
        if display_data:
            try:
                rr.rerun_shutdown()
            except Exception:  # nosec B110
                pass


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run ACT inference on Songling ALOHA without dataset recording.")
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to Songling unified YAML config.",
    )
    parser.add_argument(
        "--policy-path",
        "--policy.path",
        dest="policy_path",
        required=True,
        help="Local checkpoint directory or Hub repo id for the ACT policy.",
    )
    parser.add_argument(
        "--device",
        "--policy.device",
        dest="policy_device",
        default=None,
        help="Torch device for inference, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--fps",
        dest="fps",
        type=int,
        default=None,
        help="Control loop frequency. Defaults to the Songling YAML fps.",
    )
    parser.add_argument(
        "--duration",
        "--control-time-s",
        "--control_time_s",
        dest="duration_s",
        type=float,
        default=None,
        help="Optional max runtime in seconds. If omitted, run until Ctrl+C.",
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
        "--raw-can-mode",
        action="store_true",
        help="For Songling integrated chains, run shadow inference from raw CAN + cameras without sending actions.",
    )
    parser.add_argument(
        "--joint-names",
        default=",".join(DEFAULT_JOINT_NAMES),
        help="Comma-separated joint names per side for raw CAN mode.",
    )
    parser.add_argument(
        "--observation-ids",
        default=DEFAULT_OBSERVATION_IDS,
        help="Default CAN ids mapped to observation joints in raw CAN mode.",
    )
    parser.add_argument(
        "--action-ids",
        default=DEFAULT_ACTION_IDS,
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
        help="Abort raw shadow inference when camera reconnect cannot recover from freeze.",
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
    cfg = _parse_songling_config(args.config_path, unknown_args)

    init_logging()

    if cfg.robot.type not in {"bi_openarm_follower", "bi_songling_follower"}:
        raise ValueError(
            "Songling ACT inference expects robot.type in {bi_openarm_follower, bi_songling_follower}, "
            f"but got {cfg.robot.type!r}."
        )

    integrated_chain = _is_bimanual_songling_integrated_chain(cfg)

    display_data = cfg.display_data if args.display_data is None else _parse_cli_bool(args.display_data)
    display_ip = cfg.display_ip if args.display_ip is None else args.display_ip
    display_port = cfg.display_port if args.display_port is None else args.display_port
    display_compressed_images = (
        cfg.display_compressed_images
        if args.display_compressed_images is None
        else _parse_cli_bool(args.display_compressed_images)
    )
    if display_data and display_ip is not None and display_port is not None:
        display_compressed_images = True

    policy = ACTPolicy.from_pretrained(args.policy_path)
    requested_device = args.policy_device or str(policy.config.device)
    device = get_safe_torch_device(requested_device, log=True)
    policy.config.device = str(device)
    policy = policy.to(device).eval()

    dataset_features = _build_dataset_features(policy, cfg.robot)
    fps = args.fps if args.fps is not None else cfg.fps

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    logger.info(
        "Songling ACT inference config:\n%s",
        "\n".join(
            [
                f"  policy_path: {args.policy_path}",
                f"  device: {device}",
                f"  fps: {fps}",
                f"  duration_s: {args.duration_s}",
                f"  integrated_chain: {integrated_chain}",
                f"  raw_can_mode: {args.raw_can_mode}",
                f"  display_data: {display_data}",
                f"  display_ip: {display_ip}",
                f"  display_port: {display_port}",
            ]
        ),
    )

    if integrated_chain and not args.raw_can_mode:
        raise NotImplementedError(
            "This Songling config models an integrated leader+follower CAN chain on each side "
            "(teleop and robot share the same can0/can1 ports). The generic OpenArm/Damiao "
            "runtime used by this ACT script is not compatible with that hardware layout, so the "
            "motor handshake failure you saw is expected.\n\n"
            "Use one of the following instead:\n"
            "1. `python examples/songling_aloha/run_act_inference.py --config-path examples/songling_aloha/teleop.yaml "
            "--policy.path=... --policy.device=cuda --raw-can-mode --display_data=true`\n"
            "   This runs online ACT shadow inference from raw CAN + cameras and visualizes predicted actions, "
            "but does not send control commands yet.\n"
            "2. `python examples/songling_aloha/visualize_teleop_live.py --config-path examples/songling_aloha/teleop.yaml --raw-can-mode`\n"
            "3. `python examples/songling_aloha/record_raw_can_dataset.py --config-path examples/songling_aloha/teleop.yaml ...`\n\n"
            "Real closed-loop deployment still requires a dedicated Songling MotorsBus / robot adapter."
        )

    if args.raw_can_mode:
        _run_raw_can_shadow_inference(
            args=args,
            policy=policy,
            device=device,
            dataset_features=dataset_features,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            fps=fps,
            display_data=display_data,
            display_ip=display_ip,
            display_port=display_port,
            display_compressed_images=display_compressed_images,
        )
        return

    robot = make_robot_from_config(cfg.robot)

    if display_data:
        init_rerun(session_name=args.session_name, ip=display_ip, port=display_port)

    robot.connect()
    policy.reset()

    logger.info("Songling ACT inference started. Press Ctrl+C to stop.")

    start_time = time.perf_counter()
    try:
        with torch.inference_mode():
            while True:
                loop_start = time.perf_counter()

                if args.duration_s is not None and loop_start - start_time >= args.duration_s:
                    logger.info("Reached requested duration, stopping inference loop.")
                    break

                observation = robot.get_observation()
                batch = build_inference_frame(
                    observation=observation,
                    device=device,
                    ds_features=dataset_features,
                    task=None,
                    robot_type=robot.name,
                )
                batch = preprocessor(batch)

                action = policy.select_action(batch)
                action = postprocessor(action)
                robot_action = make_robot_action(action, dataset_features)
                sent_action = robot.send_action(robot_action)

                if display_data:
                    log_rerun_data(
                        observation=observation,
                        action=sent_action,
                        compress_images=display_compressed_images,
                    )

                precise_sleep(max(1.0 / fps - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping inference loop.")
    finally:
        try:
            robot.disconnect()
        finally:
            if display_data:
                try:
                    rr.rerun_shutdown()
                except Exception:  # nosec B110
                    pass


if __name__ == "__main__":
    main()
