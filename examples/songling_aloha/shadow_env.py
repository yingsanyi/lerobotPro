#!/usr/bin/env python

"""Raw CAN + camera shadow environment for Songling integrated chains.

This module provides a reusable environment-style wrapper around the existing
Songling raw CAN parsing utilities so scripts can consume live observations in
a shape similar to `mobile-aloha`'s `RealEnv`, without yet sending control
commands back to the integrated chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs

try:
    from record_raw_can_dataset import (
        CANBusState,
        CANDecodeOptions,
        DEFAULT_ACTION_IDS,
        DEFAULT_JOINT_NAMES,
        DEFAULT_OBSERVATION_IDS,
        _load_songling_yaml,
        _open_raw_can_bus,
        _parse_csv as _raw_parse_csv,
        _decode_piper_action_vector,
        _decode_piper_observation_vector,
        _poll_can_state,
        _read_camera_frames_with_health,
        _resolve_hardware,
        _rx_hz,
        _validate_camera_devices,
    )
except ModuleNotFoundError:
    from examples.songling_aloha.record_raw_can_dataset import (
        CANBusState,
        CANDecodeOptions,
        DEFAULT_ACTION_IDS,
        DEFAULT_JOINT_NAMES,
        DEFAULT_OBSERVATION_IDS,
        _load_songling_yaml,
        _open_raw_can_bus,
        _parse_csv as _raw_parse_csv,
        _decode_piper_action_vector,
        _decode_piper_observation_vector,
        _poll_can_state,
        _read_camera_frames_with_health,
        _resolve_hardware,
        _rx_hz,
        _validate_camera_devices,
    )


def build_position_feature_names(joint_names: list[str]) -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        for joint_name in joint_names:
            names.append(f"{side}_{joint_name}.pos")
    return names


def build_observation_dict(
    joint_names: list[str],
    left_obs: np.ndarray,
    right_obs: np.ndarray,
    camera_frames: dict[str, np.ndarray],
) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    names = build_position_feature_names(joint_names)
    values = np.concatenate((left_obs, right_obs), axis=0).astype(np.float32, copy=False)
    for name, value in zip(names, values, strict=True):
        observation[name] = float(value)
    observation.update(camera_frames)
    return observation


def build_action_dict(
    joint_names: list[str],
    left_action: np.ndarray,
    right_action: np.ndarray,
) -> dict[str, float]:
    action: dict[str, float] = {}
    for idx, joint_name in enumerate(joint_names):
        action[f"left_{joint_name}.pos"] = float(left_action[idx])
        action[f"right_{joint_name}.pos"] = float(right_action[idx])
    return action


@dataclass
class SonglingShadowStep:
    frame_index: int
    observation: dict[str, Any]
    bus_action: dict[str, float]
    candidate_action: dict[str, float]
    left_obs: np.ndarray
    right_obs: np.ndarray
    left_bus_action: np.ndarray
    right_bus_action: np.ndarray
    left_candidate_action: np.ndarray
    right_candidate_action: np.ndarray
    left_rx_hz: float
    right_rx_hz: float
    camera_health: dict[str, dict[str, Any]]


class SonglingShadowEnv:
    """Live Songling environment backed by raw CAN parsing and UVC cameras."""

    def __init__(
        self,
        *,
        joint_names: list[str],
        left_can,
        right_can,
        camera_configs: dict[str, Any],
        decode_opts: CANDecodeOptions,
        max_can_poll_msgs: int,
        camera_max_age_ms: int,
        camera_retry_timeout_ms: int,
        camera_reconnect_stale_count: int,
        camera_freeze_identical_count: int,
        camera_reconnect_cooldown_s: float,
        camera_fail_on_freeze: bool,
    ) -> None:
        self.joint_names = joint_names
        self.left_can = left_can
        self.right_can = right_can
        self.camera_configs = camera_configs
        self.decode_opts = decode_opts
        self.max_can_poll_msgs = max_can_poll_msgs
        self.camera_max_age_ms = camera_max_age_ms
        self.camera_retry_timeout_ms = camera_retry_timeout_ms
        self.camera_reconnect_stale_count = camera_reconnect_stale_count
        self.camera_freeze_identical_count = camera_freeze_identical_count
        self.camera_reconnect_cooldown_s = camera_reconnect_cooldown_s
        self.camera_fail_on_freeze = camera_fail_on_freeze

        self.cameras: dict[str, Any] = {}
        self.left_bus = None
        self.right_bus = None
        self.left_state = CANBusState()
        self.right_state = CANBusState()
        self.left_obs_last = np.zeros((len(joint_names),), dtype=np.float32)
        self.right_obs_last = np.zeros((len(joint_names),), dtype=np.float32)
        self.left_action_last = np.zeros((len(joint_names),), dtype=np.float32)
        self.right_action_last = np.zeros((len(joint_names),), dtype=np.float32)
        self.left_candidate_action_last = np.zeros((len(joint_names),), dtype=np.float32)
        self.right_candidate_action_last = np.zeros((len(joint_names),), dtype=np.float32)
        self.last_camera_frames: dict[str, np.ndarray] = {}
        self.camera_stale_counts: dict[str, int] = {}
        self.camera_identical_counts: dict[str, int] = {}
        self.camera_last_signatures: dict[str, int] = {}
        self.camera_last_reconnect_ts: dict[str, float] = {}
        self.frame_index = 0

    @classmethod
    def from_cli_args(cls, args, fallback_fps: int) -> "SonglingShadowEnv":
        raw_cfg = _load_songling_yaml(args.config_path)
        joint_names = _raw_parse_csv(args.joint_names)
        if not joint_names:
            raise ValueError("No joint names resolved. Please set --joint-names.")

        left_can, right_can, camera_configs = _resolve_hardware(
            raw_cfg=raw_cfg,
            args=args,
            joint_names=joint_names,
            fallback_fps=fallback_fps,
        )
        decode_opts = CANDecodeOptions(
            byte_offset=args.decode_byte_offset,
            byte_length=args.decode_byte_length,
            signed=args.decode_signed,
            endian=args.decode_endian,
            scale=args.decode_scale,
            bias=args.decode_bias,
        )
        return cls(
            joint_names=joint_names,
            left_can=left_can,
            right_can=right_can,
            camera_configs=camera_configs,
            decode_opts=decode_opts,
            max_can_poll_msgs=args.max_can_poll_msgs,
            camera_max_age_ms=args.camera_max_age_ms,
            camera_retry_timeout_ms=args.camera_retry_timeout_ms,
            camera_reconnect_stale_count=args.camera_reconnect_stale_count,
            camera_freeze_identical_count=args.camera_freeze_identical_count,
            camera_reconnect_cooldown_s=args.camera_reconnect_cooldown_s,
            camera_fail_on_freeze=args.camera_fail_on_freeze,
        )

    def connect(self) -> None:
        try:
            import can  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("python-can is required. Install with `pip install python-can`.") from exc

        _validate_camera_devices(self.camera_configs)
        self.cameras = make_cameras_from_configs(self.camera_configs)
        for cam in self.cameras.values():
            cam.connect(warmup=True)

        self.left_bus = _open_raw_can_bus(
            interface=self.left_can.interface,
            bitrate=self.left_can.bitrate,
            use_fd=self.left_can.use_fd,
            data_bitrate=self.left_can.data_bitrate,
        )
        self.right_bus = _open_raw_can_bus(
            interface=self.right_can.interface,
            bitrate=self.right_can.bitrate,
            use_fd=self.right_can.use_fd,
            data_bitrate=self.right_can.data_bitrate,
        )

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # nosec B110
                pass
        self.cameras = {}

        if self.left_bus is not None:
            try:
                self.left_bus.shutdown()
            except Exception:  # nosec B110
                pass
            self.left_bus = None

        if self.right_bus is not None:
            try:
                self.right_bus.shutdown()
            except Exception:  # nosec B110
                pass
            self.right_bus = None

    def poll(self) -> SonglingShadowStep:
        if self.left_bus is None or self.right_bus is None:
            raise RuntimeError("SonglingShadowEnv is not connected.")

        _poll_can_state(self.left_bus, self.left_state, self.decode_opts, max_msgs=self.max_can_poll_msgs)
        _poll_can_state(self.right_bus, self.right_state, self.decode_opts, max_msgs=self.max_can_poll_msgs)

        camera_frames, camera_health = _read_camera_frames_with_health(
            cameras=self.cameras,
            last_frames=self.last_camera_frames,
            camera_max_age_ms=self.camera_max_age_ms,
            camera_retry_timeout_ms=self.camera_retry_timeout_ms,
            stale_counts=self.camera_stale_counts,
            identical_counts=self.camera_identical_counts,
            last_signatures=self.camera_last_signatures,
            last_reconnect_ts=self.camera_last_reconnect_ts,
            camera_reconnect_stale_count=self.camera_reconnect_stale_count,
            camera_freeze_identical_count=self.camera_freeze_identical_count,
            camera_reconnect_cooldown_s=self.camera_reconnect_cooldown_s,
            camera_fail_on_freeze=self.camera_fail_on_freeze,
        )

        left_obs = _decode_piper_observation_vector(self.left_can.observation_ids, self.left_state, self.left_obs_last)
        right_obs = _decode_piper_observation_vector(
            self.right_can.observation_ids, self.right_state, self.right_obs_last
        )
        left_bus_action, _ = _decode_piper_action_vector(
            self.left_can.action_ids,
            self.left_state,
            self.left_action_last,
            fallback_observation=left_obs,
        )
        right_bus_action, _ = _decode_piper_action_vector(
            self.right_can.action_ids,
            self.right_state,
            self.right_action_last,
            fallback_observation=right_obs,
        )
        left_candidate_action = left_bus_action.copy()
        right_candidate_action = right_bus_action.copy()
        left_rx_hz = _rx_hz(self.left_state.recent_timestamps)
        right_rx_hz = _rx_hz(self.right_state.recent_timestamps)

        step = SonglingShadowStep(
            frame_index=self.frame_index,
            observation=build_observation_dict(self.joint_names, left_obs, right_obs, camera_frames),
            bus_action=build_action_dict(self.joint_names, left_bus_action, right_bus_action),
            candidate_action=build_action_dict(self.joint_names, left_candidate_action, right_candidate_action),
            left_obs=left_obs,
            right_obs=right_obs,
            left_bus_action=left_bus_action,
            right_bus_action=right_bus_action,
            left_candidate_action=left_candidate_action,
            right_candidate_action=right_candidate_action,
            left_rx_hz=left_rx_hz,
            right_rx_hz=right_rx_hz,
            camera_health=camera_health,
        )
        self.frame_index += 1
        return step


__all__ = [
    "DEFAULT_ACTION_IDS",
    "DEFAULT_JOINT_NAMES",
    "DEFAULT_OBSERVATION_IDS",
    "SonglingShadowEnv",
    "SonglingShadowStep",
    "build_action_dict",
    "build_observation_dict",
    "build_position_feature_names",
]
