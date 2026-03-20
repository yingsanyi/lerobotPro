#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _can_available

if _can_available:
    import can
else:
    can = None

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_songling_follower import SonglingFollowerConfig
from .protocol import (
    ARM_GRIPPER_CTRL,
    ARM_GRIPPER_FEEDBACK,
    ARM_INFO_HIGH_SPD_FEEDBACK_1,
    ARM_INFO_HIGH_SPD_FEEDBACK_6,
    ARM_INFO_LOW_SPD_FEEDBACK_1,
    ARM_INFO_LOW_SPD_FEEDBACK_6,
    ARM_JOINT_CTRL_12,
    ARM_JOINT_CTRL_34,
    ARM_JOINT_CTRL_56,
    ARM_JOINT_FEEDBACK_12,
    ARM_JOINT_FEEDBACK_56,
    ARM_MOTION_CTRL_1,
    ARM_MOTION_CTRL_2,
    ARM_MOTOR_ENABLE_DISABLE_CONFIG,
    ARM_STATUS_FEEDBACK,
    DEFAULT_SONGLING_JOINT_NAMES,
    JOINT_COMMAND_IDS,
    decode_gripper_feedback,
    decode_high_speed_feedback,
    decode_joint_feedback,
    decode_low_speed_feedback,
    decode_status_feedback,
    encode_gripper_command,
    encode_joint_command_pair,
    encode_motion_ctrl_1,
    encode_motion_ctrl_2,
    encode_motor_enable_disable,
    installation_pos_for_side,
)

logger = logging.getLogger(__name__)


class SonglingCANPort:
    """Minimal CAN transport and Piper-like Songling frame codec."""

    def __init__(self, config: SonglingFollowerConfig):
        self.config = config
        self.bus: can.BusABC | None = None  # type: ignore[assignment]
        self._is_connected = False
        self.status: dict[str, int] = {}
        self.joint_positions_raw = {name: 0 for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.joint_position_seen = {name: False for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.high_speed_feedback: dict[str, dict[str, int]] = {}
        self.low_speed_feedback: dict[str, dict[str, int]] = {}
        self.last_sent_raw = {name: 0 for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.last_sent_ts = 0.0
        self.last_mode_ts = 0.0
        self.mode_initialized = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.bus is not None

    def connect(self) -> None:
        if can is None:
            raise ModuleNotFoundError("python-can is required for Songling CAN control.")

        kwargs: dict[str, Any] = {
            "channel": self.config.port,
            "interface": self.config.can_interface,
            "bitrate": self.config.can_bitrate,
        }
        if self.config.use_can_fd:
            kwargs["fd"] = True
            kwargs["data_bitrate"] = self.config.can_data_bitrate
        self.bus = can.interface.Bus(**kwargs)
        self._is_connected = True
        self.poll(max_msgs=self.config.poll_max_msgs)

    def disconnect(self) -> None:
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None
        self._is_connected = False

    def poll(self, *, max_msgs: int | None = None) -> None:
        if self.bus is None:
            return
        budget = self.config.poll_max_msgs if max_msgs is None else max_msgs
        got = 0
        while got < budget:
            msg = self.bus.recv(timeout=0.0)
            if msg is None:
                break
            got += 1
            arbitration_id = int(msg.arbitration_id)
            payload = bytes(msg.data)
            try:
                if arbitration_id == ARM_STATUS_FEEDBACK:
                    status = decode_status_feedback(payload)
                    self.status = {
                        "ctrl_mode": status.ctrl_mode,
                        "arm_status": status.arm_status,
                        "mode_feed": status.mode_feed,
                        "teach_status": status.teach_status,
                        "motion_status": status.motion_status,
                        "trajectory_num": status.trajectory_num,
                        "err_code": status.err_code,
                    }
                elif ARM_JOINT_FEEDBACK_12 <= arbitration_id <= ARM_JOINT_FEEDBACK_56:
                    for name, raw in decode_joint_feedback(arbitration_id, payload).items():
                        self.joint_positions_raw[name] = raw
                        self.joint_position_seen[name] = True
                elif arbitration_id == ARM_GRIPPER_FEEDBACK:
                    gripper = decode_gripper_feedback(payload)
                    self.joint_positions_raw["gripper"] = gripper.position
                    self.joint_position_seen["gripper"] = True
                    self.high_speed_feedback["gripper"] = {
                        "effort": gripper.effort,
                        "status_code": gripper.status_code,
                    }
                elif ARM_INFO_HIGH_SPD_FEEDBACK_1 <= arbitration_id <= ARM_INFO_HIGH_SPD_FEEDBACK_6:
                    feedback = decode_high_speed_feedback(arbitration_id, payload)
                    self.high_speed_feedback[feedback.joint_name] = {
                        "motor_speed": feedback.motor_speed,
                        "current": feedback.current,
                        "position": feedback.position,
                    }
                elif ARM_INFO_LOW_SPD_FEEDBACK_1 <= arbitration_id <= ARM_INFO_LOW_SPD_FEEDBACK_6:
                    feedback = decode_low_speed_feedback(arbitration_id, payload)
                    self.low_speed_feedback[feedback.joint_name] = {
                        "voltage": feedback.voltage,
                        "foc_temp": feedback.foc_temp,
                        "motor_temp": feedback.motor_temp,
                        "foc_status_code": feedback.foc_status_code,
                        "bus_current": feedback.bus_current,
                    }
            except Exception:
                logger.debug("Failed to parse Songling CAN frame 0x%X payload=%s", arbitration_id, payload.hex())

    def _send(self, arbitration_id: int, payload: bytes) -> None:
        if self.bus is None:
            raise RuntimeError("Songling CAN bus is not connected.")
        msg = can.Message(  # type: ignore[union-attr]
            arbitration_id=arbitration_id,
            data=payload,
            is_extended_id=False,
            is_fd=bool(self.config.use_can_fd),
        )
        self.bus.send(msg)

    def enable_motors(self, *, enable: bool) -> None:
        payload = encode_motor_enable_disable(motor_num=0xFF, enable_flag=0x02 if enable else 0x01)
        self._send(ARM_MOTOR_ENABLE_DISABLE_CONFIG, payload)

    def ensure_can_command_mode(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and self.mode_initialized and (now - self.last_mode_ts) < max(self.config.mode_keepalive_s, 0.0):
            return
        installation_pos = (
            self.config.installation_pos
            if self.config.installation_pos is not None
            else installation_pos_for_side(self.config.side)
        )
        payload = encode_motion_ctrl_2(
            ctrl_mode=self.config.ctrl_mode,
            move_mode=self.config.move_mode,
            move_spd_rate_ctrl=self.config.move_spd_rate_ctrl,
            mit_mode=self.config.mit_mode,
            residence_time=self.config.residence_time,
            installation_pos=installation_pos,
        )
        self._send(ARM_MOTION_CTRL_2, payload)
        self.mode_initialized = True
        self.last_mode_ts = now

    def send_emergency_resume(self) -> None:
        self._send(ARM_MOTION_CTRL_1, encode_motion_ctrl_1(emergency_stop=0x02))

    def raw_to_user(self, joint_name: str, raw_value: int) -> float:
        scale = self.config.gripper_scale if joint_name == "gripper" else self.config.joint_scale
        return float(raw_value) * scale

    def user_to_raw(self, joint_name: str, value: float) -> int:
        scale = self.config.gripper_scale if joint_name == "gripper" else self.config.joint_scale
        if scale == 0:
            raise ValueError(f"Invalid zero scale for joint {joint_name}.")
        return int(round(float(value) / scale))

    def get_positions(self, *, poll: bool = True) -> dict[str, float]:
        if poll:
            self.poll()
        return {
            joint_name: self.raw_to_user(joint_name, raw_value)
            for joint_name, raw_value in self.joint_positions_raw.items()
        }

    def send_targets(self, targets: dict[str, float]) -> dict[str, float]:
        self.poll()
        self.ensure_can_command_mode()

        raw_targets = {
            joint_name: self.user_to_raw(joint_name, targets[joint_name]) for joint_name in DEFAULT_SONGLING_JOINT_NAMES
        }
        for _ in range(max(self.config.command_repeat, 1)):
            self._send(
                ARM_JOINT_CTRL_12,
                encode_joint_command_pair(ARM_JOINT_CTRL_12, raw_targets["joint_1"], raw_targets["joint_2"]),
            )
            self._send(
                ARM_JOINT_CTRL_34,
                encode_joint_command_pair(ARM_JOINT_CTRL_34, raw_targets["joint_3"], raw_targets["joint_4"]),
            )
            self._send(
                ARM_JOINT_CTRL_56,
                encode_joint_command_pair(ARM_JOINT_CTRL_56, raw_targets["joint_5"], raw_targets["joint_6"]),
            )
            self._send(
                ARM_GRIPPER_CTRL,
                encode_gripper_command(
                    position=raw_targets["gripper"],
                    effort=self.config.gripper_effort,
                    status_code=self.config.gripper_status_code,
                    set_zero=self.config.gripper_set_zero,
                ),
            )
            if self.config.command_interval_s > 0:
                time.sleep(self.config.command_interval_s)

        self.last_sent_ts = time.time()
        self.last_sent_raw.update(raw_targets)
        return {joint_name: self.raw_to_user(joint_name, raw_targets[joint_name]) for joint_name in raw_targets}


class SonglingFollower(Robot):
    """Experimental Songling follower using Piper-V2-like CAN frames."""

    config_class = SonglingFollowerConfig
    name = "songling_follower"

    def __init__(self, config: SonglingFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = SonglingCANPort(config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._warned_missing_obs = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in DEFAULT_SONGLING_JOINT_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam_key: (self.config.cameras[cam_key].height, self.config.cameras[cam_key].width, 3)
            for cam_key in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect(warmup=True)
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("Songling follower calibration is not implemented; assuming the arm is hardware-calibrated.")

    def configure(self) -> None:
        if not self.config.allow_unverified_commanding:
            logger.warning(
                "Songling real commanding is disabled. Set "
                "`robot.<arm>_config.allow_unverified_commanding=true` only after verifying "
                "CAN wiring, emergency stop access, and protocol behavior."
            )
            return

        if self.config.auto_enable_on_connect:
            self.bus.enable_motors(enable=True)
        if self.config.auto_configure_mode_on_connect:
            self.bus.ensure_can_command_mode(force=True)

    def setup_motors(self) -> None:
        raise NotImplementedError("Songling integrated chain motor setup is not exposed through this adapter.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        positions = self.bus.get_positions(poll=True)
        obs: dict[str, Any] = {f"{joint_name}.pos": float(positions[joint_name]) for joint_name in positions}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    def _clip_to_joint_limits(self, goal_pos: dict[str, float]) -> dict[str, float]:
        clipped = {}
        for joint_name, value in goal_pos.items():
            min_limit, max_limit = self.config.joint_limits[joint_name]
            clipped[joint_name] = max(min_limit, min(max_limit, value))
        return clipped

    def _apply_relative_limit(self, goal_pos: dict[str, float]) -> dict[str, float]:
        if self.config.max_relative_target is None:
            return goal_pos

        current = self.bus.get_positions(poll=True)
        if not all(self.bus.joint_position_seen.get(name, False) for name in DEFAULT_SONGLING_JOINT_NAMES):
            if not self._warned_missing_obs:
                logger.warning(
                    "Songling observation feedback is not fully populated yet; skipping relative target clamp once."
                )
                self._warned_missing_obs = True
            return goal_pos

        goal_present_pos = {joint_name: (goal_pos[joint_name], current[joint_name]) for joint_name in goal_pos}
        return ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.config.allow_unverified_commanding:
            raise RuntimeError(
                "Songling real commanding is disabled for safety. "
                "Set `allow_unverified_commanding=true` in the Songling robot config to opt in."
            )

        goal_pos = {
            key.removesuffix(".pos"): float(value) for key, value in action.items() if key.endswith(".pos")
        }
        missing = [joint_name for joint_name in DEFAULT_SONGLING_JOINT_NAMES if joint_name not in goal_pos]
        if missing:
            raise ValueError(f"Songling action is missing joints: {missing}")

        goal_pos = self._clip_to_joint_limits(goal_pos)
        goal_pos = self._apply_relative_limit(goal_pos)
        sent = self.bus.send_targets(goal_pos)
        return {f"{joint_name}.pos": float(value) for joint_name, value in sent.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        if self.config.allow_unverified_commanding and self.config.disable_torque_on_disconnect:
            try:
                self.bus.enable_motors(enable=False)
            except Exception:
                pass
        self.bus.disconnect()
