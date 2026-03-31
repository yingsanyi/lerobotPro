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
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from lerobot.utils.import_utils import _can_available
from lerobot.utils.songling_safety import (
    raise_songling_parameter_write_blocked,
    sanitize_songling_gripper_set_zero,
    sanitize_songling_installation_pos,
)

if _can_available:
    import can
else:
    can = None

REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_PIPER_SDK_ROOT = REPO_ROOT / "piper_sdk"
if LOCAL_PIPER_SDK_ROOT.is_dir() and str(LOCAL_PIPER_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PIPER_SDK_ROOT))

try:
    from piper_sdk import C_PiperInterface_V2
except Exception:
    C_PiperInterface_V2 = None

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
    ARM_MASTER_SLAVE_MODE_CONFIG,
    ARM_MOTION_CTRL_1,
    ARM_MOTION_CTRL_2,
    ARM_MOTOR_ENABLE_DISABLE_CONFIG,
    ARM_STATUS_FEEDBACK,
    DEFAULT_SONGLING_JOINT_NAMES,
    decode_gripper_command,
    decode_gripper_feedback,
    decode_high_speed_feedback,
    decode_joint_command,
    decode_joint_feedback,
    decode_low_speed_feedback,
    decode_status_feedback,
    encode_gripper_command,
    encode_joint_command_pair,
    encode_master_slave_config,
    encode_motion_ctrl_1,
    encode_motion_ctrl_2,
    encode_motor_enable_disable,
)

logger = logging.getLogger(__name__)
ENABLE_STATUS_RESEND_EVERY = 5


def _joint_rad_to_raw(value_rad: float) -> int:
    return int(round(math.degrees(float(value_rad)) * 1000.0))


def _joint_raw_to_rad(value_raw: int, scale: float = 1e-3) -> float:
    return math.radians(float(value_raw) * scale)


def _gripper_width_m_to_raw(width_m: float) -> int:
    return int(round(float(width_m) * 1_000_000.0))


def _gripper_raw_to_width_m(value_raw: int, scale: float = 1e-3) -> float:
    return float(value_raw) * scale * 1e-3


def _joint_limits_deg_to_rad(limits: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    converted: dict[str, list[float]] = {}
    for joint_name in DEFAULT_SONGLING_JOINT_NAMES[:-1]:
        lower_deg, upper_deg = limits[joint_name]
        converted_key = joint_name.replace("_", "")
        converted[converted_key] = [math.radians(float(lower_deg)), math.radians(float(upper_deg))]
    return converted


class _RateTracker:
    def __init__(self, *, window: int = 32):
        self.timestamps: deque[float] = deque(maxlen=max(window, 2))

    def update(self, timestamp: float) -> float:
        ts = float(timestamp)
        self.timestamps.append(ts)
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt <= 0:
            return 0.0
        return float(len(self.timestamps) - 1) / float(dt)


def _coerce_sdk_scalar(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raw_value = getattr(value, "value", None)
    if isinstance(raw_value, int):
        return raw_value
    try:
        return int(value)
    except Exception:
        pass
    text = str(value)
    for token in text.replace("(", " ").replace(")", " ").replace(":", " ").split():
        if token.startswith("0x"):
            try:
                return int(token, 16)
            except Exception:
                continue
        if token.lstrip("-").isdigit():
            try:
                return int(token)
            except Exception:
                continue
    return 0


def _wrapper_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    return float(getattr(value, "time_stamp", getattr(value, "timestamp", 0.0)) or 0.0)


def _wrapper_hz(value: Any) -> float:
    if value is None:
        return 0.0
    return float(getattr(value, "Hz", getattr(value, "hz", 0.0)) or 0.0)


def _wrapper_is_valid(value: Any) -> bool:
    return value is not None and (_wrapper_timestamp(value) > 0.0 or _wrapper_hz(value) > 0.0)


def _joint_raw_millirad_to_millideg(value_raw: int | float) -> int:
    return int(round(math.degrees(float(value_raw) * 1e-3) * 1000.0))


class SonglingPortBase:
    def __init__(self, config: SonglingFollowerConfig):
        self.config = config
        self._is_connected = False
        self.status: dict[str, int] = {}
        self.status_feedback_timestamp = 0.0
        self.status_feedback_hz = 0.0
        self.status_feedback_valid = False
        self.mode_command: dict[str, int] = {}
        self.mode_command_timestamp = 0.0
        self.mode_command_hz = 0.0
        self.mode_command_valid = False
        self.communication_status: dict[str, bool] = {}
        self.joint_positions_raw = {name: 0 for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.joint_position_seen = {name: False for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.commanded_positions_raw = {name: 0 for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.commanded_position_seen = {name: False for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.joint_feedback_timestamp = 0.0
        self.joint_feedback_hz = 0.0
        self.joint_feedback_valid = False
        self.gripper_feedback_timestamp = 0.0
        self.gripper_feedback_hz = 0.0
        self.gripper_feedback_valid = False
        self.command_feedback_timestamp = 0.0
        self.command_feedback_hz = 0.0
        self.command_feedback_valid = False
        self.high_speed_feedback: dict[str, dict[str, Any]] = {}
        self.low_speed_feedback: dict[str, dict[str, Any]] = {}
        self.last_sent_raw = {name: 0 for name in DEFAULT_SONGLING_JOINT_NAMES}
        self.last_sent_ts = 0.0
        self.last_mode_ts = 0.0
        self.mode_initialized = False
        self._joint_rate = _RateTracker()
        self._gripper_rate = _RateTracker()
        self._command_rate = _RateTracker()
        self._status_rate = _RateTracker()
        self._mode_rate = _RateTracker()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def poll(self, *, max_msgs: int | None = None) -> None:
        raise NotImplementedError

    def raw_to_user(self, joint_name: str, raw_value: int) -> float:
        scale = self.config.gripper_scale if joint_name == "gripper" else self.config.joint_scale
        zero_offset = float(self.config.joint_zero_offsets.get(joint_name, 0.0))
        return (float(raw_value) * scale) - zero_offset

    def user_to_raw(self, joint_name: str, value: float) -> int:
        scale = self.config.gripper_scale if joint_name == "gripper" else self.config.joint_scale
        if scale == 0:
            raise ValueError(f"Invalid zero scale for joint {joint_name}.")
        zero_offset = float(self.config.joint_zero_offsets.get(joint_name, 0.0))
        return int(round((float(value) + zero_offset) / scale))

    def get_positions(self, *, poll: bool = True) -> dict[str, float]:
        if poll:
            self.poll()
        return {
            joint_name: self.raw_to_user(joint_name, raw_value)
            for joint_name, raw_value in self.joint_positions_raw.items()
        }

    def get_commanded_positions(self, *, poll: bool = True) -> dict[str, float]:
        if poll:
            self.poll()
        return {
            joint_name: self.raw_to_user(joint_name, raw_value)
            for joint_name, raw_value in self.commanded_positions_raw.items()
        }

    def has_reliable_position_feedback(self) -> bool:
        return bool(
            all(self.joint_position_seen.get(joint_name, False) for joint_name in DEFAULT_SONGLING_JOINT_NAMES[:-1])
        )

    def send_emergency_resume(self) -> None:
        self._send_emergency_resume_impl()

    def _send_emergency_resume_impl(self) -> None:
        raise NotImplementedError

    def enable_motors(self, *, enable: bool) -> None:
        raise NotImplementedError

    def get_driver_enable_status(self) -> dict[str, bool | None]:
        self.poll()
        return {
            joint_name: (
                None
                if joint_name not in self.low_speed_feedback
                else self.low_speed_feedback[joint_name].get("driver_enable_status")
            )
            for joint_name in DEFAULT_SONGLING_JOINT_NAMES[:-1]
        }

    def wait_for_driver_enable_status(self, *, enabled: bool) -> bool:
        expected = bool(enabled)
        retry_count = max(int(self.config.enable_retry_count), 1)
        retry_interval_s = max(float(self.config.enable_retry_interval_s), 0.0)

        self.enable_motors(enable=enabled)
        for attempt in range(retry_count):
            self.poll(max_msgs=self.config.poll_max_msgs)
            statuses = self.get_driver_enable_status()
            known = [state for state in statuses.values() if state is not None]
            if known and all(bool(state) is expected for state in known):
                return True
            if (
                attempt + 1 < retry_count
                and (attempt + 1) % ENABLE_STATUS_RESEND_EVERY == 0
            ):
                self.enable_motors(enable=enabled)
            if retry_interval_s > 0:
                time.sleep(retry_interval_s)
        return False

    def installation_pos(self) -> int:
        return sanitize_songling_installation_pos(
            logger=logger,
            requested=getattr(self.config, "installation_pos", None),
            entrypoint=f"{self.__class__.__name__}.MotionCtrl_2",
        )

    def ensure_can_command_mode(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and self.mode_initialized and (now - self.last_mode_ts) < max(self.config.mode_keepalive_s, 0.0):
            return
        self._send_motion_ctrl_2_impl()
        self.mode_initialized = True
        self.last_mode_ts = now

    def _send_motion_ctrl_2_impl(self) -> None:
        raise NotImplementedError

    def configure_leader_follower(
        self,
        *,
        role: str | None = None,
        feedback_offset: int | None = None,
        ctrl_offset: int | None = None,
        linkage_offset: int | None = None,
    ) -> None:
        requested_role = role if role is not None else self.config.leader_follower_role
        raise_songling_parameter_write_blocked(
            "leader/follower role reconfiguration",
            details=(
                "MasterSlaveConfig downlinks are disabled. "
                f"Requested role={requested_role!r}, feedback_offset={feedback_offset!r}, "
                f"ctrl_offset={ctrl_offset!r}, linkage_offset={linkage_offset!r}."
            ),
        )

    def _configure_leader_follower_impl(
        self, *, linkage_config: int, feedback_offset: int, ctrl_offset: int, linkage_offset: int
    ) -> None:
        raise NotImplementedError

    def send_targets(self, targets: dict[str, float]) -> dict[str, float]:
        self.poll()
        raw_targets = {
            joint_name: self.user_to_raw(joint_name, targets[joint_name]) for joint_name in DEFAULT_SONGLING_JOINT_NAMES
        }
        for _ in range(max(int(self.config.command_repeat), 1)):
            self._send_joint_targets_raw(raw_targets)
            self._send_gripper_target_raw(raw_targets)
            self.ensure_can_command_mode(force=True)
            if self.config.command_interval_s > 0:
                time.sleep(self.config.command_interval_s)

        now = time.time()
        self.last_sent_ts = now
        self.last_sent_raw.update(raw_targets)
        self.commanded_positions_raw.update(raw_targets)
        self.commanded_position_seen.update({joint_name: True for joint_name in raw_targets})
        self.command_feedback_timestamp = now
        self.command_feedback_hz = self._command_rate.update(now)
        self.command_feedback_valid = True
        return {joint_name: self.raw_to_user(joint_name, raw_targets[joint_name]) for joint_name in raw_targets}

    def _send_joint_targets_raw(self, raw_targets: dict[str, int]) -> None:
        raise NotImplementedError

    def _send_gripper_target_raw(self, raw_targets: dict[str, int]) -> None:
        raise NotImplementedError

    def _mark_joint_feedback(self, timestamp: float) -> None:
        self.joint_feedback_timestamp = float(timestamp)
        self.joint_feedback_hz = self._joint_rate.update(timestamp)
        self.joint_feedback_valid = True

    def _mark_gripper_feedback(self, timestamp: float) -> None:
        self.gripper_feedback_timestamp = float(timestamp)
        self.gripper_feedback_hz = self._gripper_rate.update(timestamp)
        self.gripper_feedback_valid = True

    def _mark_command_feedback(self, timestamp: float) -> None:
        self.command_feedback_timestamp = float(timestamp)
        self.command_feedback_hz = self._command_rate.update(timestamp)
        self.command_feedback_valid = True

    def _mark_status_feedback(self, timestamp: float) -> None:
        self.status_feedback_timestamp = float(timestamp)
        self.status_feedback_hz = self._status_rate.update(timestamp)
        self.status_feedback_valid = True

    def _mark_mode_command(self, timestamp: float) -> None:
        self.mode_command_timestamp = float(timestamp)
        self.mode_command_hz = self._mode_rate.update(timestamp)
        self.mode_command_valid = True


class SonglingCANPort(SonglingPortBase):
    def __init__(self, config: SonglingFollowerConfig):
        super().__init__(config)
        self.bus: can.BusABC | None = None  # type: ignore[assignment]

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.bus is not None

    def connect(self) -> None:
        if can is None:
            raise ModuleNotFoundError("python-can is required for Songling CAN control.")

        kwargs: dict[str, Any] = {
            "channel": self.config.channel,
            "interface": self.config.interface,
            "bitrate": self.config.bitrate,
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
            timestamp = float(getattr(msg, "timestamp", 0.0) or time.time())
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
                    self._mark_status_feedback(timestamp)
                elif ARM_JOINT_FEEDBACK_12 <= arbitration_id <= ARM_JOINT_FEEDBACK_56:
                    for name, raw in decode_joint_feedback(arbitration_id, payload).items():
                        self.joint_positions_raw[name] = raw
                        self.joint_position_seen[name] = True
                    self._mark_joint_feedback(timestamp)
                elif arbitration_id == ARM_GRIPPER_FEEDBACK:
                    gripper = decode_gripper_feedback(payload)
                    self.joint_positions_raw["gripper"] = gripper.position
                    self.joint_position_seen["gripper"] = True
                    self.high_speed_feedback["gripper"] = {
                        "effort": gripper.effort,
                        "status_code": gripper.status_code,
                    }
                    self._mark_gripper_feedback(timestamp)
                elif ARM_INFO_HIGH_SPD_FEEDBACK_1 <= arbitration_id <= ARM_INFO_HIGH_SPD_FEEDBACK_6:
                    feedback = decode_high_speed_feedback(arbitration_id, payload)
                    self.high_speed_feedback[feedback.joint_name] = {
                        "motor_speed": feedback.motor_speed,
                        "current": feedback.current,
                        "position": feedback.position,
                    }
                elif ARM_INFO_LOW_SPD_FEEDBACK_1 <= arbitration_id <= ARM_INFO_LOW_SPD_FEEDBACK_6:
                    feedback = decode_low_speed_feedback(arbitration_id, payload)
                    foc_status_code = feedback.foc_status_code
                    self.low_speed_feedback[feedback.joint_name] = {
                        "voltage": feedback.voltage,
                        "foc_temp": feedback.foc_temp,
                        "motor_temp": feedback.motor_temp,
                        "foc_status_code": foc_status_code,
                        "driver_enable_status": not bool(foc_status_code & (1 << 6)),
                        "driver_error_status": bool(foc_status_code & (1 << 5)),
                        "collision_status": bool(foc_status_code & (1 << 4)),
                        "stall_status": bool(foc_status_code & (1 << 7)),
                        "bus_current": feedback.bus_current,
                    }
                elif ARM_JOINT_CTRL_12 <= arbitration_id <= ARM_JOINT_CTRL_56:
                    for name, raw in decode_joint_command(arbitration_id, payload).items():
                        self.commanded_positions_raw[name] = raw
                        self.commanded_position_seen[name] = True
                    self._mark_command_feedback(timestamp)
                elif arbitration_id == ARM_GRIPPER_CTRL:
                    gripper = decode_gripper_command(payload)
                    self.commanded_positions_raw["gripper"] = gripper.position
                    self.commanded_position_seen["gripper"] = True
                    self._mark_command_feedback(timestamp)
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

    def _enable_command_targets(self) -> list[int]:
        return [0xFF, 1, 2, 3, 4, 5, 6, 7]

    def enable_motors(self, *, enable: bool) -> None:
        targets = [0xFF] if not enable else self._enable_command_targets()
        enable_flag = 0x02 if enable else 0x01
        for motor_num in targets:
            payload = encode_motor_enable_disable(motor_num=motor_num, enable_flag=enable_flag)
            self._send(ARM_MOTOR_ENABLE_DISABLE_CONFIG, payload)

    def _send_emergency_resume_impl(self) -> None:
        self._send(ARM_MOTION_CTRL_1, encode_motion_ctrl_1(emergency_stop=0x02))

    def _send_motion_ctrl_2_impl(self) -> None:
        payload = encode_motion_ctrl_2(
            ctrl_mode=self.config.ctrl_mode,
            move_mode=self.config.move_mode,
            move_spd_rate_ctrl=self.config.speed_percent,
            mit_mode=self.config.mit_mode,
            residence_time=self.config.residence_time,
            installation_pos=self.installation_pos(),
        )
        self._send(ARM_MOTION_CTRL_2, payload)

    def _configure_leader_follower_impl(
        self, *, linkage_config: int, feedback_offset: int, ctrl_offset: int, linkage_offset: int
    ) -> None:
        payload = encode_master_slave_config(
            linkage_config=linkage_config,
            feedback_offset=feedback_offset,
            ctrl_offset=ctrl_offset,
            linkage_offset=linkage_offset,
        )
        self._send(ARM_MASTER_SLAVE_MODE_CONFIG, payload)

    def _send_joint_targets_raw(self, raw_targets: dict[str, int]) -> None:
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

    def _send_gripper_target_raw(self, raw_targets: dict[str, int]) -> None:
        self._send(
            ARM_GRIPPER_CTRL,
            encode_gripper_command(
                position=raw_targets["gripper"],
                effort=int(round(float(self.config.gripper_force) * 1000.0)),
                status_code=self.config.gripper_status_code,
                set_zero=sanitize_songling_gripper_set_zero(
                    logger=logger,
                    requested=self.config.gripper_set_zero,
                    entrypoint=f"{self.__class__.__name__}.GripperCtrl",
                ),
            ),
        )


class PiperSDKPort(SonglingPortBase):
    def __init__(self, config: SonglingFollowerConfig):
        if C_PiperInterface_V2 is None:
            raise ModuleNotFoundError(
                "piper_sdk is required for the Songling `piper_sdk` backend. "
                "Install local `piper_sdk` runtime dependencies or switch `transport_backend` back to `raw_can`."
            )
        super().__init__(config)
        self._piper_judge_flag = bool(getattr(config, "piper_judge_flag", False))
        self.interface = C_PiperInterface_V2(config.channel, self._piper_judge_flag, False)

    def connect(self) -> None:
        if self.config.use_can_fd:
            logger.warning("Piper SDK backend does not expose CAN-FD configuration; continuing with classic CAN.")
        if not bool(getattr(self.interface, "get_connect_status", lambda: False)()):
            self.interface.CreateCanBus(
                self.config.channel,
                bustype=self.config.interface,
                expected_bitrate=self.config.bitrate,
                judge_flag=self._piper_judge_flag,
            )
            self.interface.ConnectPort(can_init=False, piper_init=True, start_thread=True)
        self._is_connected = bool(getattr(self.interface, "get_connect_status", lambda: True)())
        time.sleep(0.1)
        self.poll()

    def disconnect(self) -> None:
        try:
            disconnect = getattr(self.interface, "DisconnectPort", None)
            if callable(disconnect):
                disconnect()
        finally:
            self._is_connected = False

    def poll(self, *, max_msgs: int | None = None) -> None:
        _ = max_msgs
        if not self._is_connected:
            return

        arm_status_wrapper = self.interface.GetArmStatus()
        self.status_feedback_valid = _wrapper_is_valid(arm_status_wrapper)
        if self.status_feedback_valid:
            arm_status = getattr(arm_status_wrapper, "arm_status", None)
            arm_status_timestamp = _wrapper_timestamp(arm_status_wrapper)
            arm_status_hz = _wrapper_hz(arm_status_wrapper)
            self.status = {
                "ctrl_mode": _coerce_sdk_scalar(getattr(arm_status, "ctrl_mode", 0)),
                "arm_status": _coerce_sdk_scalar(getattr(arm_status, "arm_status", 0)),
                "mode_feed": _coerce_sdk_scalar(getattr(arm_status, "mode_feed", 0)),
                "teach_status": _coerce_sdk_scalar(getattr(arm_status, "teach_status", 0)),
                "motion_status": _coerce_sdk_scalar(getattr(arm_status, "motion_status", 0)),
                "trajectory_num": _coerce_sdk_scalar(getattr(arm_status, "trajectory_num", 0)),
                "err_code": _coerce_sdk_scalar(getattr(arm_status, "err_code", 0)),
            }
            self.status_feedback_timestamp = arm_status_timestamp
            self.status_feedback_hz = (
                arm_status_hz if arm_status_hz > 0.0 else self._status_rate.update(arm_status_timestamp)
            )
            err_status = getattr(arm_status, "err_status", None)
            self.communication_status = {
                joint_name: bool(getattr(err_status, f"communication_status_{joint_name}", False)) for joint_name in DEFAULT_SONGLING_JOINT_NAMES[:-1]
            }
        else:
            self.status = {}
            self.status_feedback_timestamp = 0.0
            self.status_feedback_hz = 0.0
            self.communication_status = {}

        joint_wrapper = self.interface.GetArmJointMsgs()
        self.joint_feedback_valid = _wrapper_is_valid(joint_wrapper)
        if self.joint_feedback_valid:
            joint_timestamp = _wrapper_timestamp(joint_wrapper)
            joint_hz = _wrapper_hz(joint_wrapper)
            joint_state = getattr(joint_wrapper, "joint_state", None)
            for idx, joint_name in enumerate(DEFAULT_SONGLING_JOINT_NAMES[:-1], start=1):
                raw_value = _coerce_sdk_scalar(getattr(joint_state, f"joint_{idx}", 0))
                self.joint_positions_raw[joint_name] = raw_value
                self.joint_position_seen[joint_name] = True
            self.joint_feedback_timestamp = joint_timestamp
            self.joint_feedback_hz = joint_hz if joint_hz > 0.0 else self._joint_rate.update(joint_timestamp)

        gripper_wrapper = self.interface.GetArmGripperMsgs()
        self.gripper_feedback_valid = _wrapper_is_valid(gripper_wrapper)
        if self.gripper_feedback_valid:
            gripper = getattr(gripper_wrapper, "gripper_state", None)
            gripper_timestamp = _wrapper_timestamp(gripper_wrapper)
            gripper_hz = _wrapper_hz(gripper_wrapper)
            gripper_raw = _coerce_sdk_scalar(getattr(gripper, "grippers_angle", 0))
            self.joint_positions_raw["gripper"] = gripper_raw
            self.joint_position_seen["gripper"] = True
            self.gripper_feedback_timestamp = gripper_timestamp
            self.gripper_feedback_hz = gripper_hz if gripper_hz > 0.0 else self._gripper_rate.update(gripper_timestamp)
            self.high_speed_feedback["gripper"] = {
                "effort": _coerce_sdk_scalar(getattr(gripper, "grippers_effort", 0)),
                "status_code": _coerce_sdk_scalar(getattr(gripper, "status_code", 0)),
            }

        mode_wrapper = self.interface.GetArmCtrlCode151()
        if _wrapper_is_valid(mode_wrapper):
            mode = getattr(mode_wrapper, "ctrl_151", None)
            mode_timestamp = _wrapper_timestamp(mode_wrapper)
            mode_hz = _wrapper_hz(mode_wrapper)
            self.mode_command = {
                "ctrl_mode": _coerce_sdk_scalar(getattr(mode, "ctrl_mode", 0)),
                "move_mode": _coerce_sdk_scalar(getattr(mode, "move_mode", 0)),
                "move_spd_rate_ctrl": _coerce_sdk_scalar(getattr(mode, "move_spd_rate_ctrl", 0)),
                "mit_mode": _coerce_sdk_scalar(getattr(mode, "mit_mode", 0)),
                "residence_time": _coerce_sdk_scalar(getattr(mode, "residence_time", 0)),
                "installation_pos": _coerce_sdk_scalar(getattr(mode, "installation_pos", 0)),
            }
            self.mode_command_timestamp = mode_timestamp
            self.mode_command_hz = mode_hz if mode_hz > 0.0 else self._mode_rate.update(mode_timestamp)
            self.mode_command_valid = True

        joint_ctrl_wrapper = self.interface.GetArmJointCtrl()
        if _wrapper_is_valid(joint_ctrl_wrapper):
            joint_ctrl = getattr(joint_ctrl_wrapper, "joint_ctrl", None)
            joint_ctrl_timestamp = _wrapper_timestamp(joint_ctrl_wrapper)
            joint_ctrl_hz = _wrapper_hz(joint_ctrl_wrapper)
            for idx, joint_name in enumerate(DEFAULT_SONGLING_JOINT_NAMES[:-1], start=1):
                self.commanded_positions_raw[joint_name] = _coerce_sdk_scalar(getattr(joint_ctrl, f"joint_{idx}", 0))
                self.commanded_position_seen[joint_name] = True
            self.command_feedback_valid = True
            if joint_ctrl_timestamp > 0.0:
                self.command_feedback_timestamp = max(self.command_feedback_timestamp, joint_ctrl_timestamp)
            if joint_ctrl_hz > 0.0:
                self.command_feedback_hz = max(self.command_feedback_hz, joint_ctrl_hz)
            elif joint_ctrl_timestamp > 0.0:
                self.command_feedback_hz = max(self.command_feedback_hz, self._command_rate.update(joint_ctrl_timestamp))

        gripper_ctrl_wrapper = self.interface.GetArmGripperCtrl()
        if _wrapper_is_valid(gripper_ctrl_wrapper):
            gripper_ctrl = getattr(gripper_ctrl_wrapper, "gripper_ctrl", None)
            gripper_ctrl_timestamp = _wrapper_timestamp(gripper_ctrl_wrapper)
            gripper_ctrl_hz = _wrapper_hz(gripper_ctrl_wrapper)
            gripper_raw = _coerce_sdk_scalar(getattr(gripper_ctrl, "grippers_angle", 0))
            # Prefer the newer gripper source for UI/current-position display.
            # Some integrated chains expose only the control-state echo reliably.
            use_ctrl_for_position = (
                (not self.gripper_feedback_valid)
                or gripper_ctrl_timestamp >= float(getattr(self, "gripper_feedback_timestamp", 0.0))
            )
            if use_ctrl_for_position:
                self.gripper_feedback_valid = True
                self.joint_positions_raw["gripper"] = gripper_raw
                self.joint_position_seen["gripper"] = True
                self.gripper_feedback_timestamp = gripper_ctrl_timestamp
                self.gripper_feedback_hz = (
                    gripper_ctrl_hz if gripper_ctrl_hz > 0.0 else self._gripper_rate.update(gripper_ctrl_timestamp)
                )
                self.high_speed_feedback["gripper"] = {
                    "effort": _coerce_sdk_scalar(getattr(gripper_ctrl, "grippers_effort", 0)),
                    "status_code": _coerce_sdk_scalar(getattr(gripper_ctrl, "status_code", 0)),
                }
            self.commanded_positions_raw["gripper"] = gripper_raw
            self.commanded_position_seen["gripper"] = True
            self.command_feedback_valid = True
            if gripper_ctrl_timestamp > 0.0:
                self.command_feedback_timestamp = max(self.command_feedback_timestamp, gripper_ctrl_timestamp)
            if gripper_ctrl_hz > 0.0:
                self.command_feedback_hz = max(self.command_feedback_hz, gripper_ctrl_hz)
            elif gripper_ctrl_timestamp > 0.0:
                self.command_feedback_hz = max(
                    self.command_feedback_hz,
                    self._command_rate.update(gripper_ctrl_timestamp),
                )

        high_spd_wrapper = self.interface.GetMotorStates()
        low_spd_wrapper = self.interface.GetDriverStates()
        fallback_joint_timestamp = 0.0
        fallback_joint_hz = 0.0
        for idx, joint_name in enumerate(DEFAULT_SONGLING_JOINT_NAMES[:-1], start=1):
            motor_info = getattr(high_spd_wrapper, f"motor_{idx}", None)
            if motor_info is not None and (
                _wrapper_is_valid(high_spd_wrapper) or _coerce_sdk_scalar(getattr(motor_info, "can_id", 0)) != 0
            ):
                raw_position = _joint_raw_millirad_to_millideg(_coerce_sdk_scalar(getattr(motor_info, "pos", 0)))
                fallback_joint_timestamp = max(fallback_joint_timestamp, _wrapper_timestamp(high_spd_wrapper))
                fallback_joint_hz = max(fallback_joint_hz, _wrapper_hz(high_spd_wrapper))
                self.high_speed_feedback[joint_name] = {
                    "motor_speed": _joint_raw_millirad_to_millideg(_coerce_sdk_scalar(getattr(motor_info, "motor_speed", 0))),
                    "current": _coerce_sdk_scalar(getattr(motor_info, "current", 0)),
                    "position": raw_position,
                }
                if not self.joint_feedback_valid:
                    self.joint_positions_raw[joint_name] = raw_position
                    self.joint_position_seen[joint_name] = True

            driver_info = getattr(low_spd_wrapper, f"motor_{idx}", None)
            if driver_info is not None and (
                _wrapper_is_valid(low_spd_wrapper) or _coerce_sdk_scalar(getattr(driver_info, "can_id", 0)) != 0
            ):
                foc_status = getattr(driver_info, "foc_status", None)
                self.low_speed_feedback[joint_name] = {
                    "voltage": _coerce_sdk_scalar(getattr(driver_info, "vol", 0)),
                    "foc_temp": _coerce_sdk_scalar(getattr(driver_info, "foc_temp", 0)),
                    "motor_temp": _coerce_sdk_scalar(getattr(driver_info, "motor_temp", 0)),
                    "foc_status_code": _coerce_sdk_scalar(getattr(driver_info, "foc_status_code", 0)),
                    "driver_enable_status": bool(getattr(foc_status, "driver_enable_status", False)),
                    "driver_error_status": bool(getattr(foc_status, "driver_error_status", False)),
                    "collision_status": bool(getattr(foc_status, "collision_status", False)),
                    "stall_status": bool(getattr(foc_status, "stall_status", False)),
                    "bus_current": _coerce_sdk_scalar(getattr(driver_info, "bus_current", 0)),
                }

        if not self.joint_feedback_valid and all(
            self.joint_position_seen.get(joint_name, False) for joint_name in DEFAULT_SONGLING_JOINT_NAMES[:-1]
        ):
            self.joint_feedback_valid = True
            self.joint_feedback_timestamp = fallback_joint_timestamp
            self.joint_feedback_hz = (
                fallback_joint_hz if fallback_joint_hz > 0.0 else self._joint_rate.update(time.time())
            )

    def enable_motors(self, *, enable: bool) -> None:
        if enable:
            self.interface.EnableArm(7, 0x02)
            self.interface.GripperCtrl(
                self.commanded_positions_raw.get("gripper", 0),
                int(round(float(self.config.gripper_force) * 1000.0)),
                0x01 if self.config.gripper_status_code == 0x00 else int(self.config.gripper_status_code),
                0x00,
            )
        else:
            self.interface.DisableArm(7, 0x01)
            self.interface.GripperCtrl(
                self.commanded_positions_raw.get("gripper", 0),
                int(round(float(self.config.gripper_force) * 1000.0)),
                0x00,
                0x00,
            )

    def _send_emergency_resume_impl(self) -> None:
        self.interface.EmergencyStop(0x02)

    def send_targets(self, targets: dict[str, float]) -> dict[str, float]:
        raw_targets = {
            joint_name: self.user_to_raw(joint_name, targets[joint_name]) for joint_name in DEFAULT_SONGLING_JOINT_NAMES
        }
        self.ensure_can_command_mode(force=False)
        for _ in range(max(int(self.config.command_repeat), 1)):
            self._send_joint_targets_raw(raw_targets)
            self._send_gripper_target_raw(raw_targets)
            if self.config.command_interval_s > 0:
                time.sleep(self.config.command_interval_s)

        now = time.time()
        self.last_sent_ts = now
        self.last_sent_raw.update(raw_targets)
        self.commanded_positions_raw.update(raw_targets)
        self.commanded_position_seen.update({joint_name: True for joint_name in raw_targets})
        self.command_feedback_timestamp = now
        self.command_feedback_hz = self._command_rate.update(now)
        self.command_feedback_valid = True
        return {joint_name: self.raw_to_user(joint_name, raw_targets[joint_name]) for joint_name in raw_targets}

    def _send_motion_ctrl_2_impl(self) -> None:
        self.interface.MotionCtrl_2(
            int(self.config.ctrl_mode),
            int(self.config.move_mode),
            int(self.config.speed_percent),
            int(self.config.mit_mode),
            int(self.config.residence_time),
            self.installation_pos(),
        )
        now = time.time()
        self.mode_command = {
            "ctrl_mode": int(self.config.ctrl_mode),
            "move_mode": int(self.config.move_mode),
            "move_spd_rate_ctrl": int(self.config.speed_percent),
            "mit_mode": int(self.config.mit_mode),
            "residence_time": int(self.config.residence_time),
            "installation_pos": self.installation_pos(),
        }
        self.mode_command_timestamp = now
        self.mode_command_hz = self._mode_rate.update(now)
        self.mode_command_valid = True

    def _configure_leader_follower_impl(
        self, *, linkage_config: int, feedback_offset: int, ctrl_offset: int, linkage_offset: int
    ) -> None:
        self.interface.MasterSlaveConfig(linkage_config, feedback_offset, ctrl_offset, linkage_offset)

    def _send_joint_targets_raw(self, raw_targets: dict[str, int]) -> None:
        self.interface.JointCtrl(
            raw_targets["joint_1"],
            raw_targets["joint_2"],
            raw_targets["joint_3"],
            raw_targets["joint_4"],
            raw_targets["joint_5"],
            raw_targets["joint_6"],
        )

    def _send_gripper_target_raw(self, raw_targets: dict[str, int]) -> None:
        self.interface.GripperCtrl(
            raw_targets["gripper"],
            int(round(float(self.config.gripper_force) * 1000.0)),
            int(self.config.gripper_status_code),
            sanitize_songling_gripper_set_zero(
                logger=logger,
                requested=self.config.gripper_set_zero,
                entrypoint=f"{self.__class__.__name__}.GripperCtrl",
            ),
        )

def make_songling_port(config: SonglingFollowerConfig) -> SonglingPortBase:
    backend = (config.transport_backend or "raw_can").strip().lower()
    if backend in {"raw", "raw_can", "socketcan"}:
        return SonglingCANPort(config)
    if backend in {"piper_sdk", "pipersdk", "piper"}:
        return PiperSDKPort(config)
    raise ValueError(
        f"Unsupported Songling transport backend {config.transport_backend!r}. "
        "Expected one of: raw_can, piper_sdk."
    )
