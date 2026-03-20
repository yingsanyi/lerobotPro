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

"""Minimal Piper-V2-like Songling CAN helpers.

This module intentionally implements only the subset needed by the current
Songling real-control bridge:

- status and joint/gripper feedback decoding
- joint/gripper control frame encoding
- control-mode and motor enable/disable frame encoding

The mapping is based on the local `piper_sdk` reference bundled in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_SONGLING_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
]

# Feedback IDs
ARM_STATUS_FEEDBACK = 0x2A1
ARM_END_POSE_FEEDBACK_1 = 0x2A2
ARM_END_POSE_FEEDBACK_2 = 0x2A3
ARM_END_POSE_FEEDBACK_3 = 0x2A4
ARM_JOINT_FEEDBACK_12 = 0x2A5
ARM_JOINT_FEEDBACK_34 = 0x2A6
ARM_JOINT_FEEDBACK_56 = 0x2A7
ARM_GRIPPER_FEEDBACK = 0x2A8
ARM_INFO_HIGH_SPD_FEEDBACK_1 = 0x251
ARM_INFO_HIGH_SPD_FEEDBACK_2 = 0x252
ARM_INFO_HIGH_SPD_FEEDBACK_3 = 0x253
ARM_INFO_HIGH_SPD_FEEDBACK_4 = 0x254
ARM_INFO_HIGH_SPD_FEEDBACK_5 = 0x255
ARM_INFO_HIGH_SPD_FEEDBACK_6 = 0x256
ARM_INFO_LOW_SPD_FEEDBACK_1 = 0x261
ARM_INFO_LOW_SPD_FEEDBACK_2 = 0x262
ARM_INFO_LOW_SPD_FEEDBACK_3 = 0x263
ARM_INFO_LOW_SPD_FEEDBACK_4 = 0x264
ARM_INFO_LOW_SPD_FEEDBACK_5 = 0x265
ARM_INFO_LOW_SPD_FEEDBACK_6 = 0x266

# Control IDs
ARM_MOTION_CTRL_1 = 0x150
ARM_MOTION_CTRL_2 = 0x151
ARM_JOINT_CTRL_12 = 0x155
ARM_JOINT_CTRL_34 = 0x156
ARM_JOINT_CTRL_56 = 0x157
ARM_GRIPPER_CTRL = 0x159
ARM_MOTOR_ENABLE_DISABLE_CONFIG = 0x471

JOINT_FEEDBACK_IDS = {
    ARM_JOINT_FEEDBACK_12: ("joint_1", "joint_2"),
    ARM_JOINT_FEEDBACK_34: ("joint_3", "joint_4"),
    ARM_JOINT_FEEDBACK_56: ("joint_5", "joint_6"),
}

JOINT_COMMAND_IDS = {
    ARM_JOINT_CTRL_12: ("joint_1", "joint_2"),
    ARM_JOINT_CTRL_34: ("joint_3", "joint_4"),
    ARM_JOINT_CTRL_56: ("joint_5", "joint_6"),
}

HIGH_SPEED_FEEDBACK_IDS = {
    ARM_INFO_HIGH_SPD_FEEDBACK_1: "joint_1",
    ARM_INFO_HIGH_SPD_FEEDBACK_2: "joint_2",
    ARM_INFO_HIGH_SPD_FEEDBACK_3: "joint_3",
    ARM_INFO_HIGH_SPD_FEEDBACK_4: "joint_4",
    ARM_INFO_HIGH_SPD_FEEDBACK_5: "joint_5",
    ARM_INFO_HIGH_SPD_FEEDBACK_6: "joint_6",
}

LOW_SPEED_FEEDBACK_IDS = {
    ARM_INFO_LOW_SPD_FEEDBACK_1: "joint_1",
    ARM_INFO_LOW_SPD_FEEDBACK_2: "joint_2",
    ARM_INFO_LOW_SPD_FEEDBACK_3: "joint_3",
    ARM_INFO_LOW_SPD_FEEDBACK_4: "joint_4",
    ARM_INFO_LOW_SPD_FEEDBACK_5: "joint_5",
    ARM_INFO_LOW_SPD_FEEDBACK_6: "joint_6",
}


def _int_from_be(payload: bytes, start: int, end: int, *, signed: bool) -> int:
    return int.from_bytes(payload[start:end], byteorder="big", signed=signed)


def _int_to_be(value: int, length: int, *, signed: bool) -> bytes:
    return int(value).to_bytes(length, byteorder="big", signed=signed)


@dataclass(frozen=True)
class SonglingStatusFeedback:
    ctrl_mode: int
    arm_status: int
    mode_feed: int
    teach_status: int
    motion_status: int
    trajectory_num: int
    err_code: int


@dataclass(frozen=True)
class SonglingHighSpeedFeedback:
    joint_name: str
    motor_speed: int
    current: int
    position: int


@dataclass(frozen=True)
class SonglingLowSpeedFeedback:
    joint_name: str
    voltage: int
    foc_temp: int
    motor_temp: int
    foc_status_code: int
    bus_current: int


@dataclass(frozen=True)
class SonglingGripperFeedback:
    position: int
    effort: int
    status_code: int


def decode_status_feedback(payload: bytes) -> SonglingStatusFeedback:
    if len(payload) < 8:
        raise ValueError(f"Expected 8-byte status payload, got {len(payload)}.")
    return SonglingStatusFeedback(
        ctrl_mode=payload[0],
        arm_status=payload[1],
        mode_feed=payload[2],
        teach_status=payload[3],
        motion_status=payload[4],
        trajectory_num=payload[5],
        err_code=_int_from_be(payload, 6, 8, signed=False),
    )


def decode_joint_feedback(arbitration_id: int, payload: bytes) -> dict[str, int]:
    joint_names = JOINT_FEEDBACK_IDS.get(arbitration_id)
    if joint_names is None:
        raise ValueError(f"Unsupported joint feedback ID: 0x{arbitration_id:X}")
    if len(payload) < 8:
        raise ValueError(f"Expected 8-byte joint feedback payload, got {len(payload)}.")
    return {
        joint_names[0]: _int_from_be(payload, 0, 4, signed=True),
        joint_names[1]: _int_from_be(payload, 4, 8, signed=True),
    }


def decode_gripper_feedback(payload: bytes) -> SonglingGripperFeedback:
    if len(payload) < 8:
        raise ValueError(f"Expected 8-byte gripper feedback payload, got {len(payload)}.")
    return SonglingGripperFeedback(
        position=_int_from_be(payload, 0, 4, signed=True),
        effort=_int_from_be(payload, 4, 6, signed=True),
        status_code=payload[6],
    )


def decode_high_speed_feedback(arbitration_id: int, payload: bytes) -> SonglingHighSpeedFeedback:
    joint_name = HIGH_SPEED_FEEDBACK_IDS.get(arbitration_id)
    if joint_name is None:
        raise ValueError(f"Unsupported high-speed feedback ID: 0x{arbitration_id:X}")
    if len(payload) < 8:
        raise ValueError(f"Expected 8-byte high-speed feedback payload, got {len(payload)}.")
    return SonglingHighSpeedFeedback(
        joint_name=joint_name,
        motor_speed=_int_from_be(payload, 0, 2, signed=True),
        current=_int_from_be(payload, 2, 4, signed=False),
        position=_int_from_be(payload, 4, 8, signed=True),
    )


def decode_low_speed_feedback(arbitration_id: int, payload: bytes) -> SonglingLowSpeedFeedback:
    joint_name = LOW_SPEED_FEEDBACK_IDS.get(arbitration_id)
    if joint_name is None:
        raise ValueError(f"Unsupported low-speed feedback ID: 0x{arbitration_id:X}")
    if len(payload) < 8:
        raise ValueError(f"Expected 8-byte low-speed feedback payload, got {len(payload)}.")
    return SonglingLowSpeedFeedback(
        joint_name=joint_name,
        voltage=_int_from_be(payload, 0, 2, signed=False),
        foc_temp=_int_from_be(payload, 2, 4, signed=True),
        motor_temp=payload[4],
        foc_status_code=payload[5],
        bus_current=_int_from_be(payload, 6, 8, signed=False),
    )


def encode_motion_ctrl_1(*, emergency_stop: int = 0x00, track_ctrl: int = 0x00, grag_teach_ctrl: int = 0x00) -> bytes:
    return bytes([emergency_stop & 0xFF, track_ctrl & 0xFF, grag_teach_ctrl & 0xFF, 0, 0, 0, 0, 0])


def encode_motion_ctrl_2(
    *,
    ctrl_mode: int = 0x01,
    move_mode: int = 0x01,
    move_spd_rate_ctrl: int = 30,
    mit_mode: int = 0x00,
    residence_time: int = 0,
    installation_pos: int = 0x00,
) -> bytes:
    return bytes(
        [
            ctrl_mode & 0xFF,
            move_mode & 0xFF,
            max(0, min(int(move_spd_rate_ctrl), 100)) & 0xFF,
            mit_mode & 0xFF,
            residence_time & 0xFF,
            installation_pos & 0xFF,
            0x00,
            0x00,
        ]
    )


def encode_motor_enable_disable(*, motor_num: int = 0xFF, enable_flag: int = 0x02) -> bytes:
    return bytes([motor_num & 0xFF, enable_flag & 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


def encode_joint_command_pair(arbitration_id: int, first_value: int, second_value: int) -> bytes:
    if arbitration_id not in JOINT_COMMAND_IDS:
        raise ValueError(f"Unsupported joint command ID: 0x{arbitration_id:X}")
    return _int_to_be(first_value, 4, signed=True) + _int_to_be(second_value, 4, signed=True)


def encode_gripper_command(
    *,
    position: int,
    effort: int,
    status_code: int = 0x01,
    set_zero: int = 0x00,
) -> bytes:
    return (
        _int_to_be(position, 4, signed=True)
        + _int_to_be(effort, 2, signed=False)
        + bytes([status_code & 0xFF, set_zero & 0xFF])
    )


def installation_pos_for_side(side: str | None) -> int:
    if side == "left":
        return 0x02
    if side == "right":
        return 0x03
    return 0x00

