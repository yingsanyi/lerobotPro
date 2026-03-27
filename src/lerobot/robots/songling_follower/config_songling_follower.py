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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

SONGLING_DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-150.0, 150.0),
    "joint_2": (0.0, 180.0),
    "joint_3": (-170.0, 0.0),
    "joint_4": (-100.0, 100.0),
    "joint_5": (-70.0, 70.0),
    "joint_6": (-120.0, 120.0),
    "gripper": (0.0, 100.0),
}
SONGLING_DEFAULT_JOINT_ZERO_OFFSETS: dict[str, float] = {
    joint_name: 0.0 for joint_name in SONGLING_DEFAULT_JOINT_LIMITS
}
SONGLING_DEFAULT_PRE_DISABLE_POSE: dict[str, float] = {
    joint_name: 0.0 for joint_name in SONGLING_DEFAULT_JOINT_LIMITS
}


@dataclass
class SonglingFollowerConfigBase:
    """Configuration for one Songling/Piper arm driven through piper_sdk or raw CAN."""

    channel: str
    side: str | None = None
    transport_backend: str = "piper_sdk"
    interface: str = "socketcan"
    use_can_fd: bool = False
    bitrate: int = 1000000
    can_data_bitrate: int = 5000000
    piper_judge_flag: bool = False
    disable_torque_on_disconnect: bool = False
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(SONGLING_DEFAULT_JOINT_LIMITS))
    joint_zero_offsets: dict[str, float] = field(default_factory=lambda: dict(SONGLING_DEFAULT_JOINT_ZERO_OFFSETS))
    pre_disable_pose: dict[str, float] = field(default_factory=lambda: dict(SONGLING_DEFAULT_PRE_DISABLE_POSE))
    allow_unverified_commanding: bool = False
    auto_enable_on_connect: bool = True
    auto_configure_mode_on_connect: bool = True
    speed_percent: int = 20
    motion_mode: str = "j"
    ctrl_mode: int = 0x01
    move_mode: int = 0x01
    mit_mode: int = 0x00
    residence_time: int = 0
    installation_pos: str | None = None
    command_repeat: int = 1
    command_interval_s: float = 0.0
    mode_keepalive_s: float = 0.5
    poll_max_msgs: int = 256
    joint_scale: float = 1e-3
    gripper_scale: float = 1e-3
    gripper_force: float = 1.0
    gripper_status_code: int = 0x01
    gripper_set_zero: int = 0x00
    enable_retry_count: int = 50
    enable_retry_interval_s: float = 0.02
    auto_configure_master_slave_on_connect: bool = False
    leader_follower_role: str | None = None
    leader_follower_feedback_offset: int = 0x00
    leader_follower_ctrl_offset: int = 0x00
    leader_follower_linkage_offset: int = 0x00


@RobotConfig.register_subclass("songling_follower")
@dataclass
class SonglingFollowerConfig(RobotConfig, SonglingFollowerConfigBase):
    pass
