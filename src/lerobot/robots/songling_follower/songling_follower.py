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
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.songling_safety import raise_if_songling_parameter_config_unsafe

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_songling_follower import SonglingFollowerConfig
from .protocol import DEFAULT_SONGLING_JOINT_NAMES
from .transport import make_songling_port

logger = logging.getLogger(__name__)


class SonglingFollower(Robot):
    """Songling/Piper follower built around piper_sdk with an optional raw-CAN debug path."""

    config_class = SonglingFollowerConfig
    name = "songling_follower"

    def __init__(self, config: SonglingFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = make_songling_port(config)
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
        _ = calibrate
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

        raise_if_songling_parameter_config_unsafe(cfg=self.config, entrypoint=f"{self.__class__.__name__}.configure")

        if self.config.auto_configure_master_slave_on_connect and self.config.leader_follower_role:
            self.bus.configure_leader_follower()
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

    @check_if_not_connected
    def get_commanded_action(self, poll: bool = True) -> RobotAction:
        commanded = self.bus.get_commanded_positions(poll=poll)
        return {f"{joint_name}.pos": float(commanded[joint_name]) for joint_name in commanded}

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
    def hold_position(self, positions: dict[str, float]) -> RobotAction:
        sent = self.bus.send_targets({joint_name: float(positions[joint_name]) for joint_name in DEFAULT_SONGLING_JOINT_NAMES})
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
