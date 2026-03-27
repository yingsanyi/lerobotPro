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

from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.songling_follower import SonglingFollower, SonglingFollowerConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_songling_follower import BiSonglingFollowerConfig


class BiSonglingFollower(Robot):
    """Bimanual Songling integrated CAN followers."""

    config_class = BiSonglingFollowerConfig
    name = "bi_songling_follower"

    def __init__(self, config: BiSonglingFollowerConfig):
        super().__init__(config)
        self.config = config

        left_values = dict(config.left_arm_config.__dict__)
        left_values.pop("id", None)
        left_values.pop("calibration_dir", None)
        left_cfg = SonglingFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            **left_values,
        )
        right_values = dict(config.right_arm_config.__dict__)
        right_values.pop("id", None)
        right_values.pop("calibration_dir", None)
        right_cfg = SonglingFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            **right_values,
        )

        self.left_arm = SonglingFollower(left_cfg)
        self.right_arm = SonglingFollower(right_cfg)
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_arm_motors_ft = self.left_arm._motors_ft
        right_arm_motors_ft = self.right_arm._motors_ft
        return {
            **{f"left_{k}": v for k, v in left_arm_motors_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        left_arm_cameras_ft = self.left_arm._cameras_ft
        right_arm_cameras_ft = self.right_arm._cameras_ft
        return {
            **{f"left_{k}": v for k, v in left_arm_cameras_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_cameras_ft.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        raise NotImplementedError("Songling integrated chain motor setup is not exposed through this adapter.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = {}
        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()
        obs.update({f"left_{key}": value for key, value in left_obs.items()})
        obs.update({f"right_{key}": value for key, value in right_obs.items()})
        return obs

    @check_if_not_connected
    def get_commanded_action(self, poll: bool = True) -> RobotAction:
        left_action = self.left_arm.get_commanded_action(poll=poll)
        right_action = self.right_arm.get_commanded_action(poll=poll)
        return {
            **{f"left_{key}": value for key, value in left_action.items()},
            **{f"right_{key}": value for key, value in right_action.items()},
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
