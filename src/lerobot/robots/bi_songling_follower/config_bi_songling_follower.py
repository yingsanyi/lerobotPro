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

from dataclasses import dataclass

from lerobot.robots.songling_follower import SonglingFollowerConfigBase

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_songling_follower")
@dataclass
class BiSonglingFollowerConfig(RobotConfig):
    """Configuration for bimanual Songling integrated CAN followers."""

    left_arm_config: SonglingFollowerConfigBase
    right_arm_config: SonglingFollowerConfigBase

