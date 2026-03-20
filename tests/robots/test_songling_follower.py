#!/usr/bin/env python

from lerobot.robots import make_robot_from_config
from lerobot.robots.bi_songling_follower import BiSonglingFollowerConfig
from lerobot.robots.songling_follower import SonglingFollowerConfig
from lerobot.robots.songling_follower.protocol import (
    ARM_JOINT_CTRL_12,
    decode_gripper_feedback,
    decode_joint_feedback,
    encode_gripper_command,
    encode_joint_command_pair,
)


def test_encode_joint_command_pair_big_endian():
    payload = encode_joint_command_pair(ARM_JOINT_CTRL_12, 1000, -1000)
    assert payload.hex() == "000003e8fffffc18"


def test_decode_joint_and_gripper_feedback():
    joint_payload = bytes.fromhex("000003e8fffffc18")
    gripper_payload = bytes.fromhex("0000138803e80100")

    joint_feedback = decode_joint_feedback(0x2A5, joint_payload)
    gripper_feedback = decode_gripper_feedback(gripper_payload)

    assert joint_feedback["joint_1"] == 1000
    assert joint_feedback["joint_2"] == -1000
    assert gripper_feedback.position == 5000
    assert gripper_feedback.effort == 1000
    assert gripper_feedback.status_code == 0x01


def test_make_bi_songling_robot_from_config(tmp_path):
    left_cfg = SonglingFollowerConfig(
        port="can1",
        side="left",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )
    right_cfg = SonglingFollowerConfig(
        port="can0",
        side="right",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )
    cfg = BiSonglingFollowerConfig(
        id="songling_test",
        calibration_dir=tmp_path / "calibration",
        left_arm_config=left_cfg,
        right_arm_config=right_cfg,
    )

    robot = make_robot_from_config(cfg)

    assert robot.name == "bi_songling_follower"
    assert "left_joint_1.pos" in robot.action_features
    assert "right_gripper.pos" in robot.action_features
