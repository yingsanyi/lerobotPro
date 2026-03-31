#!/usr/bin/env python

from types import SimpleNamespace

import pytest

from lerobot.robots import make_robot_from_config
from lerobot.robots.bi_songling_follower import BiSonglingFollowerConfig
from lerobot.robots.songling_follower import SonglingFollowerConfig
from lerobot.robots.songling_follower import transport as songling_transport
from lerobot.robots.songling_follower.protocol import (
    ARM_JOINT_CTRL_12,
    DEFAULT_SONGLING_JOINT_NAMES,
    decode_gripper_feedback,
    decode_gripper_command,
    decode_joint_feedback,
    decode_joint_command,
    encode_gripper_command,
    encode_joint_command_pair,
    encode_master_slave_config,
)
from lerobot.utils.songling_safety import raise_if_songling_parameter_config_unsafe


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


def test_decode_joint_and_gripper_command():
    joint_payload = bytes.fromhex("000003e8fffffc18")
    gripper_payload = bytes.fromhex("0000138803e801ae")

    joint_command = decode_joint_command(0x155, joint_payload)
    gripper_command = decode_gripper_command(gripper_payload)

    assert joint_command["joint_1"] == 1000
    assert joint_command["joint_2"] == -1000
    assert gripper_command.position == 5000
    assert gripper_command.effort == 1000
    assert gripper_command.status_code == 0x01
    assert gripper_command.set_zero == 0xAE


def test_encode_master_slave_config():
    payload = encode_master_slave_config(
        linkage_config=0xFA,
        feedback_offset=0x10,
        ctrl_offset=0x20,
        linkage_offset=0x10,
    )
    assert payload.hex() == "fa10201000000000"


def test_make_bi_songling_robot_from_config(tmp_path):
    left_cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )
    right_cfg = SonglingFollowerConfig(
        channel="can0",
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


class _DummyPiperInterface:
    def __init__(self):
        self.create_can_bus_calls = []
        self.connect_port_calls = []
        self.disconnect_port_calls = 0
        self.connected = False

    def CreateCanBus(self, can_name, bustype="socketcan", expected_bitrate=1000000, judge_flag=False):
        self.create_can_bus_calls.append(
            {
                "can_name": can_name,
                "bustype": bustype,
                "expected_bitrate": expected_bitrate,
                "judge_flag": judge_flag,
            }
        )

    def ConnectPort(self, can_init=False, piper_init=True, start_thread=True):
        self.connect_port_calls.append(
            {
                "can_init": can_init,
                "piper_init": piper_init,
                "start_thread": start_thread,
            }
        )
        self.connected = True

    def DisconnectPort(self):
        self.disconnect_port_calls += 1
        self.connected = False

    def get_connect_status(self):
        return self.connected

    def GetArmStatus(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0, arm_status=SimpleNamespace())

    def GetArmJointMsgs(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0, joint_state=SimpleNamespace())

    def GetArmGripperMsgs(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0, gripper_state=SimpleNamespace())

    def GetArmCtrlCode151(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0, ctrl_151=SimpleNamespace())

    def GetArmJointCtrl(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0, joint_ctrl=SimpleNamespace())

    def GetArmGripperCtrl(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0, gripper_ctrl=SimpleNamespace())

    def GetMotorStates(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0)

    def GetDriverStates(self):
        return SimpleNamespace(time_stamp=0.0, Hz=0.0)

    def EnableArm(self, *args, **kwargs):
        return None

    def DisableArm(self, *args, **kwargs):
        return None

    def GripperCtrl(self, *args, **kwargs):
        return None

    def EmergencyStop(self, *args, **kwargs):
        return None

    def MotionCtrl_2(self, *args, **kwargs):
        return None

    def MasterSlaveConfig(self, *args, **kwargs):
        return None

    def JointCtrl(self, *args, **kwargs):
        return None


def _driver_state(*, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        can_id=0x261,
        vol=480,
        foc_temp=30,
        motor_temp=35,
        foc_status_code=0,
        foc_status=SimpleNamespace(
            driver_enable_status=enabled,
            driver_error_status=False,
            collision_status=False,
            stall_status=False,
        ),
        bus_current=0,
    )


def test_songling_robot_defaults_to_piper_sdk_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: _DummyPiperInterface())

    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    robot = make_robot_from_config(cfg)

    assert robot.bus.__class__.__name__ == "PiperSDKPort"


def test_songling_robot_supports_piper_sdk_backend(monkeypatch, tmp_path):
    created_args = []
    interface = _DummyPiperInterface()

    def make_interface(channel, judge_flag, can_auto_init):
        created_args.append((channel, judge_flag, can_auto_init))
        return interface

    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", make_interface)

    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        transport_backend="piper_sdk",
        piper_judge_flag=True,
        interface="slcan",
        bitrate=500000,
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    port = songling_transport.make_songling_port(cfg)
    port.connect()

    assert port.__class__.__name__ == "PiperSDKPort"
    assert created_args == [("can1", True, False)]
    assert interface.create_can_bus_calls == [
        {
            "can_name": "can1",
            "bustype": "slcan",
            "expected_bitrate": 500000,
            "judge_flag": True,
        }
    ]
    assert interface.connect_port_calls == [
        {
            "can_init": False,
            "piper_init": True,
            "start_thread": True,
        }
    ]


def test_songling_parameter_safety_rejects_installation_pos_downlink(tmp_path):
    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        installation_pos="left",
        allow_unverified_commanding=True,
        calibration_dir=tmp_path / "calibration",
    )

    with pytest.raises(RuntimeError, match="installation_pos"):
        raise_if_songling_parameter_config_unsafe(cfg=cfg, entrypoint="test")


def test_songling_port_blocks_leader_follower_reconfig(monkeypatch, tmp_path):
    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: _DummyPiperInterface())
    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        transport_backend="piper_sdk",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    port = songling_transport.PiperSDKPort(cfg)

    with pytest.raises(RuntimeError, match="MasterSlaveConfig"):
        port.configure_leader_follower(role="master")


def test_songling_port_sanitizes_runtime_installation_pos(monkeypatch, tmp_path):
    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: _DummyPiperInterface())
    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        transport_backend="piper_sdk",
        installation_pos="left",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    port = songling_transport.PiperSDKPort(cfg)

    assert port.installation_pos() == 0x00


def test_songling_port_keeps_installation_pos_blocked_even_when_runtime_reports_side(monkeypatch, tmp_path):
    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: _DummyPiperInterface())
    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        transport_backend="piper_sdk",
        installation_pos=None,
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    port = songling_transport.PiperSDKPort(cfg)
    port.mode_command_valid = True
    port.mode_command = {"installation_pos": 0x02}

    assert port.installation_pos() == 0x00


def test_wait_for_driver_enable_status_throttles_reenable(monkeypatch, tmp_path):
    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: _DummyPiperInterface())
    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        transport_backend="piper_sdk",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
        enable_retry_count=11,
        enable_retry_interval_s=0.0,
    )

    port = songling_transport.PiperSDKPort(cfg)
    enable_calls: list[bool] = []
    poll_count = {"value": 0}

    def fake_enable_motors(*, enable: bool) -> None:
        enable_calls.append(bool(enable))

    def fake_poll(*, max_msgs=None) -> None:
        _ = max_msgs
        poll_count["value"] += 1

    def fake_status() -> dict[str, bool | None]:
        if poll_count["value"] >= 11:
            return {joint_name: True for joint_name in DEFAULT_SONGLING_JOINT_NAMES[:-1]}
        return {joint_name: None for joint_name in DEFAULT_SONGLING_JOINT_NAMES[:-1]}

    monkeypatch.setattr(port, "enable_motors", fake_enable_motors)
    monkeypatch.setattr(port, "poll", fake_poll)
    monkeypatch.setattr(port, "get_driver_enable_status", fake_status)

    assert port.wait_for_driver_enable_status(enabled=True) is True
    assert enable_calls == [True, True, True]


def test_songling_robot_rejects_unknown_backend(tmp_path):
    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        transport_backend="mystery_backend",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    with pytest.raises(ValueError, match="Unsupported Songling transport backend"):
        _ = make_robot_from_config(cfg)


def test_piper_sdk_poll_uses_motor_state_positions_when_joint_msgs_missing(monkeypatch, tmp_path):
    class DummyPiperInterface(_DummyPiperInterface):
        def GetArmGripperMsgs(self):
            return SimpleNamespace(
                time_stamp=1.0,
                Hz=40.0,
                gripper_state=SimpleNamespace(grippers_angle=4321, grippers_effort=222, status_code=3),
            )

        def GetMotorStates(self):
            motors = {
                f"motor_{idx}": SimpleNamespace(
                    can_id=0x250 + idx,
                    pos=idx * 100,
                    motor_speed=idx * 10,
                    current=idx * 100,
                )
                for idx in range(1, 7)
            }
            return SimpleNamespace(time_stamp=6.0, Hz=50.0, **motors)

        def GetDriverStates(self):
            motors = {f"motor_{idx}": _driver_state(enabled=True) for idx in range(1, 7)}
            return SimpleNamespace(time_stamp=6.0, Hz=50.0, **motors)

    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: DummyPiperInterface())

    cfg = SonglingFollowerConfig(
        channel="can1",
        side="left",
        transport_backend="piper_sdk",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    port = songling_transport.PiperSDKPort(cfg)
    port._is_connected = True
    port.poll()

    assert port.joint_feedback_valid is True
    assert port.gripper_feedback_valid is True
    assert port.joint_position_seen["joint_1"] is True
    assert port.joint_positions_raw["joint_6"] == 34377
    assert port.joint_position_seen["gripper"] is True
    assert port.joint_positions_raw["gripper"] == 4321
    assert port.has_reliable_position_feedback() is True


def test_piper_sdk_poll_uses_gripper_ctrl_state_as_feedback_fallback(monkeypatch, tmp_path):
    class DummyPiperInterface(_DummyPiperInterface):
        def GetMotorStates(self):
            motors = {
                f"motor_{idx}": SimpleNamespace(
                    can_id=0x250 + idx,
                    pos=idx * 100,
                    motor_speed=idx * 10,
                    current=idx * 100,
                )
                for idx in range(1, 7)
            }
            return SimpleNamespace(time_stamp=6.0, Hz=50.0, **motors)

        def GetDriverStates(self):
            motors = {f"motor_{idx}": _driver_state(enabled=True) for idx in range(1, 7)}
            return SimpleNamespace(time_stamp=6.0, Hz=50.0, **motors)

        def GetArmGripperCtrl(self):
            return SimpleNamespace(
                time_stamp=2.0,
                Hz=30.0,
                gripper_ctrl=SimpleNamespace(grippers_angle=6789, grippers_effort=333, status_code=5, set_zero=0),
            )

    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: DummyPiperInterface())

    cfg = SonglingFollowerConfig(
        channel="can1",
        side="right",
        transport_backend="piper_sdk",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    port = songling_transport.PiperSDKPort(cfg)
    port._is_connected = True
    port.poll()

    assert port.gripper_feedback_valid is True
    assert port.joint_position_seen["gripper"] is True
    assert port.joint_positions_raw["gripper"] == 6789
    assert port.commanded_position_seen["gripper"] is True
    assert port.commanded_positions_raw["gripper"] == 6789
    assert port.gripper_feedback_timestamp == 2.0
    assert port.gripper_feedback_hz == 30.0


def test_piper_sdk_poll_prefers_newer_gripper_ctrl_state_for_display(monkeypatch, tmp_path):
    class DummyPiperInterface(_DummyPiperInterface):
        def GetMotorStates(self):
            motors = {
                f"motor_{idx}": SimpleNamespace(
                    can_id=0x250 + idx,
                    pos=idx * 100,
                    motor_speed=idx * 10,
                    current=idx * 100,
                )
                for idx in range(1, 7)
            }
            return SimpleNamespace(time_stamp=6.0, Hz=50.0, **motors)

        def GetDriverStates(self):
            motors = {f"motor_{idx}": _driver_state(enabled=True) for idx in range(1, 7)}
            return SimpleNamespace(time_stamp=6.0, Hz=50.0, **motors)

        def GetArmGripperMsgs(self):
            return SimpleNamespace(
                time_stamp=1.0,
                Hz=40.0,
                gripper_state=SimpleNamespace(grippers_angle=4321, grippers_effort=222, status_code=3),
            )

        def GetArmGripperCtrl(self):
            return SimpleNamespace(
                time_stamp=2.0,
                Hz=30.0,
                gripper_ctrl=SimpleNamespace(grippers_angle=6789, grippers_effort=333, status_code=5, set_zero=0),
            )

    monkeypatch.setattr(songling_transport, "C_PiperInterface_V2", lambda *args, **kwargs: DummyPiperInterface())

    cfg = SonglingFollowerConfig(
        channel="can0",
        side="right",
        transport_backend="piper_sdk",
        calibration_dir=tmp_path / "calibration",
        allow_unverified_commanding=False,
    )

    port = songling_transport.PiperSDKPort(cfg)
    port._is_connected = True
    port.poll()

    assert port.joint_positions_raw["gripper"] == 6789
    assert port.gripper_feedback_timestamp == 2.0
