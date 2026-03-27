#!/usr/bin/env python

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest


def _load_manual_joint_control_module():
    pytest.importorskip("PyQt5")
    if "draccus" not in sys.modules:
        draccus_stub = types.ModuleType("draccus")
        draccus_stub.parse = lambda *args, **kwargs: None
        sys.modules["draccus"] = draccus_stub
    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_dump = lambda *args, **kwargs: ""
        sys.modules["yaml"] = yaml_stub
    module_path = Path(__file__).resolve().parents[2] / "examples" / "songling_aloha" / "manual_joint_control.py"
    spec = spec_from_file_location("test_songling_manual_joint_control_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_arm(module, *, allow_unverified: bool, enabled: bool, has_feedback: bool):
    class DummyBus:
        def __init__(self):
            self.low_speed_feedback = {"joint_1": {"driver_enable_status": enabled}} if has_feedback else {}
            self.high_speed_feedback = {"joint_1": {"position": 123}} if has_feedback else {}
            self.command_feedback_valid = has_feedback
            self.joint_position_seen = {
                joint_name: has_feedback for joint_name in module.JOINT_NAMES
            }

        def get_driver_enable_status(self):
            return {joint_name: enabled for joint_name in module.JOINT_NAMES[:-1]}

    return SimpleNamespace(
        follower=SimpleNamespace(
            config=SimpleNamespace(allow_unverified_commanding=allow_unverified),
            bus=DummyBus(),
        )
    )


def test_manual_send_fallback_note_requires_unverified_with_feedback():
    module = _load_manual_joint_control_module()
    arm = _make_arm(module, allow_unverified=True, enabled=True, has_feedback=True)

    note = module._manual_send_fallback_note(arm, module.MISSING_STATUS_OR_MODE_REASON)

    assert note is not None
    assert "手动发送" in note


def test_manual_send_fallback_note_stays_disabled_without_feedback():
    module = _load_manual_joint_control_module()
    arm = _make_arm(module, allow_unverified=True, enabled=True, has_feedback=False)

    note = module._manual_send_fallback_note(arm, module.MISSING_STATUS_OR_MODE_REASON)

    assert note is None


def test_send_targets_keeps_requested_target_for_continuous_approach():
    module = _load_manual_joint_control_module()
    window = module.ManualJointControlWindow.__new__(module.ManualJointControlWindow)
    requested = {f"{joint_name}.pos": float(index) for index, joint_name in enumerate(module.JOINT_NAMES, start=1)}
    sent = {f"{joint_name}.pos": float(index) / 10.0 for index, joint_name in enumerate(module.JOINT_NAMES, start=1)}

    class DummyBus:
        def ensure_can_command_mode(self, *, force: bool = False):
            self.force = force

    class DummyFollower:
        def __init__(self):
            self.bus = DummyBus()
            self.calls = []

        def send_action(self, action):
            self.calls.append(dict(action))
            return dict(sent)

    arm = SimpleNamespace(side="left", follower=DummyFollower())
    window.connected = True
    window._manual_send_in_progress = False
    window.auto_enable_checkbox = SimpleNamespace(isChecked=lambda: False)
    window.hold_position_checkbox = SimpleNamespace(isChecked=lambda: True)
    window.motors_enabled = True
    window.commanded_targets = {"left": None}
    window._manual_send_fallback_warned = {"left": False}
    window._arm_has_feedback = lambda side: True
    window._arm_ready_for_manual_send = lambda arm: (True, None, None)
    window._requested_action = lambda side: dict(requested)
    window._requested_matches_current = lambda side, action: False
    window._log = lambda message: None
    window._log_arm_status = lambda *args, **kwargs: None
    window.refresh_positions = lambda sync_targets=False: None

    results = window._send_targets_for_arms([arm], allow_auto_enable=False, verbose=False)

    assert results == {"left": "sent"}
    assert arm.follower.calls == [requested]
    assert window.commanded_targets["left"] == {
        joint_name: requested[f"{joint_name}.pos"] for joint_name in module.JOINT_NAMES
    }


def test_send_targets_is_blocked_when_motors_disabled():
    module = _load_manual_joint_control_module()
    logs = []

    class DummyBus:
        def ensure_can_command_mode(self, *, force: bool = False):
            raise AssertionError("should not try to switch mode while motors are disabled")

    class DummyFollower:
        def __init__(self):
            self.bus = DummyBus()
            self.calls = []

        def send_action(self, action):
            self.calls.append(dict(action))
            return dict(action)

    arm = SimpleNamespace(side="left", follower=DummyFollower())
    window = module.ManualJointControlWindow.__new__(module.ManualJointControlWindow)
    window.connected = True
    window.motors_enabled = False
    window._manual_send_in_progress = False
    window._log = lambda message: logs.append(message)

    results = window._send_targets_for_arms([arm], allow_auto_enable=False, verbose=True)

    assert results == {}
    assert arm.follower.calls == []
    assert any("请先点击“电机使能”" in message for message in logs)


def test_refresh_positions_displays_gripper_command_fallback_when_feedback_missing():
    module = _load_manual_joint_control_module()

    class DummyLabel:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    class DummyBus:
        def __init__(self):
            self.joint_position_seen = {joint_name: True for joint_name in module.JOINT_NAMES}
            self.joint_position_seen["gripper"] = False
            self.commanded_position_seen = {joint_name: False for joint_name in module.JOINT_NAMES}
            self.commanded_position_seen["gripper"] = True

        def has_reliable_position_feedback(self):
            return False

    arm = SimpleNamespace(side="right", follower=SimpleNamespace(bus=DummyBus()))
    labels = {joint_name: DummyLabel() for joint_name in module.JOINT_NAMES}
    window = module.ManualJointControlWindow.__new__(module.ManualJointControlWindow)
    window.connected = True
    window.arm_handles = [arm]
    window.current_labels = {"right": labels}
    window.current_values = {"right": {}}
    window._targets_initialized_by_side = {"right": False}
    window.motors_enabled = False
    window.status_label = SimpleNamespace(setText=lambda text: None)
    window._arm_has_any_feedback = lambda side: True
    window._arm_has_position_feedback = lambda side: False
    window._arm_positions = lambda side: {
        **{joint_name: float(index) for index, joint_name in enumerate(module.JOINT_NAMES[:-1], start=1)},
        "gripper": 12.345,
    }

    window.refresh_positions(sync_targets=False)

    assert labels["gripper"].text == "12.345"


def test_hold_maintain_uses_send_action_to_continue_towards_target():
    module = _load_manual_joint_control_module()
    window = module.ManualJointControlWindow.__new__(module.ManualJointControlWindow)
    target = {joint_name: float(index) for index, joint_name in enumerate(module.JOINT_NAMES, start=1)}

    class DummyFollower:
        def __init__(self):
            self.calls = []

        def send_action(self, action):
            self.calls.append(dict(action))
            return {key: value for key, value in action.items()}

        def hold_position(self, positions):
            raise AssertionError("hold_position should not be used for continuous target approach")

    arm = SimpleNamespace(side="left", follower=DummyFollower())
    window.connected = True
    window.motors_enabled = True
    window.hold_position_checkbox = SimpleNamespace(isChecked=lambda: True)
    window._manual_send_in_progress = False
    window._hold_send_in_progress = False
    window._hold_backoff_until = 0.0
    window._last_hold_error = None
    window.arm_handles = [arm]
    window.commanded_targets = {"left": dict(target)}
    window._arm_ready_for_command = lambda arm: (True, None)
    window._arm_has_feedback = lambda side: True
    window._log = lambda message: None
    window._log_arm_status = lambda *args, **kwargs: None

    window._maintain_hold_targets()

    assert arm.follower.calls == [window._position_dict_to_action(target)]
    assert window.commanded_targets["left"] == target


def test_hold_maintain_updates_each_selected_arm():
    module = _load_manual_joint_control_module()

    class DummyFollower:
        def __init__(self, side):
            self.side = side
            self.calls = []

        def send_action(self, action):
            self.calls.append(dict(action))
            return dict(action)

    left_target = {joint_name: float(index) for index, joint_name in enumerate(module.JOINT_NAMES, start=1)}
    right_target = {joint_name: float(index * 10) for index, joint_name in enumerate(module.JOINT_NAMES, start=1)}
    left_arm = SimpleNamespace(side="left", follower=DummyFollower("left"))
    right_arm = SimpleNamespace(side="right", follower=DummyFollower("right"))

    window = SimpleNamespace(
        connected=True,
        motors_enabled=True,
        hold_position_checkbox=SimpleNamespace(isChecked=lambda: True),
        _manual_send_in_progress=False,
        _hold_send_in_progress=False,
        _hold_backoff_until=0.0,
        _last_hold_error=None,
        arm_handles=[left_arm, right_arm],
        commanded_targets={"left": dict(left_target), "right": dict(right_target)},
        _arm_ready_for_command=lambda arm: (True, None),
        _arm_has_feedback=lambda side: True,
        _position_dict_to_action=lambda positions: {f"{joint_name}.pos": float(positions[joint_name]) for joint_name in module.JOINT_NAMES},
        _log=lambda message: None,
        _log_arm_status=lambda *args, **kwargs: None,
    )

    module.ManualJointControlWindow._maintain_hold_targets(window)

    assert left_arm.follower.calls == [window._position_dict_to_action(left_target)]
    assert right_arm.follower.calls == [window._position_dict_to_action(right_target)]


def test_enable_and_configure_tracks_zero_pose_instead_of_current_hold(monkeypatch):
    module = _load_manual_joint_control_module()
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)
    target = {joint_name: float(index) for index, joint_name in enumerate(module.JOINT_NAMES, start=1)}

    class DummyBus:
        def __init__(self):
            self.mode_calls = 0

        def ensure_can_command_mode(self, *, force: bool = False):
            self.mode_calls += int(force)

    arm = SimpleNamespace(side="left", follower=SimpleNamespace(bus=DummyBus()))
    called = {}

    def run_zero_pose_sequence(arms, sequence_name, allow_auto_enable):
        called.update({"arms": [arm.side for arm in arms], "sequence_name": sequence_name, "allow_auto_enable": allow_auto_enable})
        window.commanded_targets["left"] = dict(target)
        return True

    window = SimpleNamespace(
        connected=True,
        arm_handles=[arm],
        commanded_targets={"left": None},
        hold_position_checkbox=SimpleNamespace(isChecked=lambda: True),
        _wait_for_driver_enable_status_all=lambda arms, enabled: {"left": True},
        _arm_has_feedback=lambda side: True,
        _arm_ready_for_command=lambda _arm: (True, None),
        _run_zero_pose_sequence=run_zero_pose_sequence,
        _log_arm_status=lambda *args, **kwargs: None,
        _log=lambda message: None,
        refresh_positions=lambda sync_targets=False: None,
        status_label=SimpleNamespace(setText=lambda text: None),
        motors_enabled=False,
        _preload_hold_positions_for_arm=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not preload current pose")),
        _send_hold_positions_for_arm=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not immediately hold current pose")),
    )

    module.ManualJointControlWindow.enable_and_configure(window)

    assert called == {"arms": ["left"], "sequence_name": "使能流程", "allow_auto_enable": False}
    assert window.commanded_targets["left"] == target
    assert window.motors_enabled is True


def test_run_pre_disable_sequence_uses_relaxed_disable_tolerance():
    module = _load_manual_joint_control_module()
    arm = SimpleNamespace(side="left")
    called = {}
    window = SimpleNamespace(
        _get_config_pose=lambda side, attr_name: {joint_name: 0.0 for joint_name in module.JOINT_NAMES},
        _move_selected_arms_to_pose=lambda arms, poses_by_side, **kwargs: called.update(
            {
                "arms": [item.side for item in arms],
                "poses": poses_by_side,
                **kwargs,
            }
        )
        or True,
    )

    result = module.ManualJointControlWindow._run_pre_disable_sequence(
        window,
        [arm],
        sequence_name="失能前准备",
        allow_auto_enable=False,
    )

    assert result is True
    assert called["arms"] == ["left"]
    assert called["max_attempts"] == module.PRE_DISABLE_MAX_ATTEMPTS
    assert called["joint_tolerance"] == module.PRE_DISABLE_JOINT_TOL
    assert called["gripper_tolerance"] == module.PRE_DISABLE_GRIPPER_TOL


def test_disable_motors_aborts_when_pre_disable_pose_not_reached():
    module = _load_manual_joint_control_module()
    disable_calls = []
    logs = []
    arm = SimpleNamespace(
        side="left",
        follower=SimpleNamespace(bus=SimpleNamespace(enable_motors=lambda *, enable: disable_calls.append(enable))),
    )
    window = SimpleNamespace(
        connected=True,
        _manual_send_in_progress=False,
        _log=lambda message: logs.append(message),
        _selected_arms=lambda _side: [arm],
        _run_pre_disable_sequence=lambda arms, sequence_name, allow_auto_enable: False,
        arm_handles=[arm],
        commanded_targets={"left": {"joint_1": 1.0}},
        motors_enabled=True,
        status_label=SimpleNamespace(setText=lambda text: None),
    )

    module.ManualJointControlWindow.disable_motors(window)

    assert disable_calls == []
    assert window.motors_enabled is True
    assert any("取消失能" in message for message in logs)


def test_disable_motors_auto_disables_after_reaching_pre_disable_pose():
    module = _load_manual_joint_control_module()
    disable_wait_calls = []
    logs = []
    arm = SimpleNamespace(
        side="left",
        follower=SimpleNamespace(bus=SimpleNamespace(enable_motors=lambda *, enable: None)),
    )
    window = SimpleNamespace(
        connected=True,
        _manual_send_in_progress=False,
        _log=lambda message: logs.append(message),
        _selected_arms=lambda _side: [arm],
        _run_pre_disable_sequence=lambda arms, sequence_name, allow_auto_enable: True,
        _wait_for_driver_enable_status_all=lambda arms, enabled: disable_wait_calls.append(
            ([item.side for item in arms], enabled)
        ) or {"left": True},
        arm_handles=[arm],
        commanded_targets={"left": {"joint_1": 1.0}},
        motors_enabled=True,
        status_label=SimpleNamespace(setText=lambda text: None),
    )

    module.ManualJointControlWindow.disable_motors(window)

    assert disable_wait_calls == [(["left"], False)]
    assert window.commanded_targets["left"] is None
    assert window.motors_enabled is False
    assert any("无需手动回位" in message for message in logs)
    assert any("自动统一失能" in message for message in logs)
