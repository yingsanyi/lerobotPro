#!/usr/bin/env python

"""PyQt5 desktop joint-control panel for Songling followers."""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import draccus
import yaml

try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QFontDatabase
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSlider,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:
    raise ModuleNotFoundError("PyQt5 is required to run this tool. Install with `python -m pip install PyQt5`.") from exc

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.robots import RobotConfig
from lerobot.robots import bi_songling_follower, songling_follower  # noqa: F401
from lerobot.robots.songling_follower import SonglingFollower, SonglingFollowerConfig
from lerobot.robots.songling_follower.config_songling_follower import SonglingFollowerConfigBase
from lerobot.robots.songling_follower.transport import make_songling_port
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators import bi_openarm_leader, openarm_leader  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins


DEFAULT_CONFIG_PATH = Path("examples/songling_aloha/teleop.yaml")
JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
CAN_NAME_PATTERN = re.compile(r"^(?:can|slcan)\d+$")
PREFERRED_UI_FONT_FAMILIES = (
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "SimHei",
)
UI_FONT_SIZE = 10
LOG_FONT_SIZE = 9
SLIDER_SCALE = 10
HOLD_TIMER_MS = 10
HOLD_BACKOFF_MS = 100
SOFT_ZERO_MAX_ATTEMPTS = 12
SOFT_ZERO_SETTLE_S = 0.05
SOFT_ZERO_JOINT_TOL = 0.5
SOFT_ZERO_GRIPPER_TOL = 0.5
PRE_DISABLE_MAX_ATTEMPTS = 30
PRE_DISABLE_JOINT_TOL = 3.0
PRE_DISABLE_GRIPPER_TOL = 5.0
ENABLE_PRELOAD_REPEAT = 2
ENABLE_PRELOAD_SETTLE_S = 0.02
MISSING_STATUS_OR_MODE_REASON = "未收到真实状态帧，也未观察到 `0x151` 模式命令"
GROUP_BOX_STYLE = """
QGroupBox {
    border: 1px solid #d7c8a9;
    border-radius: 10px;
    margin-top: 10px;
    background: #fbf8f1;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #5f4d32;
    font-weight: 700;
}
"""
BUTTON_STYLE_NEUTRAL = """
QPushButton {
    background: #f2ede3;
    color: #40362a;
    border: 1px solid #cdbd9a;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover { background: #ebe3d2; }
QPushButton:pressed { background: #dfd3ba; }
"""
BUTTON_STYLE_WARM = """
QPushButton {
    background: #d9a85f;
    color: #2f2418;
    border: 1px solid #bc8742;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 700;
}
QPushButton:hover { background: #e1b46a; }
QPushButton:pressed { background: #c8924d; }
"""
BUTTON_STYLE_PRIMARY = """
QPushButton {
    background: #264653;
    color: #f7f5ef;
    border: 1px solid #1d3640;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 700;
}
QPushButton:hover { background: #315767; }
QPushButton:pressed { background: #1c343e; }
"""
BUTTON_STYLE_DANGER = """
QPushButton {
    background: #b54d45;
    color: #fff6f4;
    border: 1px solid #943b35;
    border-radius: 8px;
    padding: 6px 10px;
    font-weight: 700;
}
QPushButton:hover { background: #c45a52; }
QPushButton:pressed { background: #963d37; }
"""


def _ui_text(_en: str, zh: str) -> str:
    return zh


SIDE_LABELS = {
    "left": _ui_text("Left Arm", "\u5de6\u81c2"),
    "right": _ui_text("Right Arm", "\u53f3\u81c2"),
}
JOINT_LABELS = {
    "joint_1": _ui_text("Joint 1", "\u5173\u82821"),
    "joint_2": _ui_text("Joint 2", "\u5173\u82822"),
    "joint_3": _ui_text("Joint 3", "\u5173\u82823"),
    "joint_4": _ui_text("Joint 4", "\u5173\u82824"),
    "joint_5": _ui_text("Joint 5", "\u5173\u82825"),
    "joint_6": _ui_text("Joint 6", "\u5173\u82826"),
    "gripper": _ui_text("Gripper", "\u5939\u722a"),
}
STATUS_LABELS = {
    "ctrl_mode": _ui_text("Ctrl Mode", "\u63a7\u5236\u6a21\u5f0f"),
    "arm_status": _ui_text("Arm Status", "\u673a\u68b0\u81c2\u72b6\u6001"),
    "mode_feed": _ui_text("Mode Feed", "\u6a21\u5f0f\u53cd\u9988"),
    "teach_status": _ui_text("Teach Status", "\u62d6\u62fd\u72b6\u6001"),
    "motion_status": _ui_text("Motion Status", "\u8fd0\u52a8\u72b6\u6001"),
    "trajectory_num": _ui_text("Trajectory", "\u8f68\u8ff9\u53f7"),
    "err_code": _ui_text("Err Code", "\u9519\u8bef\u7801"),
}


@dataclass
class ManualControlConfig:
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    fps: int = 30
    display_data: bool = False
    display_compressed_images: bool = False
    dataset: Any | None = None
    play_sounds: bool | None = None
    voice_lang: str | None = None
    voice_rate: int | None = None
    voice_engine: str | None = None
    voice_piper_model: str | None = None
    voice_piper_binary: str | None = None
    voice_piper_speaker: int | None = None
    display_ip: str | None = None
    display_port: int | None = None
    resume: bool | None = None


@dataclass
class ArmHandle:
    side: str
    follower: SonglingFollower


def _manual_send_fallback_note(arm: ArmHandle, strict_reason: str | None) -> str | None:
    if strict_reason != MISSING_STATUS_OR_MODE_REASON:
        return None
    if not bool(getattr(arm.follower.config, "allow_unverified_commanding", False)):
        return None

    enable_status = arm.follower.bus.get_driver_enable_status()
    known_states = [state for state in enable_status.values() if state is not None]
    if not known_states or not all(bool(state) for state in known_states):
        return None

    has_runtime_feedback = (
        bool(arm.follower.bus.low_speed_feedback)
        or bool(arm.follower.bus.high_speed_feedback)
        or bool(getattr(arm.follower.bus, "command_feedback_valid", False))
        or any(bool(arm.follower.bus.joint_position_seen.get(joint_name, False)) for joint_name in JOINT_NAMES)
    )
    if not has_runtime_feedback:
        return None

    return (
        "未收到真实状态/`0x151` 回读，但已收到驱动与电机反馈；"
        "当前按直控校验兜底放行“手动发送”，自动保持仍保持禁用。"
    )


def _patch_multiprocess_resource_tracker() -> None:
    """Work around multiprocess<->Python3.12 shutdown noise."""
    try:
        from multiprocess import resource_tracker as _rt  # type: ignore
    except Exception:
        return

    if getattr(_rt, "_songling_patch_applied", False):
        return

    def _safe_stop_locked(
        self,
        close=os.close,
        waitpid=os.waitpid,
    ):
        recursion = 0
        try:
            recursion_attr = getattr(self._lock, "_recursion_count", None)
            if callable(recursion_attr):
                recursion = int(recursion_attr())
            elif recursion_attr is not None:
                recursion = int(recursion_attr)
        except Exception:
            recursion = 0

        if recursion > 1:
            return self._reentrant_call_error()
        if self._fd is None or self._pid is None:
            return

        close(self._fd)
        self._fd = None
        waitpid(self._pid, 0)
        self._pid = None

    try:
        _rt.ResourceTracker._stop_locked = _safe_stop_locked
        _rt._songling_patch_applied = True
    except Exception:
        pass


_patch_multiprocess_resource_tracker()


def _side_label(side: str) -> str:
    return SIDE_LABELS.get(side, side)


def _joint_label(joint_name: str) -> str:
    return JOINT_LABELS.get(joint_name, joint_name)


def _format_status(status: dict[str, int]) -> str:
    if not status:
        return _ui_text("No status frame yet", "\u6682\u672a\u6536\u5230\u72b6\u6001\u5e27")
    return " | ".join(f"{STATUS_LABELS.get(key, key)}={value}" for key, value in status.items())


def _pick_qt_font_family() -> str:
    families = sorted(set(QFontDatabase().families()))
    lower_map = {family.lower(): family for family in families}

    for family in PREFERRED_UI_FONT_FAMILIES:
        if family in families:
            return family

    keyword_groups = [
        ("noto", "cjk", "sc"),
        ("noto", "sans", "sc"),
        ("source", "han", "sans"),
        ("wenquanyi",),
        ("yahei",),
        ("simhei",),
        ("pingfang", "sc"),
        ("cjk", "sc"),
        ("sc",),
    ]
    for keywords in keyword_groups:
        for lower_name, original_name in lower_map.items():
            if all(keyword in lower_name for keyword in keywords):
                return original_name

    return QApplication.font().family()


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Songling \u53cc\u81c2\u5173\u8282\u624b\u52a8\u63a7\u5236\u53f0\u3002")
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="\u5305\u542b Songling \u7edf\u4e00\u914d\u7f6e\uff08robot/teleop/dataset \u7b49\uff09\u7684 YAML\u3002",
    )
    parser.add_argument(
        "--poll-ms",
        dest="poll_ms",
        type=int,
        default=250,
        help="\u5f53\u524d\u4f4d\u7f6e\u8f6e\u8be2\u5468\u671f\uff0c\u5355\u4f4d\u6beb\u79d2\u3002",
    )
    parser.add_argument(
        "--window-title",
        dest="window_title",
        default="Songling \u53cc\u81c2\u5173\u8282\u63a7\u5236",
        help="\u684c\u9762\u7a97\u53e3\u6807\u9898\u3002",
    )
    return parser.parse_known_args()


def _load_config(config_path: Path, unknown_args: list[str]) -> ManualControlConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return draccus.parse(ManualControlConfig, config_path=config_path, args=unknown_args)


def _to_yaml_safe_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        data: dict[str, Any] = {}
        if hasattr(value, "type"):
            try:
                data["type"] = str(getattr(value, "type"))
            except Exception:
                pass
        for field in dataclass_fields(value):
            data[field.name] = _to_yaml_safe_value(getattr(value, field.name))
        return data
    if isinstance(value, dict):
        return {str(key): _to_yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_yaml_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _make_arm_config(
    *,
    app_id: str,
    calibration_dir: Path | None,
    raw_arm_cfg: SonglingFollowerConfigBase,
) -> SonglingFollowerConfig:
    values = dict(raw_arm_cfg.__dict__)
    values.pop("id", None)
    values.pop("calibration_dir", None)
    # This UI only needs CAN joint control. Keep camera config in teleop.yaml for
    # other flows, but skip camera startup here so missing UVC devices never block
    # the desktop from launching.
    values["cameras"] = {}
    values["allow_unverified_commanding"] = True
    values["auto_configure_master_slave_on_connect"] = False
    values["auto_enable_on_connect"] = False
    values["auto_configure_mode_on_connect"] = False
    values["transport_backend"] = "piper_sdk"
    return SonglingFollowerConfig(id=app_id, calibration_dir=calibration_dir, **values)


def _make_probe_port_config(
    *,
    app_id: str,
    calibration_dir: Path | None,
    raw_arm_cfg: SonglingFollowerConfigBase,
    channel: str,
) -> SonglingFollowerConfig:
    values = dict(raw_arm_cfg.__dict__)
    values.pop("id", None)
    values.pop("calibration_dir", None)
    values["channel"] = channel
    values["cameras"] = {}
    values["allow_unverified_commanding"] = False
    values["auto_configure_master_slave_on_connect"] = False
    values["auto_enable_on_connect"] = False
    values["auto_configure_mode_on_connect"] = False
    values["transport_backend"] = "piper_sdk"
    return SonglingFollowerConfig(id=app_id, calibration_dir=calibration_dir, **values)


def _build_arm_handles(cfg: ManualControlConfig) -> list[ArmHandle]:
    robot_cfg = cfg.robot
    if robot_cfg is None:
        raise ValueError("Config is missing a robot section.")

    left_arm_cfg = getattr(robot_cfg, "left_arm_config", None)
    right_arm_cfg = getattr(robot_cfg, "right_arm_config", None)
    if left_arm_cfg is None or right_arm_cfg is None:
        raise ValueError("This tool expects a bimanual Songling follower config with left/right arm sections.")

    calibration_dir_value = getattr(robot_cfg, "calibration_dir", None)
    calibration_dir = Path(calibration_dir_value) if calibration_dir_value is not None else None
    left = SonglingFollower(
        _make_arm_config(app_id=f"{robot_cfg.id}_left", calibration_dir=calibration_dir, raw_arm_cfg=left_arm_cfg)
    )
    right = SonglingFollower(
        _make_arm_config(app_id=f"{robot_cfg.id}_right", calibration_dir=calibration_dir, raw_arm_cfg=right_arm_cfg)
    )
    return [
        ArmHandle(side="left", follower=left),
        ArmHandle(side="right", follower=right),
    ]


def _score_arm_feedback(arm: ArmHandle) -> tuple[int, int, float]:
    arm.follower.bus.poll()
    positions = arm.follower.bus.get_positions(poll=False)
    seen_count = sum(1 for name in JOINT_NAMES if arm.follower.bus.joint_position_seen.get(name, False))
    status_count = len(arm.follower.bus.status)
    motion = sum(abs(float(v)) for v in positions.values())
    return seen_count, status_count, motion


def _maybe_swap_arm_handles(handles: list[ArmHandle]) -> tuple[list[ArmHandle], str | None]:
    if len(handles) != 2:
        return handles, None

    left_handle = next((arm for arm in handles if arm.side == "left"), None)
    right_handle = next((arm for arm in handles if arm.side == "right"), None)
    if left_handle is None or right_handle is None:
        return handles, None

    left_score = _score_arm_feedback(left_handle)
    right_score = _score_arm_feedback(right_handle)
    left_motion = left_score[2]
    right_motion = right_score[2]

    # Heuristic: if the configured left side looks completely idle while the configured
    # right side has strong feedback, the two CAN ports are likely swapped.
    if left_motion < 0.5 and right_motion > 1.0:
        swapped = [
            ArmHandle(side="left", follower=right_handle.follower),
            ArmHandle(side="right", follower=left_handle.follower),
        ]
        return swapped, (
            f"检测到左右臂 CAN 口可能接反，已自动交换 UI 映射："
            f"left->{right_handle.follower.config.channel}, right->{left_handle.follower.config.channel}"
        )

    return handles, None


class ManualJointControlWindow(QMainWindow):
    def __init__(self, cfg: ManualControlConfig, poll_ms: int, window_title: str, config_path: Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self.poll_ms = max(int(poll_ms), 50)
        self.setWindowTitle(window_title)
        self.setMinimumSize(1080, 620)
        self.resize(1260, 760)

        app = QApplication.instance()
        self.ui_font_family = app.font().family() if app is not None else ""
        self.actual_font_family = self.ui_font_family
        self.ui_font = app.font() if app is not None else QFont()
        self.log_font = QFont(self.ui_font)
        self.log_font.setPointSize(LOG_FONT_SIZE)

        self.arm_handles = _build_arm_handles(cfg)
        self.arm_mapping_note = None
        self.runtime_joint_limits: dict[str, dict[str, tuple[float, float]]] = {
            arm.side: dict(arm.follower.config.joint_limits) for arm in self.arm_handles
        }
        self.current_values: dict[str, dict[str, float]] = {arm.side: {} for arm in self.arm_handles}
        self.current_labels: dict[str, dict[str, QLabel]] = {arm.side: {} for arm in self.arm_handles}
        self.target_spinboxes: dict[str, dict[str, QDoubleSpinBox]] = {arm.side: {} for arm in self.arm_handles}
        self.slider_widgets: dict[str, dict[str, QSlider]] = {arm.side: {} for arm in self.arm_handles}
        self.commanded_targets: dict[str, dict[str, float] | None] = {arm.side: None for arm in self.arm_handles}

        self.motors_enabled = False
        self.connected = False
        self._targets_initialized_by_side: dict[str, bool] = {arm.side: False for arm in self.arm_handles}
        self._syncing_targets = False
        self._hold_send_in_progress = False
        self._manual_send_in_progress = False
        self._hold_backoff_until = 0.0
        self._last_hold_error: str | None = None
        self._missing_joint_feedback_warned: dict[str, bool] = {arm.side: False for arm in self.arm_handles}
        self._manual_send_fallback_warned: dict[str, bool] = {arm.side: False for arm in self.arm_handles}

        self.status_label = QLabel(_ui_text("Connecting...", "\u6b63\u5728\u8fde\u63a5..."))
        self.log_widget: QTextEdit | None = None
        self.can_available_label: QLabel | None = None
        self.can_config_labels: dict[str, QLabel] = {}
        self.can_status_labels: dict[str, QLabel] = {}
        self.can_result_widget: QTextEdit | None = None

        self._build_ui()
        self._connect_and_initialize()

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(self.poll_ms)
        self.poll_timer.timeout.connect(self._poll_current_positions)
        self.poll_timer.start()

        self.hold_timer = QTimer(self)
        self.hold_timer.setInterval(HOLD_TIMER_MS)
        self.hold_timer.timeout.connect(self._maintain_hold_targets)
        self.hold_timer.start()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        self.setCentralWidget(central)

        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(38)
        self.status_label.setStyleSheet(
            "QLabel {"
            "background: #f5f1e8;"
            "border: 1px solid #d7c8a9;"
            "border-radius: 8px;"
            "padding: 8px 10px;"
            "font-weight: 600;"
            "}"
        )

        status_group = QGroupBox(_ui_text("Runtime", "\u8fd0\u884c\u72b6\u6001"), central)
        status_group.setStyleSheet(GROUP_BOX_STYLE)
        status_layout = QHBoxLayout(status_group)
        status_layout.setContentsMargins(10, 10, 10, 10)
        status_layout.setSpacing(10)
        self.auto_poll_checkbox = QCheckBox(_ui_text("Auto Poll", "\u81ea\u52a8\u8f6e\u8be2"), status_group)
        self.auto_poll_checkbox.setChecked(True)
        status_layout.addWidget(self.auto_poll_checkbox)
        status_layout.addWidget(self.status_label, stretch=1)
        root.addWidget(status_group)

        self.page_tabs = QTabWidget(central)
        root.addWidget(self.page_tabs, stretch=1)

        control_page = QWidget(self.page_tabs)
        control_root = QVBoxLayout(control_page)
        control_root.setContentsMargins(0, 0, 0, 0)
        control_root.setSpacing(10)
        self.page_tabs.addTab(control_page, _ui_text("Console", "\u63a7\u5236\u53f0"))

        can_page = QWidget(self.page_tabs)
        can_root = QVBoxLayout(can_page)
        can_root.setContentsMargins(0, 0, 0, 0)
        can_root.setSpacing(10)
        self.page_tabs.addTab(can_page, _ui_text("CAN Config", "CAN \u914d\u7f6e"))

        action_widget = QWidget(control_page)
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(10)
        control_root.addWidget(action_widget)

        sync_group = self._build_action_group(
            _ui_text("State", "\u72b6\u6001\u4e0e\u540c\u6b65"),
            (
                (_ui_text("Refresh", "\u5237\u65b0\u72b6\u6001"), self.refresh_positions, "neutral"),
                (_ui_text("Target -> Current", "\u76ee\u6807\u56de\u5f53\u524d\u59ff\u6001"), self.sync_targets_to_current, "neutral"),
            ),
            parent=action_widget,
        )
        action_layout.addWidget(sync_group, 1)

        zero_group = self._build_action_group(
            _ui_text("Poses", "\u59ff\u6001\u4e0e\u56de\u96f6"),
            (
                (_ui_text("Left Pre-Disable Here", "\u5de6\u81c2\u5f53\u524d\u8bbe\u4e3a\u5931\u80fd\u524d\u59ff\u6001"), lambda: self.set_current_pose_as_pre_disable("left"), "neutral"),
                (_ui_text("Right Pre-Disable Here", "\u53f3\u81c2\u5f53\u524d\u8bbe\u4e3a\u5931\u80fd\u524d\u59ff\u6001"), lambda: self.set_current_pose_as_pre_disable("right"), "neutral"),
                (_ui_text("Left -> Zero Pose", "\u5de6\u81c2\u56de\u96f6\u4f4d\u59ff\u6001"), lambda: self.move_to_zero_pose("left"), "neutral"),
                (_ui_text("Right -> Zero Pose", "\u53f3\u81c2\u56de\u96f6\u4f4d\u59ff\u6001"), lambda: self.move_to_zero_pose("right"), "neutral"),
                (_ui_text("Left -> Pre-Disable", "\u5de6\u81c2\u56de\u5931\u80fd\u524d\u59ff\u6001"), lambda: self.move_to_pre_disable_pose("left"), "neutral"),
                (_ui_text("Right -> Pre-Disable", "\u53f3\u81c2\u56de\u5931\u80fd\u524d\u59ff\u6001"), lambda: self.move_to_pre_disable_pose("right"), "neutral"),
                (_ui_text("Both -> Zero Pose", "\u53cc\u81c2\u56de\u96f6\u4f4d\u59ff\u6001"), lambda: self.move_to_zero_pose(None), "neutral"),
                (_ui_text("Both -> Pre-Disable", "\u53cc\u81c2\u56de\u5931\u80fd\u524d\u59ff\u6001"), lambda: self.move_to_pre_disable_pose(None), "neutral"),
            ),
            parent=action_widget,
            columns=2,
        )
        action_layout.addWidget(zero_group, 2)

        power_group = self._build_action_group(
            _ui_text("Power", "\u7535\u673a"),
            (
                (_ui_text("Enable Motors", "\u7535\u673a\u4f7f\u80fd"), self.enable_and_configure, "warm"),
                (_ui_text("Disable Motors", "\u7535\u673a\u5931\u80fd"), self.disable_motors, "danger"),
            ),
            parent=action_widget,
        )
        action_layout.addWidget(power_group, 1)

        send_group = self._build_action_group(
            _ui_text("Send", "\u53d1\u9001"),
            (
                (_ui_text("Send Left", "\u53d1\u9001\u5de6\u81c2"), lambda: self.send_targets(selected_side="left"), "primary"),
                (_ui_text("Send Right", "\u53d1\u9001\u53f3\u81c2"), lambda: self.send_targets(selected_side="right"), "primary"),
                (_ui_text("Send Both", "\u53d1\u9001\u53cc\u81c2"), lambda: self.send_targets(selected_side=None), "primary"),
            ),
            parent=action_widget,
        )
        action_layout.addWidget(send_group, 1)

        options_group = QGroupBox(_ui_text("Send Options", "\u53d1\u9001\u9009\u9879"), control_page)
        options_group.setStyleSheet(GROUP_BOX_STYLE)
        options_layout = QGridLayout(options_group)
        options_layout.setContentsMargins(10, 10, 10, 10)
        options_layout.setHorizontalSpacing(10)
        options_layout.setVerticalSpacing(8)
        control_root.addWidget(options_group)

        options_layout.addWidget(QLabel(_ui_text("Joint Step", "\u5173\u8282\u6b65\u957f"), options_group), 0, 0)
        self.joint_step_spin = QDoubleSpinBox(options_group)
        self.joint_step_spin.setRange(0.1, 30.0)
        self.joint_step_spin.setDecimals(1)
        self.joint_step_spin.setSingleStep(0.1)
        self.joint_step_spin.setValue(1.0)
        options_layout.addWidget(self.joint_step_spin, 0, 1)

        options_layout.addWidget(QLabel(_ui_text("Gripper Step", "\u5939\u722a\u6b65\u957f"), options_group), 0, 2)
        self.gripper_step_spin = QDoubleSpinBox(options_group)
        self.gripper_step_spin.setRange(0.1, 20.0)
        self.gripper_step_spin.setDecimals(1)
        self.gripper_step_spin.setSingleStep(0.1)
        self.gripper_step_spin.setValue(2.0)
        options_layout.addWidget(self.gripper_step_spin, 0, 3)

        self.auto_enable_checkbox = QCheckBox(
            _ui_text("Auto Enable", "\u53d1\u9001\u524d\u81ea\u52a8\u4f7f\u80fd"),
            options_group,
        )
        self.auto_enable_checkbox.setChecked(False)
        self.auto_enable_checkbox.setEnabled(False)
        self.auto_enable_checkbox.setToolTip(
            _ui_text(
                "For safety, motion stays locked after disable until you explicitly enable again.",
                "\u4e3a\u4e86\u5b89\u5168\uff0c\u5931\u80fd\u540e\u4e0d\u5141\u8bb8\u81ea\u52a8\u8865\u4f7f\u80fd\uff0c\u9700\u624b\u52a8\u518d\u6b21\u70b9\u51fb\u201c\u7535\u673a\u4f7f\u80fd\u201d\u3002",
            )
        )
        options_layout.addWidget(self.auto_enable_checkbox, 0, 4)

        self.hold_position_checkbox = QCheckBox(
            _ui_text("Hold Current Target", "\u6301\u7eed\u4fdd\u6301\u5f53\u524d\u76ee\u6807"),
            options_group,
        )
        self.hold_position_checkbox.setChecked(True)
        options_layout.addWidget(self.hold_position_checkbox, 0, 5)

        hint = QLabel(
            _ui_text(
                "Recommended: Enable -> Target->Current -> Small edits -> Send",
                "\u63a8\u8350\u6d41\u7a0b\uff1a\u4f7f\u80fd\uff08\u4f1a\u81ea\u52a8\u56de\u96f6\u4f4d\u59ff\u6001\uff09 -> \u76ee\u6807\u56de\u5f53\u524d -> \u5c0f\u6b65\u4fee\u6539 -> \u53d1\u9001\u3002",
            ),
            options_group,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6c5a3b;")
        options_layout.addWidget(hint, 1, 0, 1, 6)
        options_layout.setColumnStretch(5, 1)

        self.tabs = QTabWidget(control_page)
        self.tabs.setMinimumHeight(320)
        control_root.addWidget(self.tabs, stretch=1)
        for arm in self.arm_handles:
            tab_scroll = QScrollArea(self.tabs)
            tab_scroll.setWidgetResizable(True)
            tab_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            tab_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            tab_content = QWidget(tab_scroll)
            tab_scroll.setWidget(tab_content)
            self.tabs.addTab(tab_scroll, _side_label(arm.side))
            self._build_arm_panel(tab_content, arm.side, arm.follower)

        can_intro_group = QGroupBox(_ui_text("Detect CAN", "CAN \u81ea\u52a8\u8bc6\u522b"), can_page)
        can_intro_group.setStyleSheet(GROUP_BOX_STYLE)
        can_intro_layout = QVBoxLayout(can_intro_group)
        can_intro = QLabel(
            _ui_text(
                "Check reads config and visible canX list only. Detect does not require live arm feedback; keep only one canX visible for unambiguous detection.",
                "\u201c\u68c0\u67e5\u201d\u53ea\u8bfb\u53d6\u914d\u7f6e\u6587\u4ef6\u4e0e\u5f53\u524d\u7cfb\u7edf\u53ef\u89c1\u7684 canX \u5217\u8868\u3002\u201c\u8bc6\u522b\u201d\u4e0d\u4f9d\u8d56\u673a\u68b0\u81c2\u5b9e\u65f6\u53cd\u9988\uff1b\u4e3a\u4e86\u907f\u514d\u6b67\u4e49\uff0c\u8bc6\u522b\u65f6\u8bf7\u5c3d\u91cf\u53ea\u4fdd\u7559\u4e00\u4e2a\u53ef\u89c1 canX\u3002",
            ),
            can_intro_group,
        )
        can_intro.setWordWrap(True)
        can_intro.setStyleSheet("color: #6c5a3b;")
        can_intro_layout.addWidget(can_intro)
        can_root.addWidget(can_intro_group)

        can_status_group = QGroupBox(_ui_text("CAN Status", "CAN \u72b6\u6001"), can_page)
        can_status_group.setStyleSheet(GROUP_BOX_STYLE)
        can_status_layout = QGridLayout(can_status_group)
        can_status_layout.setContentsMargins(10, 10, 10, 10)
        can_status_layout.setHorizontalSpacing(10)
        can_status_layout.setVerticalSpacing(8)
        can_status_layout.addWidget(QLabel(_ui_text("Available", "\u53ef\u89c1\u63a5\u53e3"), can_status_group), 0, 0)
        self.can_available_label = QLabel("--", can_status_group)
        self.can_available_label.setWordWrap(True)
        can_status_layout.addWidget(self.can_available_label, 0, 1, 1, 3)
        for row_idx, side in enumerate(("left", "right"), start=1):
            can_status_layout.addWidget(QLabel(_side_label(side), can_status_group), row_idx, 0)
            config_label = QLabel("--", can_status_group)
            status_label = QLabel("--", can_status_group)
            config_label.setWordWrap(True)
            status_label.setWordWrap(True)
            can_status_layout.addWidget(config_label, row_idx, 1)
            can_status_layout.addWidget(status_label, row_idx, 2, 1, 2)
            self.can_config_labels[side] = config_label
            self.can_status_labels[side] = status_label
        can_status_layout.setColumnStretch(2, 1)
        can_root.addWidget(can_status_group)

        can_group = self._build_action_group(
            _ui_text("CAN", "CAN \u914d\u7f6e"),
            (
                (_ui_text("Setup CAN", "\u4e00\u952e Up CAN"), self.setup_can_interfaces, "warm"),
                (_ui_text("Check Left CAN", "\u68c0\u67e5\u5de6\u81c2 CAN"), lambda: self.check_can_for_side("left"), "neutral"),
                (_ui_text("Check Right CAN", "\u68c0\u67e5\u53f3\u81c2 CAN"), lambda: self.check_can_for_side("right"), "neutral"),
                (_ui_text("Check Both CAN", "\u68c0\u67e5\u53cc\u81c2 CAN"), self.check_all_can, "neutral"),
                (_ui_text("Detect Left CAN", "\u8bc6\u522b\u5de6\u81c2 CAN"), lambda: self.detect_can_for_side("left"), "neutral"),
                (_ui_text("Detect Right CAN", "\u8bc6\u522b\u53f3\u81c2 CAN"), lambda: self.detect_can_for_side("right"), "neutral"),
            ),
            parent=can_page,
            columns=2,
        )
        can_group.setToolTip(
            _ui_text(
                "Connect only one arm during detection. The detected interface will be written to the config, then the app restarts.",
                "\u8bc6\u522b\u65f6\u53ea\u63a5\u4e00\u6761\u673a\u68b0\u81c2\u3002\u68c0\u6d4b\u5230\u7684 CAN \u53e3\u4f1a\u5199\u5165\u914d\u7f6e\u6587\u4ef6\uff0c\u7136\u540e\u8f6f\u4ef6\u81ea\u52a8\u91cd\u542f\u5237\u65b0\u3002",
            )
        )
        can_root.addWidget(can_group)

        can_result_group = QGroupBox(_ui_text("CAN Result", "CAN \u68c0\u67e5\u7ed3\u679c"), can_page)
        can_result_group.setStyleSheet(GROUP_BOX_STYLE)
        can_result_layout = QVBoxLayout(can_result_group)
        self.can_result_widget = QTextEdit(can_result_group)
        self.can_result_widget.setReadOnly(True)
        self.can_result_widget.setFont(self.log_font)
        self.can_result_widget.setMinimumHeight(140)
        can_result_layout.addWidget(self.can_result_widget)
        can_root.addWidget(can_result_group, stretch=1)
        can_root.addStretch(1)

        log_group = QGroupBox(_ui_text("Log", "\u65e5\u5fd7"), central)
        log_group.setStyleSheet(GROUP_BOX_STYLE)
        log_layout = QVBoxLayout(log_group)
        self.log_widget = QTextEdit(log_group)
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(self.log_font)
        self.log_widget.setMinimumHeight(80)
        log_layout.addWidget(self.log_widget)
        root.addWidget(log_group, stretch=0)
        self._update_can_page_status()

    def _build_action_group(
        self,
        title: str,
        actions: tuple[tuple[str, Any, str], ...],
        *,
        parent: QWidget,
        columns: int = 1,
    ) -> QGroupBox:
        group = QGroupBox(title, parent)
        group.setStyleSheet(GROUP_BOX_STYLE)
        layout = QGridLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        style_map = {
            "neutral": BUTTON_STYLE_NEUTRAL,
            "warm": BUTTON_STYLE_WARM,
            "primary": BUTTON_STYLE_PRIMARY,
            "danger": BUTTON_STYLE_DANGER,
        }
        for idx, (text, slot, role) in enumerate(actions):
            button = QPushButton(text, group)
            button.setMinimumHeight(32)
            button.setStyleSheet(style_map.get(role, BUTTON_STYLE_NEUTRAL))
            button.clicked.connect(slot)
            row = idx // columns
            col = idx % columns
            layout.addWidget(button, row, col)
        for col in range(columns):
            layout.setColumnStretch(col, 1)
        return group

    def _build_arm_panel(self, parent: QWidget, side: str, follower: SonglingFollower) -> None:
        layout = QGridLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.addWidget(QLabel(_ui_text("Joint", "\u5173\u8282"), parent), 0, 0)
        layout.addWidget(QLabel(_ui_text("Current", "\u5f53\u524d\u503c"), parent), 0, 1)
        layout.addWidget(QLabel(_ui_text("Target", "\u76ee\u6807\u503c"), parent), 0, 2)
        layout.addWidget(QLabel(_ui_text("Jog", "\u70b9\u52a8"), parent), 0, 4)

        for row_idx, joint_name in enumerate(JOINT_NAMES, start=1):
            limits = self.runtime_joint_limits[side][joint_name]
            joint_label = QLabel(_joint_label(joint_name), parent)
            current_label = QLabel("--", parent)
            target_spin = QDoubleSpinBox(parent)
            target_spin.setRange(float(limits[0]), float(limits[1]))
            target_spin.setDecimals(3)
            target_spin.setSingleStep(0.1)
            target_spin.valueChanged.connect(partial(self._on_target_spin_changed, side, joint_name))

            slider = QSlider(Qt.Horizontal, parent)
            slider.setRange(int(round(limits[0] * SLIDER_SCALE)), int(round(limits[1] * SLIDER_SCALE)))
            slider.setSingleStep(1)
            slider.setPageStep(5)
            slider.valueChanged.connect(partial(self._on_target_slider_changed, side, joint_name))

            minus_button = QPushButton("-", parent)
            minus_button.setFixedWidth(36)
            minus_button.clicked.connect(partial(self.jog_target, side, joint_name, -1))

            plus_button = QPushButton("+", parent)
            plus_button.setFixedWidth(36)
            plus_button.clicked.connect(partial(self.jog_target, side, joint_name, +1))

            layout.addWidget(joint_label, row_idx, 0)
            layout.addWidget(current_label, row_idx, 1)
            layout.addWidget(target_spin, row_idx, 2)
            layout.addWidget(slider, row_idx, 3)
            layout.addWidget(minus_button, row_idx, 4)
            layout.addWidget(plus_button, row_idx, 5)

            self.current_labels[side][joint_name] = current_label
            self.target_spinboxes[side][joint_name] = target_spin
            self.slider_widgets[side][joint_name] = slider

        layout.setColumnStretch(3, 1)
        layout.setRowStretch(len(JOINT_NAMES) + 1, 1)

    def _expand_joint_limit_for_value(self, side: str, joint_name: str, value: float) -> None:
        current_min, current_max = self.runtime_joint_limits[side][joint_name]
        margin = 2.0 if joint_name != "gripper" else 1.0
        new_min = min(current_min, float(value) - margin)
        new_max = max(current_max, float(value) + margin)
        if math.isclose(new_min, current_min, abs_tol=1e-9) and math.isclose(new_max, current_max, abs_tol=1e-9):
            return

        self.runtime_joint_limits[side][joint_name] = (new_min, new_max)
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        handle.follower.config.joint_limits[joint_name] = (new_min, new_max)

        spin = self.target_spinboxes[side][joint_name]
        slider = self.slider_widgets[side][joint_name]
        spin.blockSignals(True)
        spin.setRange(float(new_min), float(new_max))
        spin.blockSignals(False)
        slider.blockSignals(True)
        slider.setRange(int(round(new_min * SLIDER_SCALE)), int(round(new_max * SLIDER_SCALE)))
        slider.blockSignals(False)

    def _ensure_positions_within_limits(self, side: str, positions: dict[str, float]) -> None:
        for joint_name, value in positions.items():
            self._expand_joint_limit_for_value(side, joint_name, float(value))

    def _on_target_spin_changed(self, side: str, joint_name: str, value: float) -> None:
        if self._syncing_targets:
            return
        slider = self.slider_widgets[side][joint_name]
        slider.blockSignals(True)
        slider.setValue(int(round(float(value) * SLIDER_SCALE)))
        slider.blockSignals(False)

    def _on_target_slider_changed(self, side: str, joint_name: str, raw_value: int) -> None:
        if self._syncing_targets:
            return
        spin = self.target_spinboxes[side][joint_name]
        spin.blockSignals(True)
        spin.setValue(float(raw_value) / SLIDER_SCALE)
        spin.blockSignals(False)

    def _set_target_value(self, side: str, joint_name: str, value: float) -> None:
        self._syncing_targets = True
        try:
            spin = self.target_spinboxes[side][joint_name]
            slider = self.slider_widgets[side][joint_name]
            spin.setValue(float(value))
            slider.setValue(int(round(float(value) * SLIDER_SCALE)))
        finally:
            self._syncing_targets = False

    def _capture_current_pose_targets(
        self,
        *,
        selected_side: str | None = None,
        update_ui_targets: bool,
    ) -> dict[str, dict[str, float]]:
        captured: dict[str, dict[str, float]] = {}
        selected = [arm for arm in self.arm_handles if selected_side is None or arm.side == selected_side]
        for arm in selected:
            if not self._arm_has_feedback(arm.side):
                self.commanded_targets[arm.side] = None
                self._log(
                    f"{_side_label(arm.side)} {_ui_text('has no feedback; skipping current-pose capture', '\u672a\u6536\u5230\u53cd\u9988\uff0c\u8df3\u8fc7\u5f53\u524d\u59ff\u6001\u91c7\u96c6')}"
                )
                continue
            positions = self._arm_positions(arm.side)
            self._ensure_positions_within_limits(arm.side, positions)
            self.current_values[arm.side] = positions
            for joint_name, value in positions.items():
                self.current_labels[arm.side][joint_name].setText(f"{value:.3f}")
                if update_ui_targets:
                    self._set_target_value(arm.side, joint_name, value)
            captured[arm.side] = dict(positions)
        if captured:
            for side in captured:
                self._targets_initialized_by_side[side] = True
        return captured

    def _position_dict_to_action(self, positions: dict[str, float]) -> dict[str, float]:
        return {f"{joint_name}.pos": float(positions[joint_name]) for joint_name in JOINT_NAMES}

    def _action_to_positions(self, action: dict[str, float]) -> dict[str, float]:
        return {joint_name: float(action[f"{joint_name}.pos"]) for joint_name in JOINT_NAMES}

    def _sent_action_to_positions(self, sent: dict[str, float]) -> dict[str, float]:
        return self._action_to_positions(sent)

    def _arm_motion_abs_sum(self, side: str) -> float:
        return sum(abs(float(value)) for value in self.current_values.get(side, {}).values())

    def _log_runtime_arm_binding(self, arm: ArmHandle) -> None:
        camera_names = ", ".join(arm.follower.cameras) if arm.follower.cameras else "none"
        self._log(
            f"{_side_label(arm.side)} UI映射："
            f"cfg.side={arm.follower.config.side}, "
            f"CAN={arm.follower.config.channel}, "
            f"installation_pos=0x{arm.follower.bus.installation_pos():02x}, "
            f"cameras={camera_names}"
        )

    def _log_asymmetric_feedback_warning(self) -> None:
        if len(self.arm_handles) != 2:
            return

        left_motion = self._arm_motion_abs_sum("left")
        right_motion = self._arm_motion_abs_sum("right")
        if left_motion < 1.0 and right_motion < 1.0:
            return
        if abs(left_motion - right_motion) < 5.0:
            return

        dominant_side = "left" if left_motion > right_motion else "right"
        quiet_side = "right" if dominant_side == "left" else "left"
        self._log(
            f"警告：{_side_label(dominant_side)}反馈明显大于{_side_label(quiet_side)}"
            f"（{left_motion:.3f} vs {right_motion:.3f}）。"
            "如果点击左/右发送时实际运动边不一致，请先检查启动命令里的 CAN 口覆盖或配置文件映射，再继续发送。"
        )

    def _arm_ready_for_command(self, arm: ArmHandle) -> tuple[bool, str | None]:
        arm.follower.bus.poll()
        if getattr(arm.follower.bus, "status_feedback_valid", False):
            expected_ctrl_mode = int(getattr(arm.follower.config, "ctrl_mode", -1))
            actual_ctrl_mode = int(arm.follower.bus.status.get("ctrl_mode", -1))
            if expected_ctrl_mode >= 0 and actual_ctrl_mode != expected_ctrl_mode:
                return (
                    False,
                    f"控制模式未进入预期值（expected=0x{expected_ctrl_mode:x}, actual=0x{actual_ctrl_mode:x}）",
                )
        elif getattr(arm.follower.bus, "mode_command_valid", False):
            expected_ctrl_mode = int(getattr(arm.follower.config, "ctrl_mode", -1))
            actual_ctrl_mode = int(getattr(arm.follower.bus, "mode_command", {}).get("ctrl_mode", -1))
            if expected_ctrl_mode >= 0 and actual_ctrl_mode != expected_ctrl_mode:
                return (
                    False,
                    f"`0x151` 模式命令未进入预期值（expected=0x{expected_ctrl_mode:x}, actual=0x{actual_ctrl_mode:x}）",
                )
        else:
            return False, MISSING_STATUS_OR_MODE_REASON

        enable_status = arm.follower.bus.get_driver_enable_status()
        known_states = [state for state in enable_status.values() if state is not None]
        if known_states and not all(bool(state) for state in known_states):
            return False, "仍有驱动未使能"

        return True, None

    def _arm_ready_for_manual_send(self, arm: ArmHandle) -> tuple[bool, str | None, str | None]:
        ready, reason = self._arm_ready_for_command(arm)
        if ready:
            self._manual_send_fallback_warned[arm.side] = False
            return True, None, None

        note = _manual_send_fallback_note(arm, reason)
        if note is None:
            return False, reason, None
        return True, None, note

    def _hold_current_pose(
        self,
        *,
        log_message: bool,
        selected_sides: set[str] | None = None,
    ) -> None:
        selected_side = None
        if selected_sides is not None and len(selected_sides) == 1:
            selected_side = next(iter(selected_sides))
        captured = self._capture_current_pose_targets(selected_side=selected_side, update_ui_targets=True)
        for arm in self.arm_handles:
            if selected_sides is not None and arm.side not in selected_sides:
                self.commanded_targets[arm.side] = None
                continue
            positions = captured.get(arm.side)
            if positions is None:
                continue
            requested_action = self._position_dict_to_action(positions)
            sent = arm.follower.hold_position(positions)
            self.commanded_targets[arm.side] = self._sent_action_to_positions(sent)
            self._log_arm_status(
                arm,
                _ui_text("status after hold", "\u4fdd\u6301\u540e\u72b6\u6001"),
                include_diagnostics=True,
                requested=requested_action,
                sent=sent,
            )
        if log_message:
            self._log(
                _ui_text(
                    "Current pose hold command sent to both arms.",
                    "\u5df2\u5411\u53cc\u81c2\u53d1\u9001\u201c\u4fdd\u6301\u5f53\u524d\u59ff\u6001\u201d\u6307\u4ee4\u3002",
                )
            )

    def _send_hold_positions_for_arm(
        self,
        arm: ArmHandle,
        positions: dict[str, float],
        *,
        log_prefix: str,
    ) -> None:
        requested_action = self._position_dict_to_action(positions)
        sent = arm.follower.hold_position(positions)
        self.commanded_targets[arm.side] = self._sent_action_to_positions(sent)
        self._log_arm_status(
            arm,
            log_prefix,
            include_diagnostics=True,
            requested=requested_action,
            sent=sent,
        )

    def _preload_hold_positions_for_arm(self, arm: ArmHandle, positions: dict[str, float]) -> None:
        for _ in range(ENABLE_PRELOAD_REPEAT):
            arm.follower.hold_position(positions)
            if ENABLE_PRELOAD_SETTLE_S > 0:
                time.sleep(ENABLE_PRELOAD_SETTLE_S)

    def _maintain_hold_targets(self) -> None:
        if not self.connected or not self.motors_enabled:
            return
        if not self.hold_position_checkbox.isChecked():
            return
        if self._manual_send_in_progress:
            return
        if self._hold_send_in_progress:
            return
        if time.monotonic() < self._hold_backoff_until:
            return

        selected = [arm for arm in self.arm_handles if self.commanded_targets.get(arm.side) is not None]
        if not selected:
            return

        self._hold_send_in_progress = True
        try:
            for arm in selected:
                ready, _ = self._arm_ready_for_command(arm)
                if not ready:
                    self.commanded_targets[arm.side] = None
                    continue
                if not self._arm_has_feedback(arm.side):
                    self.commanded_targets[arm.side] = None
                    continue
                commanded = self.commanded_targets.get(arm.side)
                if commanded is None:
                    continue
                arm.follower.send_action(self._position_dict_to_action(commanded))
                self.commanded_targets[arm.side] = dict(commanded)
            self._last_hold_error = None
        except Exception as exc:
            message = str(exc)
            if message != self._last_hold_error:
                self._log(
                    f"{_ui_text('Hold failed', '\u4fdd\u6301\u5931\u8d25')}: {message}"
                )
                for arm in selected:
                    self._log_arm_status(
                        arm,
                        _ui_text("status on hold failure", "\u4fdd\u6301\u5931\u8d25\u65f6\u72b6\u6001"),
                        include_diagnostics=True,
                    )
                self._last_hold_error = message
            if "105" in message or "\u6ca1\u6709\u53ef\u7528\u7684\u7f13\u51b2\u533a\u7a7a\u95f4" in message:
                self._hold_backoff_until = time.monotonic() + (HOLD_BACKOFF_MS / 1000.0)
        finally:
            self._hold_send_in_progress = False

    def _connect_and_initialize(self) -> None:
        try:
            for arm in self.arm_handles:
                arm.follower.connect(calibrate=False)
            self.connected = True
            self.status_label.setText(
                _ui_text(
                    "Connected (read-only until enable and send)",
                    "\u5df2\u8fde\u63a5\uff08\u542f\u52a8\u53ea\u8bfb\uff0c\u9700\u4f7f\u80fd\u540e\u518d\u53d1\u9001\uff09",
                )
            )
            self._log(
                _ui_text(
                    "Connected to CAN; camera startup is skipped in this tool.",
                    "\u5df2\u8fde\u63a5 CAN\uff1b\u8fd9\u4e2a\u5de5\u5177\u542f\u52a8\u65f6\u4f1a\u8df3\u8fc7\u76f8\u673a\u521d\u59cb\u5316\u3002",
                )
            )
            self._log(
                "当前上位机流程按“直控两个从臂和夹爪”设计，只检查下发到从臂的目标是否正确，不会去重新配置主从关系。"
            )
            self._log(f"{_ui_text('UI font', '\u754c\u9762\u5b57\u4f53')}: {self.ui_font_family}")
            self._log(f"{_ui_text('Actual widget font', '\u63a7\u4ef6\u5b9e\u9645\u5b57\u4f53')}: {QApplication.font().family()}")
            self._log(f"{_ui_text('Qt style', 'Qt \u6837\u5f0f')}: {QApplication.style().objectName()}")
            self._log("当前严格遵循配置中的左右臂映射，不再根据启动反馈自动交换 UI 左右。")
            for arm in self.arm_handles:
                self._log(f"{_side_label(arm.side)} CAN: {arm.follower.config.channel}")
                self._log_runtime_arm_binding(arm)
                self._log_command_source_warning(arm)
            if self.arm_mapping_note:
                self._log(self.arm_mapping_note)
            for arm in self.arm_handles:
                self._log(
                    f"{_side_label(arm.side)} "
                    f"{_ui_text('camera startup', '\u76f8\u673a\u542f\u52a8')}: "
                    f"{_ui_text('skipped', '\u5df2\u8df3\u8fc7')}"
                )
                self._log_arm_status(arm, _ui_text("status after connect", "\u8fde\u63a5\u540e\u72b6\u6001"), include_diagnostics=True)
            self.refresh_positions(sync_targets=True)
            self._log_asymmetric_feedback_warning()
        except Exception as exc:
            self._disconnect_arms()
            self.connected = False
            self.motors_enabled = False
            self.status_label.setText(_ui_text("Connect failed", "\u8fde\u63a5\u5931\u8d25"))
            self._log(f"{_ui_text('Connect failed', '\u8fde\u63a5\u5931\u8d25')}: {exc}")
            self._log("你现在仍然可以使用“识别左臂 CAN / 识别右臂 CAN”按钮写回配置，程序会自动重启后再重新连接。")
            QMessageBox.warning(
                self,
                _ui_text("Connect failed", "\u8fde\u63a5\u5931\u8d25"),
                f"{exc}\n\n可先使用界面里的 CAN 识别按钮更新配置，软件会自动重启后重新连接。",
            )

    def _log(self, message: str) -> None:
        if self.log_widget is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.log_widget.append(f"[{timestamp}] {message}")

    def _arm_positions(self, side: str) -> dict[str, float]:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        positions = dict(handle.follower.bus.get_positions(poll=True))
        if (
            not handle.follower.bus.joint_position_seen.get("gripper", False)
            and handle.follower.bus.commanded_position_seen.get("gripper", False)
        ):
            commanded = handle.follower.bus.get_commanded_positions(poll=False)
            if "gripper" in commanded:
                positions["gripper"] = float(commanded["gripper"])
        return positions

    def _arm_target_positions(self, side: str) -> dict[str, float]:
        return {joint_name: float(self.target_spinboxes[side][joint_name].value()) for joint_name in JOINT_NAMES}

    def _arm_command_positions(self, side: str) -> dict[str, float] | None:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        handle.follower.bus.poll()
        if not getattr(handle.follower.bus, "command_feedback_valid", False):
            return None
        return handle.follower.bus.get_commanded_positions(poll=False)

    def _format_joint_values(self, values: dict[str, float] | None) -> str:
        if not values:
            return "--"
        parts = []
        for joint_name in JOINT_NAMES:
            if joint_name in values:
                parts.append(f"{joint_name}={float(values[joint_name]):.3f}")
            else:
                parts.append(f"{joint_name}=--")
        return ", ".join(parts)

    def _format_position_mismatch(
        self,
        expected: dict[str, float] | None,
        actual: dict[str, float] | None,
        *,
        atol: float = 1e-3,
    ) -> str | None:
        if not expected or not actual:
            return None
        changed: list[str] = []
        for joint_name in JOINT_NAMES:
            if joint_name not in expected or joint_name not in actual:
                changed.append(f"{joint_name}=--")
                continue
            expected_value = float(expected[joint_name])
            actual_value = float(actual[joint_name])
            if not math.isclose(expected_value, actual_value, abs_tol=atol):
                changed.append(f"{joint_name}: expected={expected_value:.3f} actual={actual_value:.3f}")
        if not changed:
            return None
        return "; ".join(changed)

    def _format_action_raw(self, arm: ArmHandle, values: dict[str, float] | None) -> str:
        if not values:
            return "--"
        parts = []
        for joint_name in JOINT_NAMES:
            if joint_name not in values:
                parts.append(f"{joint_name}=--")
                continue
            raw_value = arm.follower.bus.user_to_raw(joint_name, float(values[joint_name]))
            parts.append(f"{joint_name}={raw_value}")
        return ", ".join(parts)

    def _format_pose(self, pose: dict[str, float]) -> str:
        return ", ".join(f"{joint_name}={float(pose.get(joint_name, 0.0)):.3f}" for joint_name in JOINT_NAMES)

    def _zero_pose(self) -> dict[str, float]:
        return {joint_name: 0.0 for joint_name in JOINT_NAMES}

    def _run_zero_pose_sequence(
        self,
        arms: list[ArmHandle],
        *,
        sequence_name: str,
        allow_auto_enable: bool,
    ) -> bool:
        if not arms:
            return False
        zero_poses = {arm.side: self._zero_pose() for arm in arms}
        return self._move_selected_arms_to_pose(
            arms,
            zero_poses,
            start_message=f"{sequence_name}：先回到零位姿态。",
            success_message=f"{sequence_name}：已回到零位姿态。",
            failure_prefix=f"{sequence_name} 零位姿态",
            allow_auto_enable=allow_auto_enable,
        )

    def _selected_arms(self, selected_side: str | None) -> list[ArmHandle]:
        return [arm for arm in self.arm_handles if selected_side is None or arm.side == selected_side]

    def _get_config_pose(self, side: str, attr_name: str) -> dict[str, float]:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        pose = getattr(handle.follower.config, attr_name, {}) or {}
        return {joint_name: float(pose.get(joint_name, 0.0)) for joint_name in JOINT_NAMES}

    def _set_config_pose(self, side: str, attr_name: str, pose: dict[str, float]) -> None:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        normalized = {joint_name: float(pose.get(joint_name, 0.0)) for joint_name in JOINT_NAMES}
        setattr(handle.follower.config, attr_name, dict(normalized))
        setattr(handle.follower.bus.config, attr_name, dict(normalized))

        robot_cfg = self.cfg.robot
        if robot_cfg is None:
            return
        if side == "left":
            arm_cfg = getattr(robot_cfg, "left_arm_config", None)
        else:
            arm_cfg = getattr(robot_cfg, "right_arm_config", None)
        if arm_cfg is not None:
            setattr(arm_cfg, attr_name, dict(normalized))

    def _persist_runtime_config(self) -> None:
        if self.config_path is None:
            self._log("未提供配置文件路径，无法自动持久化当前配置。")
            return
        try:
            payload = _to_yaml_safe_value(self.cfg)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            self._log(f"已将当前配置持久化到: {self.config_path}")
        except Exception as exc:
            self._log(f"持久化配置失败: {exc}")

    def _configured_can_for_side(self, side: str) -> str:
        robot_cfg = getattr(self.cfg, "robot", None)
        if robot_cfg is None:
            return "--"
        arm_cfg = getattr(robot_cfg, f"{side}_arm_config", None)
        if arm_cfg is None:
            return "--"
        value = getattr(arm_cfg, "channel", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        value = getattr(arm_cfg, "port", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "--"

    def _set_can_side_status(self, side: str, text: str) -> None:
        label = self.can_status_labels.get(side)
        if label is not None:
            label.setText(text)

    def _append_can_result(self, message: str) -> None:
        if self.can_result_widget is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.can_result_widget.append(f"[{timestamp}] {message}")

    def _update_can_page_status(self) -> None:
        interfaces = self._available_can_interfaces()
        if self.can_available_label is not None:
            self.can_available_label.setText(", ".join(interfaces) if interfaces else "--")
        for side in ("left", "right"):
            config_label = self.can_config_labels.get(side)
            if config_label is not None:
                config_label.setText(self._configured_can_for_side(side))
            status_label = self.can_status_labels.get(side)
            if status_label is not None and not status_label.text():
                configured = self._configured_can_for_side(side)
                exists = configured in interfaces if configured != "--" else False
                status_label.setText(
                    f"配置口={'已出现' if exists else '未出现'} | 当前配置={configured}"
                )

    def _disconnect_arms(self) -> None:
        for arm in getattr(self, "arm_handles", []):
            try:
                arm.follower.disconnect()
            except Exception:
                pass

    def _visible_can_interfaces(self) -> list[str]:
        detected: list[str] = []
        try:
            net_root = Path("/sys/class/net")
            if net_root.is_dir():
                detected.extend(
                    sorted(name for name in os.listdir(net_root) if CAN_NAME_PATTERN.match(name))
                )
        except Exception:
            pass

        ordered: list[str] = []
        seen: set[str] = set()
        for name in detected:
            normalized = (name or "").strip()
            if not normalized or normalized in seen or not CAN_NAME_PATTERN.match(normalized):
                continue
            ordered.append(normalized)
            seen.add(normalized)
        return ordered

    def _available_can_interfaces(self) -> list[str]:
        # UI "检查" shows what the system really sees right now.
        # Do not merge configured defaults here, otherwise detection may falsely report can0/can1.
        return self._visible_can_interfaces()

    def _probe_can_interface_for_side(self, side: str, channel: str) -> tuple[bool, tuple[int, int, float], str]:
        robot_cfg = self.cfg.robot
        if robot_cfg is None:
            raise ValueError("Config is missing robot section.")
        raw_arm_cfg = getattr(robot_cfg, f"{side}_arm_config", None)
        if raw_arm_cfg is None:
            raise ValueError(f"Config is missing robot.{side}_arm_config.")

        calibration_dir_value = getattr(robot_cfg, "calibration_dir", None)
        calibration_dir = Path(calibration_dir_value) if calibration_dir_value is not None else None
        probe_cfg = _make_probe_port_config(
            app_id=f"{robot_cfg.id}_{side}_probe_{channel}",
            calibration_dir=calibration_dir,
            raw_arm_cfg=raw_arm_cfg,
            channel=channel,
        )
        port = make_songling_port(probe_cfg)
        try:
            port.connect()
            for _ in range(3):
                time.sleep(0.1)
                port.poll()
            positions = port.get_positions(poll=False)
            seen_count = sum(1 for joint_name in JOINT_NAMES if port.joint_position_seen.get(joint_name, False))
            signal_count = (
                int(getattr(port, "status_feedback_valid", False))
                + int(getattr(port, "joint_feedback_valid", False))
                + int(getattr(port, "gripper_feedback_valid", False))
                + int(getattr(port, "command_feedback_valid", False))
                + int(bool(getattr(port, "low_speed_feedback", {})))
                + int(bool(getattr(port, "high_speed_feedback", {})))
            )
            motion = sum(abs(float(value)) for value in positions.values())
            ok = bool(seen_count > 0 or signal_count >= 2)
            detail = (
                f"channel={channel} seen={seen_count} signals={signal_count} "
                f"motion={motion:.3f} status={int(bool(getattr(port, 'status_feedback_valid', False)))} "
                f"joint={int(bool(getattr(port, 'joint_feedback_valid', False)))} "
                f"gripper={int(bool(getattr(port, 'gripper_feedback_valid', False)))} "
                f"command={int(bool(getattr(port, 'command_feedback_valid', False)))}"
            )
            return ok, (seen_count, signal_count, motion), detail
        finally:
            try:
                port.disconnect()
            except Exception:
                pass

    def _set_detected_channel_in_config(self, side: str, channel: str) -> None:
        robot_cfg = getattr(self.cfg, "robot", None)
        if robot_cfg is not None:
            arm_cfg = getattr(robot_cfg, f"{side}_arm_config", None)
            if arm_cfg is not None and hasattr(arm_cfg, "channel"):
                setattr(arm_cfg, "channel", channel)
        teleop_cfg = getattr(self.cfg, "teleop", None)
        if teleop_cfg is not None:
            arm_cfg = getattr(teleop_cfg, f"{side}_arm_config", None)
            if arm_cfg is not None and hasattr(arm_cfg, "port"):
                setattr(arm_cfg, "port", channel)
        self._update_can_page_status()

    def _restart_application(self) -> None:
        if hasattr(self, "poll_timer"):
            self.poll_timer.stop()
        if hasattr(self, "hold_timer"):
            self.hold_timer.stop()
        self._disconnect_arms()
        QApplication.processEvents()
        os.execv(sys.executable, [sys.executable, *sys.argv])

    def _prompt_sudo_password(self) -> str | None:
        QMessageBox.information(
            self,
            "需要管理员密码",
            "一键 Up CAN 需要管理员权限。\n请输入当前系统用户的 sudo 密码，配置完成后软件会自动重启刷新连接状态。",
        )
        password, ok = QInputDialog.getText(
            self,
            "sudo 密码",
            "请输入 sudo 密码：",
            QLineEdit.Password,
        )
        if not ok:
            return None
        if not password:
            return None
        return str(password)

    def _run_sudo_command(self, command: list[str], password: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # nosec B603
            ["sudo", "-S", "-p", "", *command],
            input=f"{password}\n",
            capture_output=True,
            text=True,
        )

    def setup_can_interfaces(self) -> None:
        self._append_can_result(
            "开始执行一键 Up CAN（等价于 lerobot-setup-can --mode=setup --interfaces=can0,can1 --use_fd=false --bitrate=1000000）。"
        )
        password = self._prompt_sudo_password()
        if password is None:
            self._append_can_result("用户取消了一键 Up CAN。")
            return

        if self.connected:
            self._append_can_result("执行前断开当前连接，避免旧的 CAN 状态残留。")
        self._disconnect_arms()
        self.connected = False
        self.motors_enabled = False
        self.status_label.setText("正在执行 CAN setup | 完成后将自动重启")

        validate = self._run_sudo_command(["-v"], password)
        if validate.returncode != 0:
            error = (validate.stderr or validate.stdout or "sudo 密码校验失败。").strip()
            self._append_can_result(f"sudo 校验失败：{error}")
            self._log(f"一键 Up CAN 失败：{error}")
            QMessageBox.warning(self, "CAN 设置失败", f"sudo 校验失败：{error}")
            return

        interfaces = ["can0", "can1"]
        last_error: str | None = None
        success_count = 0
        for interface in interfaces:
            show = subprocess.run(["ip", "link", "show", interface], capture_output=True, text=True)  # nosec B603
            if show.returncode != 0:
                self._append_can_result(f"{interface}: 接口不存在，已跳过。")
                continue

            steps = [
                ["ip", "link", "set", interface, "down"],
                ["ip", "link", "set", interface, "type", "can", "bitrate", "1000000"],
                ["ip", "link", "set", interface, "up"],
            ]
            interface_ok = True
            for step in steps:
                result = self._run_sudo_command(step, password)
                if result.returncode != 0:
                    interface_ok = False
                    last_error = (result.stderr or result.stdout or f"{interface} 配置失败").strip()
                    self._append_can_result(f"{interface}: 命令失败：{' '.join(step)}")
                    if result.stdout:
                        self._append_can_result(result.stdout.strip())
                    if result.stderr:
                        self._append_can_result(result.stderr.strip())
                    break
            if interface_ok:
                success_count += 1
                self._append_can_result(f"{interface}: 已完成 down/type/up。")

        self._update_can_page_status()
        self.check_all_can()

        if success_count == 0:
            message = last_error or "没有任何 CAN 接口完成 setup。"
            self._log(f"一键 Up CAN 失败：{message}")
            QMessageBox.warning(self, "CAN 设置失败", message)
            return

        QMessageBox.information(
            self,
            "CAN 设置完成",
            "CAN 接口已重新 Up，程序将自动重启以刷新连接状态，让机械臂进入新的控制状态。",
        )
        self._append_can_result("CAN setup 成功，准备自动重启软件刷新连接状态。")
        self._restart_application()

    def _check_live_can_for_side(self, side: str) -> str:
        handle = next((arm for arm in self.arm_handles if arm.side == side), None)
        if handle is None:
            return "未找到该侧机械臂句柄。"
        try:
            handle.follower.bus.poll()
            positions = self._arm_positions(side)
        except Exception as exc:
            return f"实时检查失败：{exc}"
        seen = sum(1 for joint_name in JOINT_NAMES if handle.follower.bus.joint_position_seen.get(joint_name, False))
        command_gripper = int(bool(handle.follower.bus.commanded_position_seen.get("gripper", False)))
        motion = sum(abs(float(value)) for value in positions.values())
        return (
            f"配置={self._configured_can_for_side(side)} "
            f"seen={seen} status={int(bool(getattr(handle.follower.bus, 'status_feedback_valid', False)))} "
            f"joint={int(bool(getattr(handle.follower.bus, 'joint_feedback_valid', False)))} "
            f"gripper={int(bool(getattr(handle.follower.bus, 'gripper_feedback_valid', False)))} "
            f"cmd_gripper={command_gripper} motion={motion:.3f}"
        )

    def check_can_for_side(self, side: str) -> None:
        self._update_can_page_status()
        candidates = self._available_can_interfaces()
        configured = self._configured_can_for_side(side)
        exists = configured in candidates if configured != "--" else False
        summary = (
            f"当前配置={configured} | 配置口{'已出现' if exists else '未出现'} | 可见接口={', '.join(candidates) if candidates else '--'}"
        )
        self._set_can_side_status(side, summary)
        self._append_can_result(f"{_side_label(side)} 检查：{summary}")

    def check_all_can(self) -> None:
        self.check_can_for_side("left")
        self.check_can_for_side("right")

    def detect_can_for_side(self, side: str) -> None:
        if self.config_path is None:
            self._log("未提供配置文件路径，无法写回识别到的 CAN 口。")
            QMessageBox.warning(self, "CAN 配置失败", "未提供配置文件路径，无法写回识别到的 CAN 口。")
            return

        self._log(
            f"开始识别{_side_label(side)} CAN：请只连接这一条机械臂，并确保它当前接在上位机某个 canX 接口上。"
        )
        if self.connected:
            self._log("识别前先断开当前会话，避免旧连接占用 CAN 接口。")
        self._disconnect_arms()
        self.connected = False
        self.motors_enabled = False
        self.status_label.setText("已断开 | 正在识别 CAN 配置")
        self._set_can_side_status(side, "正在识别...")

        candidates = self._available_can_interfaces()
        if not candidates:
            message = "未发现可探测的 canX/slcanX 接口。请先确认 CAN 设备已接入并已创建接口。"
            self._log(message)
            self._set_can_side_status(side, message)
            self._append_can_result(f"{_side_label(side)} 识别失败：{message}")
            QMessageBox.warning(self, "CAN 识别失败", message)
            return

        if len(candidates) != 1:
            message = (
                f"当前检测到多个可见接口：{', '.join(candidates)}。"
                "为避免歧义，请只保留一个可见 canX/slcanX 接口后再点识别。"
            )
            self.status_label.setText("CAN 识别失败")
            self._log(message)
            self._set_can_side_status(side, message)
            self._append_can_result(f"{_side_label(side)} 识别失败：{message}")
            QMessageBox.warning(self, "CAN 识别失败", message)
            return

        detected_channel = candidates[0]
        detail = f"visible_interfaces={detected_channel}"
        self._set_detected_channel_in_config(side, detected_channel)
        self._persist_runtime_config()
        self._log(f"{_side_label(side)} CAN 已识别为 {detected_channel}，并已写回配置文件。{detail}")
        self._set_can_side_status(side, f"已识别并写回：{detected_channel}")
        self._append_can_result(f"{_side_label(side)} 已识别并写回：{detected_channel} | {detail}")
        QMessageBox.information(
            self,
            "CAN 识别完成",
            f"{_side_label(side)} 已识别为 {detected_channel}。\n配置文件已更新，程序将立即重启以重新加载新配置。",
        )
        self._restart_application()

    def set_current_pose_as_pre_disable(self, side: str) -> None:
        if not self.connected:
            return
        if not self._arm_has_feedback(side):
            self._log(f"{_side_label(side)} 未收到反馈，无法设置失能前姿态。")
            return
        current = self._arm_positions(side)
        self._set_config_pose(side, "pre_disable_pose", current)
        self._log(
            f"{_side_label(side)} 已将当前姿态设为失能前姿态：{self._format_pose(current)}"
        )
        self._persist_runtime_config()

    def move_to_zero_pose(self, selected_side: str | None) -> None:
        if not self.connected:
            return
        if not self._require_motors_enabled(_ui_text("Move To Zero Pose", "\u56de\u96f6\u4f4d\u59ff\u6001")):
            return
        selected = self._selected_arms(selected_side)
        if not selected:
            return
        poses_by_side = {arm.side: self._zero_pose() for arm in selected}
        self._move_selected_arms_to_pose(
            selected,
            poses_by_side,
            start_message=(
                "已将双臂目标设为零位姿态，准备自动回零。"
                if selected_side is None
                else f"{_side_label(selected_side)} 已将目标设为零位姿态，准备自动回零。"
            ),
            success_message=(
                "双臂已回到零位姿态。"
                if selected_side is None
                else f"{_side_label(selected_side)} 已回到零位姿态。"
            ),
            failure_prefix="零位姿态",
            allow_auto_enable=False,
        )

    def _is_side_near_pose(
        self,
        side: str,
        pose: dict[str, float],
        *,
        joint_tolerance: float = SOFT_ZERO_JOINT_TOL,
        gripper_tolerance: float = SOFT_ZERO_GRIPPER_TOL,
    ) -> bool:
        current = self.current_values.get(side) or self._arm_positions(side)
        for joint_name in JOINT_NAMES:
            tolerance = gripper_tolerance if joint_name == "gripper" else joint_tolerance
            if not math.isclose(float(current[joint_name]), float(pose[joint_name]), abs_tol=tolerance):
                return False
        return True

    def _move_selected_arms_to_pose(
        self,
        arms: list[ArmHandle],
        poses_by_side: dict[str, dict[str, float]],
        *,
        start_message: str,
        success_message: str,
        failure_prefix: str,
        allow_auto_enable: bool,
        max_attempts: int = SOFT_ZERO_MAX_ATTEMPTS,
        joint_tolerance: float = SOFT_ZERO_JOINT_TOL,
        gripper_tolerance: float = SOFT_ZERO_GRIPPER_TOL,
    ) -> bool:
        if not arms:
            return False

        active_arms: list[ArmHandle] = []
        blocked_messages: list[str] = []
        for arm in arms:
            if not self._arm_has_feedback(arm.side):
                blocked_messages.append(f"{_side_label(arm.side)} 无状态反馈")
                continue
            if not self._arm_has_position_feedback(arm.side):
                blocked_messages.append(f"{_side_label(arm.side)} 无关节反馈")
                continue
            active_arms.append(arm)

        if not active_arms:
            self._log(f"{failure_prefix} 自动阶段未启动：{', '.join(blocked_messages) if blocked_messages else '无可用机械臂'}。")
            return False

        for arm in active_arms:
            pose = poses_by_side[arm.side]
            for joint_name in JOINT_NAMES:
                self._set_target_value(arm.side, joint_name, float(pose[joint_name]))

        self._log(start_message)
        if blocked_messages:
            self._log(f"{failure_prefix} 自动阶段跳过：{', '.join(blocked_messages)}。")

        pending_sides = {arm.side for arm in active_arms}
        blocked_sides: set[str] = set()
        for _ in range(1, max(int(max_attempts), 1) + 1):
            send_results = self._send_targets_for_arms(
                [arm for arm in active_arms if arm.side in pending_sides],
                allow_auto_enable=allow_auto_enable,
                verbose=False,
            )
            for side, result in send_results.items():
                if result != "sent":
                    pending_sides.discard(side)
                    blocked_sides.add(side)
            time.sleep(SOFT_ZERO_SETTLE_S)
            self.refresh_positions(sync_targets=False)
            pending_sides = {
                arm.side
                for arm in active_arms
                if arm.side in pending_sides
                and not self._is_side_near_pose(
                    arm.side,
                    poses_by_side[arm.side],
                    joint_tolerance=joint_tolerance,
                    gripper_tolerance=gripper_tolerance,
                )
            }
            if not pending_sides:
                if blocked_sides:
                    break
                self._log(success_message)
                return True

        failed_sides = pending_sides | blocked_sides
        pending_text = ", ".join(_side_label(side) for side in sorted(failed_sides))
        self._log(f"{failure_prefix} 自动阶段仍未完全到位：{pending_text}。")
        return False

    def _run_pre_disable_sequence(
        self,
        arms: list[ArmHandle],
        *,
        sequence_name: str,
        allow_auto_enable: bool,
    ) -> bool:
        if not arms:
            return False
        pre_disable_poses = {arm.side: self._get_config_pose(arm.side, "pre_disable_pose") for arm in arms}
        return self._move_selected_arms_to_pose(
            arms,
            pre_disable_poses,
            start_message=f"{sequence_name}：先回到失能前姿态。",
            success_message=f"{sequence_name}：已回到失能前姿态。",
            failure_prefix=f"{sequence_name} 失能前姿态",
            allow_auto_enable=allow_auto_enable,
            max_attempts=PRE_DISABLE_MAX_ATTEMPTS,
            joint_tolerance=PRE_DISABLE_JOINT_TOL,
            gripper_tolerance=PRE_DISABLE_GRIPPER_TOL,
        )

    def move_to_pre_disable_pose(self, selected_side: str | None) -> None:
        if not self.connected:
            return
        if not self._require_motors_enabled(_ui_text("Move To Pre-Disable Pose", "\u56de\u5931\u80fd\u524d\u59ff\u6001")):
            return
        arms = self._selected_arms(selected_side)
        if not arms:
            return
        poses_by_side = {arm.side: self._get_config_pose(arm.side, "pre_disable_pose") for arm in arms}
        self._move_selected_arms_to_pose(
            arms,
            poses_by_side,
            start_message=(
                "已将双臂目标设为失能前姿态，准备自动移动。"
                if selected_side is None
                else f"{_side_label(selected_side)} 已将目标设为失能前姿态，准备自动移动。"
            ),
            success_message=(
                "双臂已回到失能前姿态。"
                if selected_side is None
                else f"{_side_label(selected_side)} 已回到失能前姿态。"
            ),
            failure_prefix="失能前姿态",
            allow_auto_enable=False,
        )

    def _drivers_all_enabled(self, arm: ArmHandle) -> bool:
        statuses = arm.follower.bus.get_driver_enable_status()
        known = [state for state in statuses.values() if state is not None]
        return bool(known) and all(bool(state) for state in known)

    def _wait_for_driver_enable_status_all(
        self,
        arms: list[ArmHandle],
        *,
        enabled: bool,
    ) -> dict[str, bool]:
        expected = bool(enabled)
        if not arms:
            return {}

        retry_count = max(max(int(arm.follower.config.enable_retry_count) for arm in arms), 1)
        retry_interval_s = max(max(float(arm.follower.config.enable_retry_interval_s) for arm in arms), 0.0)
        final_status = {arm.side: False for arm in arms}

        for _ in range(retry_count):
            for arm in arms:
                arm.follower.bus.enable_motors(enable=enabled)
            if retry_interval_s > 0:
                time.sleep(retry_interval_s)
            for arm in arms:
                arm.follower.bus.poll()
                statuses = arm.follower.bus.get_driver_enable_status()
                known = [state for state in statuses.values() if state is not None]
                final_status[arm.side] = bool(known) and all(bool(state) is expected for state in known)
            if all(final_status.values()):
                return final_status
        return final_status

    def _format_driver_details(self, arm: ArmHandle) -> str:
        details = []
        for joint_name in JOINT_NAMES[:-1]:
            info = arm.follower.bus.low_speed_feedback.get(joint_name)
            if not info:
                details.append(f"{joint_name}(--)")
                continue
            details.append(
                f"{joint_name}(en={'1' if info.get('driver_enable_status') else '0'},"
                f"err={'1' if info.get('driver_error_status') else '0'},"
                f"col={'1' if info.get('collision_status') else '0'},"
                f"stall={'1' if info.get('stall_status') else '0'},"
                f"ibus={float(info.get('bus_current', 0)) / 1000.0:.3f}A,"
                f"foc={int(info.get('foc_temp', 0))},"
                f"motor={int(info.get('motor_temp', 0))})"
            )
        return "; ".join(details)

    def _format_feedback_health(self, arm: ArmHandle) -> str:
        bus = arm.follower.bus
        status_feedback = (
            f"status_feedback={'OK' if getattr(bus, 'status_feedback_valid', False) else 'MISSING'}"
            f"(hz={float(getattr(bus, 'status_feedback_hz', 0.0)):.1f},"
            f"ts={float(getattr(bus, 'status_feedback_timestamp', 0.0)):.3f})"
        )
        joint_feedback = (
            f"joint_feedback={'OK' if bus.joint_feedback_valid else 'MISSING'}"
            f"(hz={bus.joint_feedback_hz:.1f},ts={bus.joint_feedback_timestamp:.3f})"
        )
        gripper_feedback = (
            f"gripper_feedback={'OK' if bus.gripper_feedback_valid else 'MISSING'}"
            f"(hz={bus.gripper_feedback_hz:.1f},ts={bus.gripper_feedback_timestamp:.3f})"
        )
        mode_feedback = (
            f"mode_command={'OK' if getattr(bus, 'mode_command_valid', False) else 'MISSING'}"
            f"(hz={float(getattr(bus, 'mode_command_hz', 0.0)):.1f},"
            f"ts={float(getattr(bus, 'mode_command_timestamp', 0.0)):.3f})"
        )
        command_feedback = (
            f"command_feedback={'OK' if getattr(bus, 'command_feedback_valid', False) else 'MISSING'}"
            f"(hz={float(getattr(bus, 'command_feedback_hz', 0.0)):.1f},"
            f"ts={float(getattr(bus, 'command_feedback_timestamp', 0.0)):.3f})"
        )
        comm_bits = ", ".join(
            f"{joint_name}={'ERR' if bus.communication_status.get(joint_name, False) else 'OK'}"
            for joint_name in JOINT_NAMES[:-1]
        )
        return f"{status_feedback} | {joint_feedback} | {gripper_feedback} | {mode_feedback} | {command_feedback} | comm={comm_bits}"

    def _format_high_speed_details(self, arm: ArmHandle) -> str:
        details = []
        for joint_name in JOINT_NAMES[:-1]:
            info = arm.follower.bus.high_speed_feedback.get(joint_name)
            if not info:
                details.append(f"{joint_name}(--)")
                continue
            pos_user = arm.follower.bus.raw_to_user(joint_name, int(info.get("position", 0)))
            details.append(
                f"{joint_name}(spd={int(info.get('motor_speed', 0))},"
                f"cur={int(info.get('current', 0))},"
                f"pos_raw={int(info.get('position', 0))},"
                f"pos={pos_user:.3f})"
            )
        return "; ".join(details)

    def _log_command_source_warning(self, arm: ArmHandle) -> None:
        bus_command = self._arm_command_positions(arm.side)
        mode_command = getattr(arm.follower.bus, "mode_command", {}) if getattr(arm.follower.bus, "mode_command_valid", False) else None
        if mode_command is not None:
            self._log(
                f"{_side_label(arm.side)} 启动时检测到 `0x151` 模式命令："
                f"ctrl_mode=0x{int(mode_command.get('ctrl_mode', 0)):x}, "
                f"move_mode=0x{int(mode_command.get('move_mode', 0)):x}, "
                f"mit_mode=0x{int(mode_command.get('mit_mode', 0)):x}, "
                f"installation_pos=0x{int(mode_command.get('installation_pos', 0)):x}"
            )

        if bus_command is None:
            self._log(
                f"{_side_label(arm.side)} 启动时未检测到 `0x155..0x159` 控制帧，当前更像是“无主臂干扰”的直控校验环境。"
            )
            return

        self._log(
            f"{_side_label(arm.side)} 启动时已检测到共享 CAN 上存在控制目标："
            f"{self._format_joint_values(bus_command)}"
        )
        self._log(
            f"{_side_label(arm.side)} 提示：如果你现在要验证上位机直控从臂，请先断开主臂或停止主臂发控制帧；"
            "否则主臂会持续覆盖上位机下发的 `0x151/0x155..0x159` 指令。"
        )

    def _log_arm_diagnostics(
        self,
        arm: ArmHandle,
        *,
        requested: dict[str, float] | None = None,
        sent: dict[str, float] | None = None,
    ) -> None:
        current = dict(self.current_values.get(arm.side) or self._arm_positions(arm.side))
        target = self._arm_target_positions(arm.side)
        requested_positions = None
        if requested is not None:
            requested_positions = {
                key.removesuffix(".pos"): float(value) for key, value in requested.items() if key.endswith(".pos")
            }
        sent_positions = None
        if sent is not None:
            sent_positions = {
                key.removesuffix(".pos"): float(value) for key, value in sent.items() if key.endswith(".pos")
            }
        bus_command_positions = self._arm_command_positions(arm.side)

        self._log(
            f"{_side_label(arm.side)}关键诊断："
            f"当前={self._format_joint_values(current)} | "
            f"界面目标={self._format_joint_values(target)} | "
            f"请求={self._format_joint_values(requested_positions)} | "
            f"实发={self._format_joint_values(sent_positions)} | "
            f"总线目标={self._format_joint_values(bus_command_positions)}"
        )
        self._log(
            f"{_side_label(arm.side)}原始指令："
            f"界面目标={self._format_action_raw(arm, target)} | "
            f"请求={self._format_action_raw(arm, requested_positions)} | "
            f"实发={self._format_action_raw(arm, sent_positions)} | "
            f"总线目标={self._format_action_raw(arm, bus_command_positions)}"
        )
        self._log(f"{_side_label(arm.side)}低速反馈：{self._format_driver_details(arm)}")
        self._log(f"{_side_label(arm.side)}高速反馈：{self._format_high_speed_details(arm)}")
        if sent_positions is not None and bus_command_positions is not None:
            mismatch = self._format_position_mismatch(sent_positions, bus_command_positions, atol=1e-3)
            if mismatch is None:
                self._log(f"{_side_label(arm.side)}总线目标校验：OK，`0x155..0x159` 与本次实发一致。")
            else:
                self._log(
                    f"{_side_label(arm.side)}总线目标校验：MISMATCH，`0x155..0x159` 与本次实发不一致：{mismatch}"
                )

    def _log_arm_status(
        self,
        arm: ArmHandle,
        prefix: str,
        *,
        include_diagnostics: bool = False,
        requested: dict[str, float] | None = None,
        sent: dict[str, float] | None = None,
    ) -> None:
        arm.follower.bus.poll()
        enable_status = arm.follower.bus.get_driver_enable_status()
        enable_text = ", ".join(
            f"{joint_name}={'ON' if state else 'OFF' if state is not None else '--'}"
            for joint_name, state in enable_status.items()
        )
        if getattr(arm.follower.bus, "status_feedback_valid", False):
            summary = _format_status(dict(arm.follower.bus.status))
        elif getattr(arm.follower.bus, "mode_command_valid", False):
            mode_cmd = getattr(arm.follower.bus, "mode_command", {})
            summary = (
                f"未收到真实状态帧，按 `0x151` 观测："
                f"ctrl_mode=0x{int(mode_cmd.get('ctrl_mode', 0)):x}, "
                f"move_mode=0x{int(mode_cmd.get('move_mode', 0)):x}, "
                f"mit_mode=0x{int(mode_cmd.get('mit_mode', 0)):x}, "
                f"installation_pos=0x{int(mode_cmd.get('installation_pos', 0)):x}"
            )
        else:
            summary = "未收到真实状态帧，也未观察到 `0x151` 模式命令"
        self._log(f"{_side_label(arm.side)}{prefix}\uff1a{summary} | \u9a71\u52a8\u4f7f\u80fd={enable_text}")
        self._log(f"{_side_label(arm.side)}反馈健康：{self._format_feedback_health(arm)}")
        if include_diagnostics:
            self._log_arm_diagnostics(arm, requested=requested, sent=sent)

    def _arm_has_feedback(self, side: str) -> bool:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        handle.follower.bus.poll()
        return bool(
            getattr(handle.follower.bus, "status_feedback_valid", False)
            or getattr(handle.follower.bus, "mode_command_valid", False)
            or handle.follower.bus.joint_feedback_valid
            or handle.follower.bus.gripper_feedback_valid
            or bool(handle.follower.bus.low_speed_feedback)
            or bool(handle.follower.bus.high_speed_feedback)
        )

    def _arm_has_position_feedback(self, side: str) -> bool:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        handle.follower.bus.poll()
        has_feedback = handle.follower.bus.has_reliable_position_feedback()
        if not has_feedback and not self._missing_joint_feedback_warned.get(side, False):
            self._log(
                f"{_side_label(side)} 还未收到完整的关节+夹爪反馈，涉及安全保持/回零/自动动作的功能会先禁用。"
            )
            self._missing_joint_feedback_warned[side] = True
        if has_feedback:
            self._missing_joint_feedback_warned[side] = False
        return has_feedback

    def _arm_has_any_feedback(self, side: str) -> bool:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        handle.follower.bus.poll()
        return any(bool(handle.follower.bus.joint_position_seen.get(joint_name, False)) for joint_name in JOINT_NAMES) or bool(
            handle.follower.bus.commanded_position_seen.get("gripper", False)
        )

    def _arm_joint_has_display_value(self, side: str, joint_name: str) -> bool:
        handle = next(arm for arm in self.arm_handles if arm.side == side)
        if handle.follower.bus.joint_position_seen.get(joint_name, False):
            return True
        return bool(joint_name == "gripper" and handle.follower.bus.commanded_position_seen.get("gripper", False))

    def _require_motors_enabled(self, operation_name: str) -> bool:
        if self.motors_enabled:
            return True
        self._log(f"当前电机处于失能状态，已禁止“{operation_name}”。请先点击“电机使能”。")
        return False

    def _requested_matches_current(self, side: str, requested: dict[str, float]) -> bool:
        current = self.current_values.get(side) or self._arm_positions(side)
        return all(
            math.isclose(float(current[joint_name]), float(requested[f"{joint_name}.pos"]), abs_tol=1e-6)
            for joint_name in JOINT_NAMES
        )

    def refresh_positions(self, sync_targets: bool = False) -> None:
        if not self.connected:
            return
        for arm in self.arm_handles:
            if not self._arm_has_any_feedback(arm.side):
                for joint_name in JOINT_NAMES:
                    self.current_labels[arm.side][joint_name].setText("--")
                continue
            positions = self._arm_positions(arm.side)
            self.current_values[arm.side] = positions
            if self._arm_has_position_feedback(arm.side):
                self._ensure_positions_within_limits(arm.side, positions)
            for joint_name in JOINT_NAMES:
                if self._arm_joint_has_display_value(arm.side, joint_name):
                    value = float(positions[joint_name])
                    self.current_labels[arm.side][joint_name].setText(f"{value:.3f}")
                else:
                    self.current_labels[arm.side][joint_name].setText("--")
                if (
                    (sync_targets or not self._targets_initialized_by_side.get(arm.side, False))
                    and arm.follower.bus.has_reliable_position_feedback()
                    and joint_name in positions
                ):
                    value = float(positions[joint_name])
                    self._set_target_value(arm.side, joint_name, value)
                    self._targets_initialized_by_side[arm.side] = True
        self.status_label.setText(
            _ui_text(
                f"Connected | motors {'enabled' if self.motors_enabled else 'disabled'}",
                f"\u5df2\u8fde\u63a5 | \u7535\u673a{'\u5df2\u4f7f\u80fd' if self.motors_enabled else '\u672a\u4f7f\u80fd'}",
            )
        )

    def sync_targets_to_current(self) -> None:
        self._capture_current_pose_targets(selected_side=None, update_ui_targets=True)
        self._log(
            _ui_text(
                "Targets restored to current robot pose. Sending now should not jump.",
                "\u76ee\u6807\u503c\u5df2\u6062\u590d\u5230\u5f53\u524d\u673a\u68b0\u81c2\u59ff\u6001\u3002\u6b64\u65f6\u53d1\u9001\u4e0d\u4f1a\u4ea7\u751f\u7a81\u8df3\u3002",
            )
        )

    def jog_target(self, side: str, joint_name: str, direction: int) -> None:
        step = self.gripper_step_spin.value() if joint_name == "gripper" else self.joint_step_spin.value()
        current = self.target_spinboxes[side][joint_name].value()
        limits = self.runtime_joint_limits[side][joint_name]
        new_value = min(max(current + (direction * step), limits[0]), limits[1])
        self._set_target_value(side, joint_name, new_value)

    def enable_and_configure(self) -> None:
        if not self.connected:
            return
        try:
            for arm in self.arm_handles:
                self.commanded_targets[arm.side] = None
            self._log("开始执行使能流程：会先尝试使能驱动，再将机械臂自动引导到零位姿态。")
            for arm in self.arm_handles:
                arm.follower.bus.ensure_can_command_mode(force=True)
            enabled_map = self._wait_for_driver_enable_status_all(self.arm_handles, enabled=True)
            time.sleep(0.05)
            ready_arms: list[ArmHandle] = []
            for arm in self.arm_handles:
                enabled_ok = enabled_map.get(arm.side, False)
                if not enabled_ok:
                    self._log(
                        f"{_side_label(arm.side)} {_ui_text('driver enable handshake did not complete', '\u9a71\u52a8\u4f7f\u80fd\u63e1\u624b\u672a\u5b8c\u6210')}"
                    )
                arm.follower.bus.ensure_can_command_mode(force=True)
                if not self._arm_has_feedback(arm.side):
                    self._log(
                        f"{_side_label(arm.side)} {_ui_text('has no status feedback; zero-pose recovery and hold will be skipped for safety', '\u672a\u6536\u5230\u72b6\u6001\u53cd\u9988\uff0c\u4e3a\u5b89\u5168\u8d77\u89c1\u5c06\u8df3\u8fc7\u201c\u96f6\u4f4d\u59ff\u6001\u201d\u6062\u590d\u548c\u540e\u7eed\u4fdd\u6301')}"
                    )
                    self.commanded_targets[arm.side] = None
                    continue
                ready, reason = self._arm_ready_for_command(arm)
                if not ready:
                    self.commanded_targets[arm.side] = None
                    self._log(
                        f"{_side_label(arm.side)} \u521a\u4f7f\u80fd\u540e\u672a\u8fdb\u5165\u53ef\u5b89\u5168\u56de\u96f6\u4f4d\u59ff\u6001\u72b6\u6001\uff0c\u5148\u8df3\u8fc7\u81ea\u52a8\u79fb\u52a8\uff1a{reason}"
                    )
                    continue
                ready_arms.append(arm)
            self.motors_enabled = True
            self.status_label.setText(_ui_text("Connected | motors enabled", "\u5df2\u8fde\u63a5 | \u7535\u673a\u5df2\u4f7f\u80fd"))
            for arm in self.arm_handles:
                self._log_arm_status(arm, _ui_text("status after enable", "\u4f7f\u80fd\u540e\u72b6\u6001"), include_diagnostics=True)
            self._log(
                _ui_text(
                    "Enable and CAN-control-mode commands were sent to both arms.",
                    "\u5df2\u5411\u53cc\u81c2\u53d1\u9001\u4f7f\u80fd\u548c CAN \u63a7\u5236\u6a21\u5f0f\u6307\u4ee4\u3002",
                )
            )
            if ready_arms:
                self._run_zero_pose_sequence(
                    ready_arms,
                    sequence_name="使能流程",
                    allow_auto_enable=False,
                )
                self.refresh_positions(sync_targets=False)
            else:
                self._log("当前没有可自动回零位姿态的机械臂，已跳过使能后的自动移动。")
            if self.hold_position_checkbox.isChecked():
                tracked_sides = {
                    arm.side for arm in ready_arms if self.commanded_targets.get(arm.side) is not None
                }
                if tracked_sides:
                    self._log(
                        _ui_text(
                            f"Zero-pose tracking is active at {HOLD_TIMER_MS} ms.",
                            f"\u5df2\u5c06\u53ef\u7528\u673a\u68b0\u81c2\u76ee\u6807\u8bbe\u4e3a\u201c\u96f6\u4f4d\u59ff\u6001\u201d\uff0c\u5e76\u4ee5 {HOLD_TIMER_MS} ms \u5468\u671f\u6301\u7eed\u8ffd\u8e2a\u3002",
                        )
                    )
                else:
                    self._log("当前没有可持续追踪零位姿态的机械臂，已跳过自动保持。")
            else:
                self._log("未勾选持续保持，已跳过启动后的自动保持。")
            self._log(
                _ui_text(
                    "Next: confirm the arms have approached the zero pose, then click 'Target -> Current' before jogging.",
                    "\u63a5\u4e0b\u6765\u8bf7\u5148\u786e\u8ba4\u673a\u68b0\u81c2\u5df2\u63a5\u8fd1\u201c\u96f6\u4f4d\u59ff\u6001\u201d\uff0c\u518d\u70b9\u201c\u76ee\u6807\u56de\u5f53\u524d\u59ff\u6001\u201d\u540e\u5c0f\u6b65\u53d1\u9001\u3002",
                )
            )
        except Exception as exc:
            self._log(f"{_ui_text('Enable failed', '\u4f7f\u80fd\u5931\u8d25')}: {exc}")
            QMessageBox.critical(self, _ui_text("Enable failed", "\u4f7f\u80fd\u5931\u8d25"), str(exc))

    def disable_motors(self) -> None:
        if not self.connected:
            return
        try:
            self._manual_send_in_progress = True
            self._log("开始执行失能流程：会自动回到失能前姿态，到位后直接统一失能，无需手动回位。")
            prepared = self._run_pre_disable_sequence(
                self._selected_arms(None),
                sequence_name="失能前准备",
                allow_auto_enable=False,
            )
            if not prepared:
                self._log("失能前姿态未完全到位，本次已取消失能，避免机械臂直接下坠碰撞。")
                return
            self._log("双臂已到失能前姿态，开始自动统一失能。")
            disabled_map: dict[str, bool] | None = None
            wait_for_disable = getattr(self, "_wait_for_driver_enable_status_all", None)
            if callable(wait_for_disable):
                try:
                    disabled_map = wait_for_disable(self.arm_handles, enabled=False)
                except Exception as exc:
                    self._log(f"失能握手确认失败，已退回单次失能指令：{exc}")
            if disabled_map is None:
                for arm in self.arm_handles:
                    arm.follower.bus.enable_motors(enable=False)
            else:
                incomplete = [side for side, ok in disabled_map.items() if not ok]
                if incomplete:
                    side_text = ", ".join(_side_label(side) for side in sorted(incomplete))
                    self._log(f"以下机械臂未确认到失能反馈，请现场确认：{side_text}")
            for arm in self.arm_handles:
                self.commanded_targets[arm.side] = None
            self.motors_enabled = False
            self.status_label.setText(_ui_text("Connected | motors disabled", "\u5df2\u8fde\u63a5 | \u7535\u673a\u672a\u4f7f\u80fd"))
            self._log(
                _ui_text(
                    "Both arms reached the pre-disable pose and were then disabled automatically.",
                    "\u5df2\u5bf9\u53cc\u81c2\u6267\u884c\u201c\u81ea\u52a8\u56de\u5931\u80fd\u524d\u59ff\u6001 -> \u76f4\u63a5\u5931\u80fd\u201d\u6d41\u7a0b\u3002",
                )
            )
        except Exception as exc:
            self._log(f"{_ui_text('Disable failed', '\u5931\u80fd\u5931\u8d25')}: {exc}")
            QMessageBox.critical(self, _ui_text("Disable failed", "\u5931\u80fd\u5931\u8d25"), str(exc))
        finally:
            self._manual_send_in_progress = False

    def _requested_action(self, side: str) -> dict[str, float]:
        return {f"{joint_name}.pos": self.target_spinboxes[side][joint_name].value() for joint_name in JOINT_NAMES}

    def _format_clamp_log(self, requested: dict[str, float], sent: dict[str, float]) -> str | None:
        changed: list[str] = []
        for key, target in requested.items():
            sent_value = float(sent.get(key, target))
            if not math.isclose(sent_value, float(target), abs_tol=1e-6):
                changed.append(f"{key}: requested={float(target):.3f} sent={sent_value:.3f}")
        if not changed:
            return None
        return "; ".join(changed)

    def _send_targets_for_arms(
        self,
        arms: list[ArmHandle],
        *,
        allow_auto_enable: bool,
        verbose: bool = True,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        if not self.connected:
            return results
        if not self.motors_enabled:
            if verbose:
                self._require_motors_enabled(_ui_text("Send Target", "\u53d1\u9001\u76ee\u6807"))
            return results
        try:
            self._manual_send_in_progress = True
            _ = allow_auto_enable

            for arm in arms:
                if not self._arm_has_feedback(arm.side):
                    self.commanded_targets[arm.side] = None
                    results[arm.side] = "no_status"
                    if verbose:
                        self._log(
                            f"{_side_label(arm.side)} {_ui_text('has no status feedback; send skipped', '\u672a\u6536\u5230\u72b6\u6001\u53cd\u9988\uff0c\u5df2\u8df3\u8fc7\u53d1\u9001')}"
                        )
                    continue
                arm.follower.bus.ensure_can_command_mode(force=True)
                ready, reason, fallback_note = self._arm_ready_for_manual_send(arm)
                if not ready:
                    self.commanded_targets[arm.side] = None
                    results[arm.side] = "not_ready"
                    if verbose:
                        self._log(
                            f"{_side_label(arm.side)} 未进入可安全发送状态，已跳过发送：{reason}"
                        )
                        self._log_arm_status(
                            arm,
                            _ui_text("status before skipped send", "\u8df3\u8fc7\u53d1\u9001\u524d\u72b6\u6001"),
                            include_diagnostics=True,
                        )
                    continue
                if (
                    verbose
                    and fallback_note is not None
                    and not self._manual_send_fallback_warned.get(arm.side, False)
                ):
                    self._log(f"{_side_label(arm.side)} {fallback_note}")
                    self._manual_send_fallback_warned[arm.side] = True
                requested = self._requested_action(arm.side)
                if self._requested_matches_current(arm.side, requested):
                    if verbose:
                        self._log(
                            f"{_side_label(arm.side)} {_ui_text('target already equals current feedback; robot may not move', '\u76ee\u6807\u503c\u4e0e\u5f53\u524d\u53cd\u9988\u4e00\u81f4\uff0c\u673a\u68b0\u81c2\u53ef\u80fd\u4e0d\u4f1a\u79fb\u52a8\u3002')}"
                        )
                sent = arm.follower.send_action(requested)
                clamp_log = self._format_clamp_log(requested, sent)
                self.commanded_targets[arm.side] = self._action_to_positions(requested)
                results[arm.side] = "sent"
                if clamp_log is None:
                    if verbose:
                        self._log(f"{_ui_text('Sent target to', '\u5df2\u53d1\u9001')}{_side_label(arm.side)}.")
                else:
                    if verbose:
                        self._log(
                            f"{_side_label(arm.side)} {_ui_text('target sent but safety-clamped', '\u76ee\u6807\u5df2\u53d1\u9001\uff0c\u4f46\u88ab\u5b89\u5168\u9650\u5e45')}: {clamp_log}"
                        )
                        if self.hold_position_checkbox.isChecked():
                            self._log(
                                f"{_side_label(arm.side)} 已保留原目标；持续保持会继续按安全步长向原目标逼近。"
                            )
                        else:
                            self._log(
                                f"{_side_label(arm.side)} 已保留原目标；如需继续逼近，请再次点击发送。"
                            )
                if verbose:
                    self._log_arm_status(
                        arm,
                        _ui_text("status after send", "\u53d1\u9001\u540e\u72b6\u6001"),
                        include_diagnostics=True,
                        requested=requested,
                        sent=sent,
                    )

            self.refresh_positions(sync_targets=False)
        except Exception as exc:
            self._log(f"{_ui_text('Send failed', '\u53d1\u9001\u5931\u8d25')}: {exc}")
            QMessageBox.critical(self, _ui_text("Send failed", "\u53d1\u9001\u5931\u8d25"), str(exc))
        finally:
            self._manual_send_in_progress = False
        return results

    def send_targets(self, selected_side: str | None) -> None:
        self._send_targets_for_arms(self._selected_arms(selected_side), allow_auto_enable=True)

    def _poll_current_positions(self) -> None:
        if self.auto_poll_checkbox.isChecked() and self.connected:
            try:
                self.refresh_positions(sync_targets=False)
            except Exception as exc:
                self.status_label.setText(_ui_text("Polling failed", "\u8f6e\u8be2\u5931\u8d25"))
                self._log(f"{_ui_text('Polling failed', '\u8f6e\u8be2\u5931\u8d25')}: {exc}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if hasattr(self, "poll_timer"):
            self.poll_timer.stop()
        if hasattr(self, "hold_timer"):
            self.hold_timer.stop()
        self._disconnect_arms()
        event.accept()


def main() -> None:
    args, unknown_args = _parse_args()
    register_third_party_plugins()
    cfg = _load_config(args.config_path, unknown_args)

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ui_font_family = _pick_qt_font_family()
    ui_font = QFont(ui_font_family, UI_FONT_SIZE)
    app.setFont(ui_font)

    window = ManualJointControlWindow(
        cfg=cfg,
        poll_ms=args.poll_ms,
        window_title=args.window_title,
        config_path=args.config_path,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    from PyQt5.QtGui import QFontInfo

    main()
