from __future__ import annotations

import logging
from typing import Any


SONGLING_FOLLOWER_TYPES = {"songling_follower", "bi_songling_follower"}
OPENARM_LEADER_TYPES = {"openarm_leader", "bi_openarm_leader"}
_SONGLING_SAFETY_WARNED_KEYS: set[str] = set()


def _cfg_type(cfg: Any) -> str | None:
    value = getattr(cfg, "type", None)
    return str(value) if value is not None else None


def _normalized_port(cfg: Any) -> str | None:
    if cfg is None:
        return None

    for attr in ("channel", "port"):
        value = getattr(cfg, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _port_map(cfg: Any) -> dict[str, str] | None:
    if cfg is None:
        return None

    if hasattr(cfg, "left_arm_config") and hasattr(cfg, "right_arm_config"):
        left_port = _normalized_port(getattr(cfg, "left_arm_config", None))
        right_port = _normalized_port(getattr(cfg, "right_arm_config", None))
        if left_port is None or right_port is None:
            return None
        return {"left": left_port, "right": right_port}

    port = _normalized_port(cfg)
    if port is None:
        return None
    return {"single": port}


def is_songling_integrated_openarm_risk(robot_cfg: Any, teleop_cfg: Any) -> bool:
    if robot_cfg is None or teleop_cfg is None:
        return False
    if _cfg_type(robot_cfg) not in SONGLING_FOLLOWER_TYPES:
        return False
    if _cfg_type(teleop_cfg) not in OPENARM_LEADER_TYPES:
        return False

    robot_ports = _port_map(robot_cfg)
    teleop_ports = _port_map(teleop_cfg)
    if robot_ports is None or teleop_ports is None:
        return False
    if robot_ports.keys() != teleop_ports.keys():
        return False
    return all(robot_ports[key] == teleop_ports[key] for key in robot_ports)


def raise_if_songling_integrated_openarm_risk(
    *,
    robot_cfg: Any,
    teleop_cfg: Any,
    entrypoint: str,
) -> None:
    if not is_songling_integrated_openarm_risk(robot_cfg, teleop_cfg):
        return

    raise NotImplementedError(
        f"{entrypoint} refused to connect a software OpenArm leader to a Songling integrated chain. "
        "This hardware already teleoperates in hardware on the same CAN bus per side. "
        "Use the Songling raw-CAN / follower-direct tools instead, and do not run "
        "`openarm_leader` or `bi_openarm_leader` against the same CAN interfaces."
    )


def _normalize_songling_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _explicit_songling_installation_pos_code(value: Any) -> int:
    if value is None:
        return 0x00
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)

    normalized = _normalize_songling_text(value)
    if normalized in {"", "none", "null", "default", "keep", "preserve", "invalid", "0", "0x00"}:
        return 0x00
    if normalized in {"parallel", "horizontal", "center", "centre", "0x01", "1"}:
        return 0x01
    if normalized in {"left", "0x02", "2"}:
        return 0x02
    if normalized in {"right", "0x03", "3"}:
        return 0x03
    return -1


def songling_parameter_write_blocked_reason(action: str, *, details: str | None = None) -> str:
    message = (
        f"Songling/Piper parameter write blocked for safety: {action}. "
        "This fork no longer sends commands that may change stored arm parameters, "
        "installation compensation, teach-pendant settings, zero points, or leader/follower role."
    )
    if details:
        message = f"{message} {details}"
    return message


def raise_songling_parameter_write_blocked(action: str, *, details: str | None = None) -> None:
    raise RuntimeError(songling_parameter_write_blocked_reason(action, details=details))


def _songling_warn_once(logger: logging.Logger, key: str, message: str) -> None:
    if key in _SONGLING_SAFETY_WARNED_KEYS:
        return
    _SONGLING_SAFETY_WARNED_KEYS.add(key)
    logger.warning(message)


def sanitize_songling_installation_pos(
    *,
    logger: logging.Logger,
    requested: Any,
    entrypoint: str,
) -> int:
    requested_code = _explicit_songling_installation_pos_code(requested)
    if requested_code not in {0x00, -1}:
        _songling_warn_once(
            logger,
            f"{entrypoint}|installation_pos|{requested!r}",
            songling_parameter_write_blocked_reason(
                "installation position downlink",
                details=(
                    f"{entrypoint} ignored installation_pos={requested!r} and now always sends 0x00 "
                    "to avoid overwriting the arm's original installation compensation."
                ),
            ),
        )
    return 0x00


def sanitize_songling_gripper_set_zero(
    *,
    logger: logging.Logger,
    requested: Any,
    entrypoint: str,
) -> int:
    try:
        requested_code = int(requested)
    except Exception:
        requested_code = 0
    if requested_code != 0:
        raise_songling_parameter_write_blocked(
            "gripper zero-setting downlink",
            details=f"{entrypoint} requested gripper_set_zero={requested!r}. That write path is disabled.",
        )
    _ = logger
    return 0x00


def raise_if_songling_parameter_config_unsafe(*, cfg: Any, entrypoint: str) -> None:
    violations: list[str] = []

    installation_pos = getattr(cfg, "installation_pos", None)
    installation_code = _explicit_songling_installation_pos_code(installation_pos)
    if installation_code == -1:
        violations.append(f"unsupported installation_pos={installation_pos!r}")
    elif installation_code != 0x00:
        violations.append(
            f"installation_pos={installation_pos!r} would send a persistent installation-mode hint to the arm"
        )

    try:
        gripper_set_zero = int(getattr(cfg, "gripper_set_zero", 0) or 0)
    except Exception:
        gripper_set_zero = 0
    if gripper_set_zero != 0:
        violations.append(f"gripper_set_zero={gripper_set_zero!r} would request a zero-setting write")

    if bool(getattr(cfg, "auto_configure_master_slave_on_connect", False)):
        violations.append("auto_configure_master_slave_on_connect=true would send MasterSlaveConfig")

    role = _normalize_songling_text(getattr(cfg, "leader_follower_role", None))
    if role not in {"", "none", "null", "default", "standalone", "independent"}:
        violations.append(f"leader_follower_role={getattr(cfg, 'leader_follower_role', None)!r} would reconfigure arm role")

    if violations:
        raise_songling_parameter_write_blocked(
            "unsafe Songling runtime config",
            details=f"{entrypoint} refused config with: {'; '.join(violations)}",
        )
