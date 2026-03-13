#!/usr/bin/env python

"""Rerun-only live visualization for Songling ALOHA profile.

This script reuses LeRobot's bimanual OpenArm follower/leader configs and logs:
- 3 camera streams: left_high, left_elbow, right_elbow
- per-joint cmd / obs / tracking error scalars
- optional raw CAN traffic statistics in bring-up mode

Unlike earlier versions, this script does not use OpenCV GUI or compose text
dashboards. Visualization is done exclusively through Rerun.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

can = None
draccus = None
np = None
rr = None

make_cameras_from_configs = None
TeleoperateConfig = None
RobotConfig = None
TeleoperatorConfig = None
precise_sleep = None
init_rerun = None
make_robot_from_config = None
make_teleoperator_from_config = None

CAMERA_KEYS = ("left_high", "left_elbow", "right_elbow")


def _patch_multiprocess_resource_tracker() -> None:
    """Work around multiprocess<->Python3.12 RLock private API mismatch.

    Some environments hit:
      AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
    during interpreter shutdown in multiprocess.resource_tracker.__del__.
    """
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


def _load_runtime_deps(raw_can_mode: bool) -> None:
    global can, draccus, np, rr
    global make_cameras_from_configs, TeleoperateConfig, precise_sleep, init_rerun
    global make_robot_from_config, make_teleoperator_from_config
    global RobotConfig, TeleoperatorConfig

    try:
        import draccus as _draccus  # type: ignore
        import numpy as _np  # type: ignore
        import rerun as _rr  # type: ignore

        from lerobot.cameras.utils import make_cameras_from_configs as _make_cameras_from_configs
        from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig as _TeleoperateConfig
        from lerobot.robots import RobotConfig as _RobotConfig
        from lerobot.teleoperators import TeleoperatorConfig as _TeleoperatorConfig
        from lerobot.utils.robot_utils import precise_sleep as _precise_sleep
        from lerobot.utils.visualization_utils import init_rerun as _init_rerun

        # Register built-in choices used by profile config.
        from lerobot.robots import bi_openarm_follower as _bi_openarm_follower  # noqa: F401
        from lerobot.robots import make_robot_from_config as _make_robot_from_config
        from lerobot.teleoperators import bi_openarm_leader as _bi_openarm_leader  # noqa: F401
        from lerobot.teleoperators import make_teleoperator_from_config as _make_teleoperator_from_config
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependencies for Songling ALOHA visualization. "
            "Activate your LeRobot env (e.g. `conda activate lerobot_v050`) and install "
            "`pip install -e '.[openarms]'`."
        ) from e

    draccus = _draccus
    np = _np
    rr = _rr
    make_cameras_from_configs = _make_cameras_from_configs
    TeleoperateConfig = _TeleoperateConfig
    RobotConfig = _RobotConfig
    TeleoperatorConfig = _TeleoperatorConfig
    precise_sleep = _precise_sleep
    init_rerun = _init_rerun
    make_robot_from_config = _make_robot_from_config
    make_teleoperator_from_config = _make_teleoperator_from_config

    if raw_can_mode:
        try:
            import can as _can  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "python-can is required for --raw-can-mode. Install it in your LeRobot environment."
            ) from e
        can = _can


def _joint_sort_key(name: str) -> tuple[int, int]:
    if name.startswith("joint_"):
        m = re.search(r"joint_(\d+)\.pos$", name)
        if m:
            return (0, int(m.group(1)))
    if name == "gripper.pos":
        return (1, 0)
    return (2, 0)


def _extract_side_pos(data: dict[str, float] | None, side_prefix: str) -> dict[str, float]:
    if not data:
        return {}
    out: dict[str, float] = {}
    prefix = f"{side_prefix}_"
    for key, value in data.items():
        if key.startswith(prefix) and key.endswith(".pos"):
            out[key.removeprefix(prefix)] = float(value)
    return out


def _extract_all_pos(data: dict[str, float] | None) -> dict[str, float]:
    if not data:
        return {}
    out: dict[str, float] = {}
    for key, value in data.items():
        if key.endswith(".pos"):
            out[key] = float(value)
    return out


def _safe_err(follower_val: float, leader_val: float) -> float:
    if math.isnan(follower_val) or math.isnan(leader_val):
        return float("nan")
    return follower_val - leader_val


def _connect_with_timeout(name: str, device, calibrate: bool, timeout_s: float) -> None:
    error_holder: list[Exception] = []
    done = threading.Event()

    def _target() -> None:
        try:
            device.connect(calibrate=calibrate)
        except Exception as exc:  # nosec B110
            error_holder.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    if not done.wait(timeout_s):
        # Best-effort cleanup on timeout to avoid leaked CAN handles.
        try:
            device.disconnect()
        except Exception:  # nosec B110
            pass
        raise TimeoutError(f"{name}.connect() timed out after {timeout_s:.1f}s")
    if error_holder:
        # Best-effort cleanup on failed connect.
        try:
            device.disconnect()
        except Exception:  # nosec B110
            pass
        raise error_holder[0]


def _camera_configs_from_cfg(cfg) -> dict[str, Any]:
    robot_cfg = getattr(cfg, "robot", None)
    left_arm_cfg = getattr(robot_cfg, "left_arm_config", None)
    right_arm_cfg = getattr(robot_cfg, "right_arm_config", None)
    if left_arm_cfg is None or right_arm_cfg is None:
        raise ValueError(
            "--raw-can-mode requires a bimanual robot config with robot.left_arm_config and robot.right_arm_config."
        )

    left_cams = getattr(left_arm_cfg, "cameras", None)
    right_cams = getattr(right_arm_cfg, "cameras", None)
    if not isinstance(left_cams, dict) or not isinstance(right_cams, dict):
        raise ValueError(
            "--raw-can-mode requires robot.left_arm_config.cameras and robot.right_arm_config.cameras mappings."
        )
    missing = []
    if "high" not in left_cams:
        missing.append("robot.left_arm_config.cameras.high")
    if "elbow" not in left_cams:
        missing.append("robot.left_arm_config.cameras.elbow")
    if "elbow" not in right_cams:
        missing.append("robot.right_arm_config.cameras.elbow")
    if missing:
        raise ValueError(f"Missing required camera config(s): {', '.join(missing)}")
    return {
        "left_high": left_cams["high"],
        "left_elbow": left_cams["elbow"],
        "right_elbow": right_cams["elbow"],
    }


def _open_raw_can_bus(interface: str, bitrate: int, use_fd: bool):
    kwargs: dict[str, Any] = {"channel": interface, "interface": "socketcan", "bitrate": bitrate}
    if use_fd:
        kwargs["fd"] = True
    return can.interface.Bus(**kwargs)


def _poll_can_state(bus, state: dict, max_msgs: int = 256) -> None:
    got = 0
    while got < max_msgs:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            break
        got += 1
        state["total_msgs"] += 1
        now = time.time()
        state["recent_timestamps"].append(now)
        state["latest"][msg.arbitration_id] = now

    if got > 0:
        latest_items = sorted(state["latest"].items(), key=lambda kv: kv[1], reverse=True)
        state["latest_by_id"] = latest_items


def _rx_hz(ts_q: deque[float]) -> float:
    if not ts_q:
        return 0.0
    now = time.time()
    while ts_q and (now - ts_q[0]) > 1.0:
        ts_q.popleft()
    return float(len(ts_q))


def _log_image(path: str, frame, compress: bool) -> None:
    entity = rr.Image(frame).compress() if compress else rr.Image(frame)
    rr.log(path, entity=entity)


def _log_scalar(path: str, value: float) -> None:
    if math.isfinite(value):
        rr.log(path, rr.Scalars(float(value)))


def _run_raw_can_mode(
    cfg,
    fps_override: int | None,
    rerun_ip: str | None,
    rerun_port: int | None,
    camera_max_age_ms: int,
    can_poll_max_msgs: int,
    rerun_log_fps: float,
    compress_images: bool,
    camera_retry_timeout_ms: int,
    camera_reconnect_stale_count: int,
    camera_reconnect_cooldown_s: float,
    graceful_rerun_shutdown: bool,
) -> None:
    fps = fps_override if fps_override is not None else cfg.fps

    cameras = {}
    left_bus = None
    right_bus = None
    rerun_started = False
    try:
        cameras = make_cameras_from_configs(_camera_configs_from_cfg(cfg))
        for cam_name, cam in cameras.items():
            cam.connect(warmup=True)
            print(f"[INFO] Connected camera: {cam_name}", flush=True)

        left_iface = cfg.robot.left_arm_config.port
        right_iface = cfg.robot.right_arm_config.port
        left_bitrate = cfg.robot.left_arm_config.can_bitrate
        right_bitrate = cfg.robot.right_arm_config.can_bitrate
        left_use_fd = cfg.robot.left_arm_config.use_can_fd
        right_use_fd = cfg.robot.right_arm_config.use_can_fd

        left_bus = _open_raw_can_bus(left_iface, left_bitrate, left_use_fd)
        right_bus = _open_raw_can_bus(right_iface, right_bitrate, right_use_fd)

        init_rerun(session_name="songling_aloha_raw_can", ip=rerun_ip, port=rerun_port)
        rerun_started = True

        left_state = {
            "total_msgs": 0,
            "recent_timestamps": deque(maxlen=4096),
            "latest": {},
            "latest_by_id": [],
        }
        right_state = {
            "total_msgs": 0,
            "recent_timestamps": deque(maxlen=4096),
            "latest": {},
            "latest_by_id": [],
        }

        last_frames: dict[str, np.ndarray] = {}
        last_frame_ts: dict[str, float] = {}
        stale_counts: dict[str, int] = {}
        last_reconnect_ts: dict[str, float] = {}
        loop_hz = 0.0
        frame_idx = 0
        last_rerun_log_s = 0.0
        while True:
            t0 = time.perf_counter()

            _poll_can_state(left_bus, left_state, max_msgs=can_poll_max_msgs)
            _poll_can_state(right_bus, right_state, max_msgs=can_poll_max_msgs)

            now_s = time.time()
            left_hz = _rx_hz(left_state["recent_timestamps"])
            right_hz = _rx_hz(right_state["recent_timestamps"])

            if (now_s - last_rerun_log_s) >= max(1.0 / max(rerun_log_fps, 0.1), 0.001):
                rr.set_time("frame", sequence=frame_idx)

                for cam_key, cam in cameras.items():
                    stale = 0.0
                    stale_ms = 0.0
                    try:
                        frame = cam.read_latest(max_age_ms=camera_max_age_ms)
                        last_frames[cam_key] = frame
                        last_frame_ts[cam_key] = time.perf_counter()
                        stale_counts[cam_key] = 0
                    except TimeoutError:
                        # read_latest() can be stale when CPU is busy.
                        # Try a short blocking read to fetch a fresh frame before falling back.
                        try:
                            frame = cam.async_read(timeout_ms=max(camera_retry_timeout_ms, 1))
                            last_frames[cam_key] = frame
                            last_frame_ts[cam_key] = time.perf_counter()
                            stale_counts[cam_key] = 0
                        except Exception:
                            stale = 1.0
                            stale_ms = (
                                (time.perf_counter() - last_frame_ts[cam_key]) * 1e3
                                if cam_key in last_frame_ts
                                else -1.0
                            )
                            frame = last_frames.get(cam_key)
                            stale_counts[cam_key] = stale_counts.get(cam_key, 0) + 1
                    except Exception:
                        stale = 1.0
                        stale_ms = (
                            (time.perf_counter() - last_frame_ts[cam_key]) * 1e3 if cam_key in last_frame_ts else -1.0
                        )
                        frame = last_frames.get(cam_key)
                        stale_counts[cam_key] = stale_counts.get(cam_key, 0) + 1

                    if stale > 0.0 and camera_reconnect_stale_count > 0:
                        should_reconnect = stale_counts.get(cam_key, 0) >= camera_reconnect_stale_count
                        cooldown_ok = (time.time() - last_reconnect_ts.get(cam_key, 0.0)) >= max(
                            camera_reconnect_cooldown_s, 0.0
                        )
                        if should_reconnect and cooldown_ok:
                            print(
                                f"[WARN] Camera '{cam_key}' stale x{stale_counts.get(cam_key, 0)}; reconnecting...",
                                flush=True,
                            )
                            try:
                                cam.disconnect()
                            except Exception:  # nosec B110
                                pass
                            try:
                                cam.connect(warmup=True)
                                stale_counts[cam_key] = 0
                                last_reconnect_ts[cam_key] = time.time()
                                print(f"[INFO] Camera '{cam_key}' reconnected.", flush=True)
                            except Exception as reconnect_exc:  # nosec B110
                                last_reconnect_ts[cam_key] = time.time()
                                print(
                                    f"[WARN] Camera '{cam_key}' reconnect failed: {reconnect_exc}",
                                    flush=True,
                                )

                    _log_scalar(f"cameras/{cam_key}/stale", stale)
                    _log_scalar(f"cameras/{cam_key}/stale_ms", stale_ms)
                    _log_scalar(f"cameras/{cam_key}/stale_count", float(stale_counts.get(cam_key, 0)))
                    if frame is not None:
                        _log_image(f"cameras/{cam_key}", frame, compress=compress_images)

                _log_scalar("can/left/total_msgs", float(left_state["total_msgs"]))
                _log_scalar("can/right/total_msgs", float(right_state["total_msgs"]))
                _log_scalar("can/left/rx_hz", float(left_hz))
                _log_scalar("can/right/rx_hz", float(right_hz))
                if left_state["latest_by_id"]:
                    _log_scalar("can/left/latest_id", float(left_state["latest_by_id"][0][0]))
                if right_state["latest_by_id"]:
                    _log_scalar("can/right/latest_id", float(right_state["latest_by_id"][0][0]))
                _log_scalar("perf/loop_hz", float(loop_hz))
                last_rerun_log_s = now_s

            dt = time.perf_counter() - t0
            loop_hz = 1.0 / dt if dt > 0 else 0.0
            frame_idx += 1
            precise_sleep(max((1.0 / fps) - dt, 0.0))
    except KeyboardInterrupt:
        print("[INFO] Received Ctrl+C, stopping raw CAN visualization.", flush=True)
    finally:
        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception:  # nosec B110
                pass
        if left_bus is not None:
            try:
                left_bus.shutdown()
            except Exception:  # nosec B110
                pass
        if right_bus is not None:
            try:
                right_bus.shutdown()
            except Exception:  # nosec B110
                pass
        if rerun_started:
            try:
                if graceful_rerun_shutdown:
                    rr.rerun_shutdown()
            except Exception:  # nosec B110
                pass


def _build_passthrough_action_from_obs(obs: dict[str, float]) -> dict[str, float]:
    action: dict[str, float] = {}
    for key, value in obs.items():
        if key.endswith(".pos"):
            action[key] = float(value)
    return action


def _iter_camera_frames_from_obs(obs: dict[str, Any]) -> list[tuple[str, Any]]:
    """Find camera-like arrays in observation.

    Keep Songling-specific ordering first when those keys exist, then include any
    remaining ndarray-like entries to support generic LeRobot camera names.
    """
    frames: list[tuple[str, Any]] = []
    used: set[str] = set()

    for cam_key in CAMERA_KEYS:
        frame = obs.get(cam_key)
        if frame is None:
            continue
        frames.append((cam_key, frame))
        used.add(cam_key)

    for key, value in obs.items():
        if key in used:
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            frames.append((key, value))
    return frames


def _is_bimanual_cfg(cfg) -> bool:
    robot_has_arms = hasattr(getattr(cfg, "robot", None), "left_arm_config") and hasattr(
        getattr(cfg, "robot", None), "right_arm_config"
    )
    teleop_has_arms = hasattr(getattr(cfg, "teleop", None), "left_arm_config") and hasattr(
        getattr(cfg, "teleop", None), "right_arm_config"
    )
    return bool(robot_has_arms and teleop_has_arms)


def _run_teleop_mode(
    cfg,
    fps_override: int | None,
    skip_teleop: bool,
    no_send_action: bool,
    calibrate_on_connect: bool,
    connect_timeout_s: float,
    rerun_ip: str | None,
    rerun_port: int | None,
    rerun_log_fps: float,
    compress_images: bool,
    graceful_rerun_shutdown: bool,
) -> None:
    # Songling integrated master/slave chains teleoperate in hardware and share one CAN port per side.
    # The software pattern `bi_openarm_leader` + `bi_openarm_follower` on the same CAN interfaces
    # is not a valid runtime path for this hardware. Failing fast here avoids misleading handshake errors.
    if cfg.teleop is not None and _is_bimanual_cfg(cfg):
        same_left_port = getattr(cfg.teleop.left_arm_config, "port", None) == getattr(cfg.robot.left_arm_config, "port", None)
        same_right_port = getattr(cfg.teleop.right_arm_config, "port", None) == getattr(
            cfg.robot.right_arm_config, "port", None
        )
        if same_left_port and same_right_port:
            raise NotImplementedError(
                "This config models an integrated Songling chain where leader+follower share the same CAN port on each side. "
                "Do not run software teleop mode here. Use `--raw-can-mode` for bring-up/visualization, or implement a "
                "dedicated Songling robot adapter in LeRobot before enabling record/teleoperate/replay."
            )

    if cfg.teleop is None and not skip_teleop:
        raise ValueError("teleop config is required in the provided config file.")

    fps = fps_override if fps_override is not None else cfg.fps
    teleop = None if skip_teleop else make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop_connected = False
    robot_connected = False
    rerun_started = False
    try:
        _connect_with_timeout("robot", robot, calibrate=calibrate_on_connect, timeout_s=connect_timeout_s)
        robot_connected = True
        if teleop is not None:
            _connect_with_timeout("teleop", teleop, calibrate=calibrate_on_connect, timeout_s=connect_timeout_s)
            teleop_connected = True

        init_rerun(session_name="songling_aloha_teleop", ip=rerun_ip, port=rerun_port)
        rerun_started = True

        loop_hz = 0.0
        frame_idx = 0
        last_rerun_log_s = 0.0
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            if teleop is not None:
                action = teleop.get_action()
                action_source = 0.0  # teleop.get_action()
            else:
                action = _build_passthrough_action_from_obs(obs)
                action_source = 1.0  # passthrough_from_obs

            if not no_send_action:
                _ = robot.send_action(action)

            now_s = time.time()
            if (now_s - last_rerun_log_s) >= max(1.0 / max(rerun_log_fps, 0.1), 0.001):
                rr.set_time("frame", sequence=frame_idx)

                for cam_key, frame in _iter_camera_frames_from_obs(obs):
                    _log_image(f"cameras/{cam_key}", frame, compress=compress_images)

                cmd_pos = _extract_all_pos(action)
                obs_pos = _extract_all_pos(obs)
                all_joint_names = sorted(set(cmd_pos.keys()) | set(obs_pos.keys()), key=_joint_sort_key)
                for joint in all_joint_names:
                    cmd_val = cmd_pos.get(joint, float("nan"))
                    obs_val = obs_pos.get(joint, float("nan"))
                    err_val = _safe_err(obs_val, cmd_val)
                    _log_scalar(f"teleop/cmd/{joint}", float(cmd_val))
                    _log_scalar(f"follower/obs/{joint}", float(obs_val))
                    _log_scalar(f"tracking/err/{joint}", float(err_val))

                _log_scalar("meta/action_source", float(action_source))
                _log_scalar("perf/loop_hz", float(loop_hz))
                last_rerun_log_s = now_s

            dt = time.perf_counter() - t0
            loop_hz = 1.0 / dt if dt > 0 else 0.0
            frame_idx += 1
            precise_sleep(max((1.0 / fps) - dt, 0.0))
    except KeyboardInterrupt:
        print("[INFO] Received Ctrl+C, stopping teleop visualization.", flush=True)
    finally:
        if teleop is not None and teleop_connected:
            try:
                teleop.disconnect()
            except Exception:  # nosec B110
                pass
        if robot_connected:
            try:
                robot.disconnect()
            except Exception:  # nosec B110
                pass
        if rerun_started:
            try:
                if graceful_rerun_shutdown:
                    rr.rerun_shutdown()
            except Exception:  # nosec B110
                pass


def _parse_songling_config(config_path: str | None, unknown_args: list[str]):
    """Parse either TeleoperateConfig or RecordConfig from a shared Songling YAML.

    This enables a single `teleop.yaml` to carry both teleop and dataset defaults.
    """
    try:
        # NOTE: Songling's unified YAML contains both teleoperate-like fields and record-only fields
        # (dataset/play_sounds/resume). LeRobot's TeleoperateConfig/RecordConfig are strict and
        # reject unknown keys, so we parse a local superset config here.
        @dataclass
        class _SonglingUnifiedConfig:
            robot: RobotConfig
            teleop: TeleoperatorConfig | None = None
            fps: int = 60
            display_data: bool = False
            display_ip: str | None = None
            display_port: int | None = None
            display_compressed_images: bool = False
            dataset: Any | None = None
            play_sounds: bool | None = None
            resume: bool | None = None

        return draccus.parse(_SonglingUnifiedConfig, config_path=config_path, args=unknown_args)
    except Exception as unified_exc:
        # Backward-compat: accept pure TeleoperateConfig yaml (no dataset fields).
        try:
            return draccus.parse(TeleoperateConfig, config_path=config_path, args=unknown_args)
        except Exception:
            raise unified_exc


def _resolve_rerun_endpoint(args: argparse.Namespace, cfg) -> tuple[str | None, int | None]:
    # CLI explicit overrides take precedence, then fallback to config defaults.
    cli_ip = args.rerun_ip if args.rerun_ip is not None else args.display_ip
    cli_port = args.rerun_port if args.rerun_port is not None else args.display_port
    cfg_ip = getattr(cfg, "display_ip", None)
    cfg_port = getattr(cfg, "display_port", None)

    rerun_ip = cli_ip if cli_ip is not None else cfg_ip
    rerun_port = cli_port if cli_port is not None else cfg_port
    if (rerun_ip is None) != (rerun_port is None):
        raise ValueError(
            "Please set both display host and port together "
            "(--rerun-ip/--rerun-port or --display_ip/--display_port, "
            "or provide both in config YAML)."
        )
    return rerun_ip, rerun_port


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun-only live visualization for Songling ALOHA profile.")
    parser.add_argument(
        "--config-path",
        "--config_path",
        type=Path,
        default=Path("examples/songling_aloha/teleop.yaml"),
        help="Path to teleop config yaml.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override FPS from config.")
    parser.add_argument(
        "--raw-can-mode",
        action="store_true",
        help="Do not use LeRobot robot/teleop connect loop. Visualize 3 cameras + raw CAN traffic in real time.",
    )
    parser.add_argument(
        "--skip-teleop",
        action="store_true",
        help="Do not connect/read teleop device. Use follower observation as passthrough action source.",
    )
    parser.add_argument(
        "--no-send-action",
        action="store_true",
        help="Read and visualize only, without sending action to robot.",
    )
    parser.add_argument(
        "--calibrate-on-connect",
        action="store_true",
        help="Enable calibration flow during connect(). Default is off to avoid blocking prompts.",
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=8.0,
        help="Timeout for each device connect() call.",
    )
    parser.add_argument(
        "--rerun-ip",
        default=None,
        help="Rerun server IP. If omitted, local viewer spawn is used.",
    )
    parser.add_argument(
        "--rerun-port",
        type=int,
        default=None,
        help="Rerun server port (used with --rerun-ip).",
    )
    parser.add_argument(
        "--display-ip",
        "--display_ip",
        dest="display_ip",
        default=None,
        help="Alias of --rerun-ip for lerobot-teleoperate compatible flags.",
    )
    parser.add_argument(
        "--display-port",
        "--display_port",
        dest="display_port",
        type=int,
        default=None,
        help="Alias of --rerun-port for lerobot-teleoperate compatible flags.",
    )
    parser.add_argument(
        "--rerun-log-fps",
        type=float,
        default=5.0,
        help="Rerun logging frequency limit to reduce UI lag.",
    )
    parser.add_argument(
        "--camera-max-age-ms",
        type=int,
        default=500,
        help="Max allowed age for read_latest() frames in raw CAN mode before marking as stale.",
    )
    parser.add_argument(
        "--compress-images",
        action="store_true",
        help="Compress images before logging to Rerun. Default is auto (enabled for remote Rerun server).",
    )
    parser.add_argument(
        "--display-compressed-images",
        "--display_compressed_images",
        dest="display_compressed_images",
        action="store_true",
        help="Alias of --compress-images for lerobot-teleoperate compatible flags.",
    )
    parser.add_argument(
        "--can-poll-max-msgs",
        type=int,
        default=64,
        help="Max CAN frames to process per loop per interface (raw mode). Lower for less lag.",
    )
    parser.add_argument(
        "--camera-retry-timeout-ms",
        type=int,
        default=60,
        help="When a camera frame is stale, wait this long for a fresh async frame before reusing old frame.",
    )
    parser.add_argument(
        "--camera-reconnect-stale-count",
        type=int,
        default=24,
        help="Reconnect a camera when it has this many consecutive stale/read failures. Set <=0 to disable.",
    )
    parser.add_argument(
        "--camera-reconnect-cooldown-s",
        type=float,
        default=2.0,
        help="Minimum seconds between reconnect attempts for the same camera.",
    )
    parser.add_argument(
        "--skip-rerun-shutdown",
        action="store_true",
        help="Skip explicit rr.rerun_shutdown(). Useful when remote Rerun frequently disconnects during shutdown.",
    )
    args, unknown_args = parser.parse_known_args()

    _patch_multiprocess_resource_tracker()
    _load_runtime_deps(raw_can_mode=args.raw_can_mode)

    # Allow lerobot-teleoperate-style overrides (e.g. --robot.*, --teleop.*, --display_*).
    config_path = str(args.config_path) if args.config_path else None
    cfg = _parse_songling_config(config_path=config_path, unknown_args=unknown_args)
    rerun_ip, rerun_port = _resolve_rerun_endpoint(args=args, cfg=cfg)

    compress_images = bool(
        args.compress_images
        or args.display_compressed_images
        or getattr(cfg, "display_compressed_images", False)
        or (rerun_ip is not None and rerun_port is not None)
    )

    # For remote Rerun sinks, explicit graceful shutdown often prints noisy
    # "gracefully disconnected" gRPC errors while remaining functionally safe.
    remote_sink = rerun_ip is not None and rerun_port is not None
    graceful_rerun_shutdown = (not args.skip_rerun_shutdown) and (not remote_sink)

    if args.raw_can_mode:
        _run_raw_can_mode(
            cfg=cfg,
            fps_override=args.fps,
            rerun_ip=rerun_ip,
            rerun_port=rerun_port,
            camera_max_age_ms=args.camera_max_age_ms,
            can_poll_max_msgs=args.can_poll_max_msgs,
            rerun_log_fps=args.rerun_log_fps,
            compress_images=compress_images,
            camera_retry_timeout_ms=args.camera_retry_timeout_ms,
            camera_reconnect_stale_count=args.camera_reconnect_stale_count,
            camera_reconnect_cooldown_s=args.camera_reconnect_cooldown_s,
            graceful_rerun_shutdown=graceful_rerun_shutdown,
        )
        return

    _run_teleop_mode(
        cfg=cfg,
        fps_override=args.fps,
        skip_teleop=args.skip_teleop,
        no_send_action=args.no_send_action,
        calibrate_on_connect=args.calibrate_on_connect,
        connect_timeout_s=args.connect_timeout_s,
        rerun_ip=rerun_ip,
        rerun_port=rerun_port,
        rerun_log_fps=args.rerun_log_fps,
        compress_images=compress_images,
        graceful_rerun_shutdown=graceful_rerun_shutdown,
    )


if __name__ == "__main__":
    main()
