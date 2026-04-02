#!/usr/bin/env python

"""Record Songling ALOHA demonstrations into LeRobot format.

This recorder follows the current integrated-chain semantics documented in
`piper_sdk/asserts/double_piper.MD` and used by the live Songling monitor:

- each side is one master/slave pair sharing one CAN interface
- `observation.state` comes from follower/slave feedback
- `action` comes only from master/leader control echo
- if control echo is missing or stale, recorder can either drop that loop or buffer observation until labels recover
"""

from __future__ import annotations

import argparse
from collections import deque
import glob
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.bi_songling_follower import BiSonglingFollower
from lerobot.robots.bi_songling_follower.config_bi_songling_follower import BiSonglingFollowerConfig
from lerobot.robots.songling_follower import SonglingFollowerConfigBase
from lerobot.robots.songling_follower.protocol import (
    ARM_GRIPPER_CTRL,
    ARM_GRIPPER_FEEDBACK,
    ARM_JOINT_CTRL_12,
    ARM_JOINT_CTRL_34,
    ARM_JOINT_CTRL_56,
    ARM_JOINT_FEEDBACK_12,
    ARM_JOINT_FEEDBACK_34,
    ARM_JOINT_FEEDBACK_56,
)
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path("examples/songling_aloha/teleop.yaml")
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "songling_aloha"
CAPTURE_SUMMARY_FILENAME = "songling_recording_summary.json"
DEFAULT_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
OBSERVATION_FRAME_IDS = [ARM_JOINT_FEEDBACK_12, ARM_JOINT_FEEDBACK_34, ARM_JOINT_FEEDBACK_56, ARM_GRIPPER_FEEDBACK]
ACTION_FRAME_IDS = [ARM_JOINT_CTRL_12, ARM_JOINT_CTRL_34, ARM_JOINT_CTRL_56, ARM_GRIPPER_CTRL]
DEFAULT_COMMAND_FRESHNESS_S = 0.35
DEFAULT_STALE_GRACE_S = 0.0
MISSING_LABEL_POLICY_CHOICES = ("drop", "buffer-observation")
DEFAULT_MISSING_LABEL_POLICY = "buffer-observation"
DEFAULT_BUFFERED_OBSERVATION_MAX_FRAMES = 90
DEFAULT_START_ON_FIRST_ACTION = True
DEFAULT_EPISODE_START_MIN_FRESH_FRAMES = 1
DEFAULT_EPISODE_END_TAIL_S = 0.5
CAMERA_CONNECT_RETRIES = 3
CAMERA_CONNECT_RETRY_WARMUP_S = 3.0
CAMERA_CONNECT_RETRY_SETTLE_S = 0.5
VOICE_EVENT_GAP_S = 0.12
LOCAL_PIPER_BINARY = PROJECT_ROOT / "third_party" / "piper" / "piper" / "piper"
LOCAL_PIPER_MODELS_DIR = PROJECT_ROOT / "third_party" / "piper" / "models"
VOICE_ENGINE_CHOICES = ("auto", "piper", "log-say", "off")
_VOICE_WARN_ONCE_KEYS: set[str] = set()
_PIPER_MODEL_DISCOVERY_CACHE: dict[str, str | None] = {}
_TONE_WAV_CACHE: dict[str, bytes] = {}
_PIPER_WAV_CACHE: dict[tuple[str, str, str, int | None], bytes] = {}
_VOICE_PLAY_LOCK = threading.Lock()
_VOICE_OUTPUT_LOCK = threading.Lock()
_VOICE_EVENT_TS_LOCK = threading.Lock()
_LAST_VOICE_EVENT_TS: dict[str, float] = {}


@dataclass
class DatasetOptions:
    repo_id: str
    single_task: str
    root: Path
    fps: int
    episode_time_s: float
    reset_time_s: float
    num_episodes: int
    video: bool
    push_to_hub: bool
    private: bool
    tags: list[str] | None
    num_image_writer_processes: int
    num_image_writer_threads_per_camera: int
    video_encoding_batch_size: int
    vcodec: str
    streaming_encoding: bool
    encoder_queue_maxsize: int
    encoder_threads: int | None
    resume: bool
    auto_increment_root: bool
    play_sounds: bool
    voice_lang: str
    voice_rate: int
    voice_engine: str
    voice_piper_model: str | None
    voice_piper_binary: str | None
    voice_piper_speaker: int | None


@dataclass
class DisplayOptions:
    display_data: bool
    display_ip: str | None
    display_port: int | None
    display_compressed_images: bool


@dataclass
class CameraHealthState:
    last_frames: dict[str, np.ndarray] = field(default_factory=dict)
    stale_counts: dict[str, int] = field(default_factory=dict)
    identical_counts: dict[str, int] = field(default_factory=dict)
    last_signatures: dict[str, int] = field(default_factory=dict)
    last_reconnect_ts: dict[str, float] = field(default_factory=dict)


@dataclass
class CommandEchoStatus:
    has_control: bool
    is_valid: bool
    is_fresh: bool
    age_s: float | None
    missing_joints: list[str] = field(default_factory=list)


@dataclass
class BufferedObservation:
    loop_index: int
    observation_state: np.ndarray
    camera_frames: dict[str, np.ndarray]
    left_status: CommandEchoStatus
    right_status: CommandEchoStatus


def _patch_multiprocess_resource_tracker() -> None:
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


def _parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false.")


def _parse_optional_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    return int(value)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tags(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        tags = [str(item).strip() for item in value if str(item).strip()]
        return tags or None
    tags = [item for item in _parse_csv(str(value)) if item]
    return tags or None


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return _parse_cli_bool(str(value))


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _coerce_voice_engine(value: Any) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "disabled": "off",
        "logsay": "log-say",
    }
    engine = aliases.get(normalized, normalized)
    if engine not in VOICE_ENGINE_CHOICES:
        raise ValueError(
            f"Unsupported voice engine: {value}. Use one of: {', '.join(VOICE_ENGINE_CHOICES)}."
        )
    return engine


def _show_error_popup(title: str, message: str) -> bool:
    """Best-effort GUI popup for fatal preflight errors.

    Returns True if a popup was shown; False if GUI is unavailable.
    """

    has_display = bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    if not has_display and platform.system().lower() not in {"windows", "darwin"}:
        return False

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        messagebox.showerror(title, message, parent=root)
        root.destroy()
        return True
    except Exception:
        return False


def _query_interface_up_status(interface_name: str) -> tuple[bool | None, str]:
    """Return (is_up, detail). is_up=None means unknown (unable to determine)."""

    if not interface_name:
        return False, "empty interface name"

    sysfs_path = Path("/sys/class/net") / interface_name
    if not sysfs_path.exists():
        return False, f"interface '{interface_name}' does not exist"

    try:
        result = subprocess.run(
            ["ip", "-details", "link", "show", "dev", interface_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        result = None

    if result is not None and result.returncode == 0 and result.stdout:
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        first_line = lines[0] if lines else ""

        flags = ""
        if "<" in first_line and ">" in first_line:
            flags = first_line.split("<", 1)[1].split(">", 1)[0]
        flag_set = {item.strip().upper() for item in flags.split(",") if item.strip()}
        has_up_flag = "UP" in flag_set

        can_state = None
        for line in lines:
            if line.startswith("can state "):
                can_state = line.split("can state ", 1)[1].split()[0].strip()
                break

        if not has_up_flag:
            return False, f"flags={flags or 'n/a'}"
        if can_state is not None and can_state.upper() in {"STOPPED", "BUS-OFF"}:
            return False, f"flags={flags or 'n/a'}, can_state={can_state}"
        return True, f"flags={flags or 'n/a'}, can_state={can_state or 'n/a'}"

    # Fallback: sysfs operstate (for CAN this may be 'unknown' when interface is UP).
    try:
        operstate = (sysfs_path / "operstate").read_text(encoding="utf-8").strip().lower()
    except Exception:
        operstate = "unknown"

    if operstate in {"up", "unknown"}:
        return True, f"operstate={operstate}"
    if operstate in {"down", "dormant", "notpresent", "lowerlayerdown"}:
        return False, f"operstate={operstate}"
    return None, f"operstate={operstate}"


def _ensure_required_can_interfaces_up(interface_names: list[str]) -> None:
    checked: list[tuple[str, bool | None, str]] = []
    failed: list[tuple[str, str]] = []

    for ifname in interface_names:
        is_up, detail = _query_interface_up_status(ifname)
        checked.append((ifname, is_up, detail))
        if is_up is not True:
            failed.append((ifname, detail))

    if not failed:
        return

    detail_lines = [f"- {name}: {reason}" for name, reason in failed]
    message = (
        "检测到必需 CAN 口未处于 UP 状态，已停止采集。\n\n"
        "请先重新拉起 CAN 接口后再重试，例如：\n"
        "sudo ip link set can0 up type can bitrate 1000000\n"
        "sudo ip link set can1 up type can bitrate 1000000\n\n"
        "检查结果：\n"
        + "\n".join(detail_lines)
    )

    print("[ERROR] CAN interface preflight failed:")
    for ifname, is_up, detail in checked:
        print(f"[ERROR]   {ifname}: up={is_up} ({detail})")

    popup_shown = _show_error_popup("Songling 采集前检查失败", message)
    if not popup_shown:
        print(message)

    raise RuntimeError("CAN interfaces are not UP. Aborting recording.")


def _warn_voice_once(key: str, message: str) -> None:
    if key in _VOICE_WARN_ONCE_KEYS:
        return
    _VOICE_WARN_ONCE_KEYS.add(key)
    print(message)


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at `{path}`, got {type(value)}.")
    return value


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _next_available_directory(root_path: Path) -> Path:
    if not root_path.exists():
        return root_path

    parent = root_path.parent
    stem = root_path.name
    match = re.match(r"^(.*?)(?:_(\d+))?$", stem)
    if match:
        base = match.group(1)
        number = match.group(2)
    else:
        base = stem
        number = None

    width = max(len(number), 3) if number is not None else 3
    idx = int(number) if number is not None else 0
    while True:
        idx += 1
        candidate = parent / f"{base}_{idx:0{width}d}"
        if not candidate.exists():
            return candidate


def _ensure_local_root_is_writable(root_value: str | Path, resume: bool, auto_increment_root: bool) -> Path:
    root_path = Path(root_value).expanduser()
    if not root_path.is_absolute():
        root_path = (Path.cwd() / root_path).resolve()

    parent = root_path.parent
    existing_ancestor = parent
    while not existing_ancestor.exists() and existing_ancestor != existing_ancestor.parent:
        existing_ancestor = existing_ancestor.parent

    if not existing_ancestor.exists() or not os.access(existing_ancestor, os.W_OK):
        raise PermissionError(
            f"Cannot write under dataset root parent: {parent}\n"
            "Please choose a writable local path under your user directory."
        )

    parent.mkdir(parents=True, exist_ok=True)

    if root_path.exists() and not resume:
        if auto_increment_root:
            next_root = _next_available_directory(root_path)
            print(f"[INFO] Dataset root exists. Auto-created next run directory: {next_root}")
            return next_root
        raise FileExistsError(
            f"Dataset root already exists: {root_path}\n"
            "Use a new path, set --resume=true to append, or set --dataset.auto_increment_root=true."
        )
    if resume and not root_path.exists():
        raise FileNotFoundError(
            f"--resume=true requested but dataset root does not exist: {root_path}\n"
            "Create the dataset first with --resume=false, then append with --resume=true."
        )
    return root_path


def _resolve_piper_binary(candidate: str | None) -> str | None:
    search_order: list[str] = []
    if candidate:
        search_order.append(candidate)
    search_order.extend(["piper", str(LOCAL_PIPER_BINARY)])

    for item in search_order:
        resolved = shutil.which(item)
        if resolved is not None:
            return resolved
        path = Path(item).expanduser()
        if path.exists() and path.is_file():
            return str(path)
    return None


def _discover_piper_model(lang: str) -> str | None:
    if lang in _PIPER_MODEL_DISCOVERY_CACHE:
        return _PIPER_MODEL_DISCOVERY_CACHE[lang]

    env_candidates = [
        os.environ.get("PIPER_MODEL", ""),
        os.environ.get("PIPER_MODEL_ZH", "") if lang == "zh" else os.environ.get("PIPER_MODEL_EN", ""),
    ]
    for raw in env_candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists() and path.is_file():
            _PIPER_MODEL_DISCOVERY_CACHE[lang] = str(path)
            return str(path)

    roots = [
        LOCAL_PIPER_MODELS_DIR,
        Path.home() / ".local/share/piper",
        Path.home() / ".cache/piper",
        Path("/usr/share/piper"),
        Path("/opt/piper/models"),
    ]
    patterns = ("*zh*.onnx", "*cmn*.onnx", "*chinese*.onnx") if lang == "zh" else ("*en*.onnx", "*english*.onnx")
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                matches = sorted(root.rglob(pattern))
            except Exception:
                matches = []
            if matches:
                _PIPER_MODEL_DISCOVERY_CACHE[lang] = str(matches[0])
                return str(matches[0])

    _PIPER_MODEL_DISCOVERY_CACHE[lang] = None
    return None


def _resolve_piper_model_for_lang(lang: str, configured_model: str | None) -> str | None:
    if configured_model:
        path = Path(configured_model).expanduser()
        if path.exists() and path.is_file():
            return str(path)
        _warn_voice_once(
            f"missing-configured-piper-model-{lang}",
            f"[WARN] Configured Piper model not found: {path}. Trying auto-discovery.",
        )
    return _discover_piper_model(lang)


def _voice_text(event: str, lang: str, *, episode_idx: int | None = None) -> str:
    spoken_episode_idx = int(episode_idx or 0) + 1
    if lang == "zh":
        table = {
            "start_recording": "开始录制。",
            "recording_episode": f"第 {spoken_episode_idx} 段。" if spoken_episode_idx == 1 else f"第 {spoken_episode_idx} 段，开始录制。",
            "rerecord_episode": "重录当前片段。",
            "quick_save": "快速保存当前片段。",
            "saved_episode": f"第 {spoken_episode_idx} 段，保存完成。",
            "reset_environment": "请重新布置场景。",
            "stop_recording": "录制结束。",
        }
    else:
        table = {
            "start_recording": "Start recording.",
            "recording_episode": f"Episode {spoken_episode_idx}." if spoken_episode_idx == 1 else f"Recording episode {spoken_episode_idx}.",
            "rerecord_episode": "Re-record current episode.",
            "quick_save": "Quick save current segment.",
            "saved_episode": f"Episode {spoken_episode_idx} saved.",
            "reset_environment": "Reset the environment.",
            "stop_recording": "Recording finished.",
        }
    return table.get(event, event)


def _piper_env_with_runtime_libs(bin_path: str) -> dict[str, str]:
    env = os.environ.copy()
    if platform.system() != "Linux":
        return env

    lib_dir = str(Path(bin_path).expanduser().resolve().parent)
    current = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p] if current else []
    if lib_dir not in parts:
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
    return env


def _play_wav_bytes(wav_data: bytes, *, blocking: bool) -> bool:
    players: list[list[str]] = []
    if shutil.which("aplay") is not None:
        players.append(["aplay", "-q"])
    if shutil.which("ffplay") is not None:
        players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i"])

    if not players:
        return False

    for cmd in players:
        fd, path_str = tempfile.mkstemp(prefix="songling_voice_", suffix=".wav")
        os.close(fd)
        wav_path = Path(path_str)
        try:
            wav_path.write_bytes(wav_data)
            full_cmd = [*cmd, str(wav_path)]
            if blocking:
                subprocess.run(full_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                wav_path.unlink(missing_ok=True)
                return True

            proc = subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            def _cleanup() -> None:
                try:
                    proc.wait(timeout=10.0)
                except Exception:
                    pass
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

            threading.Thread(target=_cleanup, daemon=True).start()
            return True
        except Exception:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

    return False


def _build_tone_wav_bytes(sample_rate: int = 22050) -> bytes:
    key = "default"
    if key in _TONE_WAV_CACHE:
        return _TONE_WAV_CACHE[key]

    pcm = bytearray()
    for freq_hz, duration_s in ((880, 0.08), (1047, 0.12)):
        n = max(1, int(duration_s * sample_rate))
        for i in range(n):
            value = int(32767 * 0.32 * math.sin(2.0 * math.pi * float(freq_hz) * (i / sample_rate)))
            pcm.extend(int(value).to_bytes(2, byteorder="little", signed=True))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(pcm))

    data = buf.getvalue()
    _TONE_WAV_CACHE[key] = data
    return data


def _get_or_build_piper_wav_bytes(
    text: str,
    *,
    lang: str,
    piper_binary: str | None,
    piper_model: str | None,
    piper_speaker: int | None,
) -> bytes | None:
    with _VOICE_PLAY_LOCK:
        bin_path = _resolve_piper_binary(piper_binary)
        if bin_path is None:
            _warn_voice_once("missing-piper-bin", "[WARN] Piper binary not found. Falling back to log_say.")
            return None

        model_path = _resolve_piper_model_for_lang(lang, piper_model)
        if model_path is None:
            _warn_voice_once(
                f"missing-piper-model-{lang}",
                "[WARN] Piper model not found for current language. Falling back to log_say.",
            )
            return None

        cache_key = (text, lang, model_path, piper_speaker)
        cached_wav = _PIPER_WAV_CACHE.get(cache_key)
        if cached_wav is not None:
            return cached_wav

        fd, out_path_str = tempfile.mkstemp(prefix="songling_piper_", suffix=".wav")
        os.close(fd)
        out_path = Path(out_path_str)
        try:
            cmd = [bin_path, "--model", model_path, "--output_file", str(out_path)]
            if piper_speaker is not None:
                cmd.extend(["--speaker", str(int(piper_speaker))])
            subprocess.run(
                cmd,
                input=text,
                text=True,
                check=True,
                env=_piper_env_with_runtime_libs(bin_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wav_bytes = out_path.read_bytes()
            _PIPER_WAV_CACHE[cache_key] = wav_bytes
            return wav_bytes
        except Exception as exc:
            _warn_voice_once("piper-synth-failed", f"[WARN] Piper synthesis failed ({exc}). Falling back to log_say.")
            return None
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass


def _speak_with_piper_blocking(
    text: str,
    *,
    lang: str,
    piper_binary: str | None,
    piper_model: str | None,
    piper_speaker: int | None,
) -> bool:
    wav_bytes = _get_or_build_piper_wav_bytes(
        text,
        lang=lang,
        piper_binary=piper_binary,
        piper_model=piper_model,
        piper_speaker=piper_speaker,
    )
    if wav_bytes is None:
        return False
    return _play_wav_bytes(wav_bytes, blocking=True)


def _speak_with_piper(
    text: str,
    *,
    lang: str,
    piper_binary: str | None,
    piper_model: str | None,
    piper_speaker: int | None,
    blocking: bool,
) -> bool:
    if blocking:
        return _speak_with_piper_blocking(
            text,
            lang=lang,
            piper_binary=piper_binary,
            piper_model=piper_model,
            piper_speaker=piper_speaker,
        )

    def _worker() -> None:
        _speak_with_piper_blocking(
            text,
            lang=lang,
            piper_binary=piper_binary,
            piper_model=piper_model,
            piper_speaker=piper_speaker,
        )

    try:
        threading.Thread(target=_worker, daemon=True).start()
        return True
    except Exception:
        return False


def _safe_log_say(
    text: str,
    play_sounds: bool,
    blocking: bool = False,
    *,
    lang: str = "en",
    voice_engine: str = "auto",
    piper_binary: str | None = None,
    piper_model: str | None = None,
    piper_speaker: int | None = None,
) -> None:
    print(f"[VOICE] {text}")
    if not play_sounds:
        return

    def _speak_serialized() -> None:
        with _VOICE_OUTPUT_LOCK:
            engine = _coerce_voice_engine(voice_engine)
            if engine in {"auto", "piper"}:
                if _speak_with_piper_blocking(
                    text,
                    lang=lang,
                    piper_binary=piper_binary,
                    piper_model=piper_model,
                    piper_speaker=piper_speaker,
                ):
                    return

            if engine == "off":
                return

            try:
                log_say(text, play_sounds=True, blocking=True)
                return
            except Exception:
                pass

            try:
                _play_wav_bytes(_build_tone_wav_bytes(), blocking=True)
            except Exception:
                pass

    if blocking:
        _speak_serialized()
        return

    try:
        threading.Thread(target=_speak_serialized, daemon=True).start()
    except Exception:
        _speak_serialized()


def _prewarm_voice_prompts(opts: DatasetOptions) -> None:
    if not opts.play_sounds:
        return
    if _coerce_voice_engine(opts.voice_engine) not in {"auto", "piper"}:
        return

    prompts: list[tuple[str, int | None]] = [
        ("start_recording", None),
        ("recording_episode", 0),
        ("saved_episode", 0),
        ("reset_environment", None),
        ("quick_save", None),
        ("rerecord_episode", None),
        ("stop_recording", None),
    ]

    for event, episode_idx in prompts:
        text = _voice_text(event, opts.voice_lang, episode_idx=episode_idx)
        _ = _get_or_build_piper_wav_bytes(
            text,
            lang=opts.voice_lang,
            piper_binary=opts.voice_piper_binary,
            piper_model=opts.voice_piper_model,
            piper_speaker=opts.voice_piper_speaker,
        )


def _load_songling_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = draccus.load(dict[str, Any], f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} is not a valid mapping.")
    return cfg


def _parse_index_or_path(value: Any) -> int | Path:
    if isinstance(value, int):
        return value
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    if text and (text.isdigit() or (text[0] in {"+", "-"} and text[1:].isdigit())):
        return int(text)
    return Path(text)


def _normalize_opencv_source(index_or_path: int | Path) -> int | Path:
    if not isinstance(index_or_path, Path) or platform.system() != "Linux":
        return index_or_path

    try:
        resolved = index_or_path.expanduser().resolve(strict=True)
    except Exception:
        return index_or_path

    name = resolved.name
    if name.startswith("video") and name[5:].isdigit():
        idx = int(name[5:])
        print(f"[INFO] Resolved camera path {index_or_path} -> /dev/{name} (index={idx}) for OpenCV.")
        return idx
    return index_or_path


def _camera_config_from_dict(raw: dict[str, Any], fallback_fps: int) -> OpenCVCameraConfig:
    cam_type = raw.get("type", "opencv")
    if cam_type != "opencv":
        raise ValueError(f"Songling recorder currently supports OpenCV cameras only, got {cam_type!r}.")
    if "index_or_path" not in raw:
        raise ValueError("Camera config misses `index_or_path`.")

    return OpenCVCameraConfig(
        index_or_path=_normalize_opencv_source(_parse_index_or_path(raw["index_or_path"])),
        width=int(raw.get("width", 640)),
        height=int(raw.get("height", 480)),
        fps=int(raw.get("fps", fallback_fps)),
        color_mode=raw.get("color_mode", "rgb"),
        rotation=raw.get("rotation", 0),
        warmup_s=int(raw.get("warmup_s", 1)),
        fourcc=raw.get("fourcc"),
        backend=raw.get("backend", 0),
    )


def _validate_camera_devices(camera_configs: dict[str, OpenCVCameraConfig]) -> None:
    missing: list[tuple[str, Path]] = []
    inaccessible: list[tuple[str, Path, str]] = []

    for cam_key, cam_cfg in camera_configs.items():
        if not isinstance(cam_cfg.index_or_path, Path):
            continue
        index_or_path = cam_cfg.index_or_path
        if not index_or_path.exists():
            missing.append((cam_key, index_or_path))
            continue
        try:
            fd = os.open(index_or_path, os.O_RDONLY | os.O_NONBLOCK)
            os.close(fd)
        except OSError as exc:
            inaccessible.append((cam_key, index_or_path, str(exc)))

    if not missing and not inaccessible:
        return

    available = sorted(glob.glob("/dev/v4l/by-id/*"))
    available_text = "\n".join(f"  - {path}" for path in available) if available else "  (none)"
    issues: list[str] = []
    if missing:
        issues.append("Missing camera device path(s):\n" + "\n".join(f"  - {name}: {path}" for name, path in missing))
    if inaccessible:
        issues.append(
            "Camera device path exists but cannot be opened:\n"
            + "\n".join(f"  - {name}: {path} ({err})" for name, path, err in inaccessible)
        )
    raise FileNotFoundError(
        "\n".join(issues)
        + "\nCurrent /dev/v4l/by-id entries:\n"
        + available_text
        + "\nPlease replug the missing camera or override camera paths on CLI."
    )


def _apply_common_arm_overrides(
    cfg: SonglingFollowerConfigBase,
    raw_arm: dict[str, Any],
) -> None:
    cfg.transport_backend = str(raw_arm.get("transport_backend", cfg.transport_backend))
    cfg.interface = str(raw_arm.get("interface", cfg.interface))
    cfg.piper_judge_flag = bool(raw_arm.get("piper_judge_flag", cfg.piper_judge_flag))
    cfg.disable_torque_on_disconnect = bool(
        raw_arm.get("disable_torque_on_disconnect", cfg.disable_torque_on_disconnect)
    )
    cfg.max_relative_target = raw_arm.get("max_relative_target", cfg.max_relative_target)

    if isinstance(raw_arm.get("joint_limits"), dict):
        cfg.joint_limits = dict(raw_arm["joint_limits"])
    if isinstance(raw_arm.get("joint_zero_offsets"), dict):
        cfg.joint_zero_offsets = dict(raw_arm["joint_zero_offsets"])
    if isinstance(raw_arm.get("pre_disable_pose"), dict):
        cfg.pre_disable_pose = dict(raw_arm["pre_disable_pose"])

    cfg.speed_percent = int(_first(raw_arm, "speed_percent", "move_spd_rate_ctrl", default=cfg.speed_percent))
    cfg.motion_mode = str(raw_arm.get("motion_mode", cfg.motion_mode))
    cfg.ctrl_mode = int(raw_arm.get("ctrl_mode", cfg.ctrl_mode))
    cfg.move_mode = int(raw_arm.get("move_mode", cfg.move_mode))
    cfg.mit_mode = int(raw_arm.get("mit_mode", cfg.mit_mode))
    cfg.residence_time = int(raw_arm.get("residence_time", cfg.residence_time))
    cfg.installation_pos = raw_arm.get("installation_pos", cfg.installation_pos)
    cfg.command_repeat = int(raw_arm.get("command_repeat", cfg.command_repeat))
    cfg.command_interval_s = float(raw_arm.get("command_interval_s", cfg.command_interval_s))
    cfg.mode_keepalive_s = float(raw_arm.get("mode_keepalive_s", cfg.mode_keepalive_s))
    cfg.poll_max_msgs = int(raw_arm.get("poll_max_msgs", cfg.poll_max_msgs))
    cfg.joint_scale = float(raw_arm.get("joint_scale", cfg.joint_scale))
    cfg.gripper_scale = float(raw_arm.get("gripper_scale", cfg.gripper_scale))
    cfg.gripper_force = float(raw_arm.get("gripper_force", cfg.gripper_force))
    cfg.gripper_status_code = int(raw_arm.get("gripper_status_code", cfg.gripper_status_code))
    cfg.gripper_set_zero = int(raw_arm.get("gripper_set_zero", cfg.gripper_set_zero))
    cfg.enable_retry_count = int(raw_arm.get("enable_retry_count", cfg.enable_retry_count))
    cfg.enable_retry_interval_s = float(raw_arm.get("enable_retry_interval_s", cfg.enable_retry_interval_s))
    cfg.leader_follower_role = raw_arm.get("leader_follower_role", cfg.leader_follower_role)
    cfg.leader_follower_feedback_offset = int(
        raw_arm.get("leader_follower_feedback_offset", cfg.leader_follower_feedback_offset)
    )
    cfg.leader_follower_ctrl_offset = int(
        raw_arm.get("leader_follower_ctrl_offset", cfg.leader_follower_ctrl_offset)
    )
    cfg.leader_follower_linkage_offset = int(
        raw_arm.get("leader_follower_linkage_offset", cfg.leader_follower_linkage_offset)
    )

    cfg.allow_unverified_commanding = False
    cfg.auto_enable_on_connect = False
    cfg.auto_configure_mode_on_connect = False
    cfg.auto_configure_master_slave_on_connect = False


def _build_runtime_robot_config(raw_cfg: dict[str, Any], args: argparse.Namespace, fallback_fps: int) -> BiSonglingFollowerConfig:
    raw_robot = _expect_dict(raw_cfg.get("robot"), "robot")
    robot_type = raw_robot.get("type")
    if robot_type not in {None, "bi_songling_follower", "bi_openarm_follower"}:
        raise ValueError(
            f"Unsupported robot.type={robot_type!r}. "
            "This recorder expects a Songling-style bimanual integrated chain config."
        )

    left_raw = _expect_dict(raw_robot.get("left_arm_config"), "robot.left_arm_config")
    right_raw = _expect_dict(raw_robot.get("right_arm_config"), "robot.right_arm_config")

    left_cameras_raw = _expect_dict(left_raw.get("cameras"), "robot.left_arm_config.cameras")
    right_cameras_raw = _expect_dict(right_raw.get("cameras"), "robot.right_arm_config.cameras")
    left_high_raw = dict(_expect_dict(left_cameras_raw.get("high"), "robot.left_arm_config.cameras.high"))
    left_elbow_raw = dict(_expect_dict(left_cameras_raw.get("elbow"), "robot.left_arm_config.cameras.elbow"))
    right_elbow_raw = dict(_expect_dict(right_cameras_raw.get("elbow"), "robot.right_arm_config.cameras.elbow"))

    if args.left_high is not None:
        left_high_raw["index_or_path"] = args.left_high
    if args.left_elbow is not None:
        left_elbow_raw["index_or_path"] = args.left_elbow
    if args.right_elbow is not None:
        right_elbow_raw["index_or_path"] = args.right_elbow

    left_cfg = SonglingFollowerConfigBase(
        channel=str(args.left_interface if args.left_interface is not None else _first(left_raw, "channel", "port")),
        side=str(left_raw.get("side", "left")),
        transport_backend=str(left_raw.get("transport_backend", "piper_sdk")),
        interface=str(left_raw.get("interface", "socketcan")),
        use_can_fd=_coerce_bool(args.left_use_fd if args.left_use_fd is not None else left_raw.get("use_can_fd"), False),
        bitrate=_coerce_int(
            args.left_bitrate if args.left_bitrate is not None else _first(left_raw, "bitrate", "can_bitrate"),
            1000000,
        ),
        can_data_bitrate=_coerce_int(
            args.left_data_bitrate if args.left_data_bitrate is not None else left_raw.get("can_data_bitrate"),
            5000000,
        ),
        cameras={
            "high": _camera_config_from_dict(left_high_raw, fallback_fps),
            "elbow": _camera_config_from_dict(left_elbow_raw, fallback_fps),
        },
    )
    right_cfg = SonglingFollowerConfigBase(
        channel=str(
            args.right_interface if args.right_interface is not None else _first(right_raw, "channel", "port")
        ),
        side=str(right_raw.get("side", "right")),
        transport_backend=str(right_raw.get("transport_backend", "piper_sdk")),
        interface=str(right_raw.get("interface", "socketcan")),
        use_can_fd=_coerce_bool(
            args.right_use_fd if args.right_use_fd is not None else right_raw.get("use_can_fd"),
            False,
        ),
        bitrate=_coerce_int(
            args.right_bitrate if args.right_bitrate is not None else _first(right_raw, "bitrate", "can_bitrate"),
            1000000,
        ),
        can_data_bitrate=_coerce_int(
            args.right_data_bitrate if args.right_data_bitrate is not None else right_raw.get("can_data_bitrate"),
            5000000,
        ),
        cameras={
            "elbow": _camera_config_from_dict(right_elbow_raw, fallback_fps),
        },
    )

    _apply_common_arm_overrides(left_cfg, left_raw)
    _apply_common_arm_overrides(right_cfg, right_raw)

    calibration_dir = raw_robot.get("calibration_dir")
    return BiSonglingFollowerConfig(
        id=str(raw_robot.get("id", "songling_aloha_runtime")),
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        left_arm_config=left_cfg,
        right_arm_config=right_cfg,
    )


def _resolve_dataset_options(raw_cfg: dict[str, Any], args: argparse.Namespace) -> DatasetOptions:
    raw_dataset = raw_cfg.get("dataset") if isinstance(raw_cfg.get("dataset"), dict) else {}
    top_level_fps = raw_cfg.get("fps")

    repo_id = args.dataset_repo_id if args.dataset_repo_id is not None else raw_dataset.get("repo_id")
    if not repo_id:
        raise ValueError("Missing dataset repo id. Set --dataset.repo_id=... or add dataset.repo_id in YAML.")

    single_task = args.dataset_single_task if args.dataset_single_task is not None else raw_dataset.get("single_task")
    if not single_task:
        raise ValueError(
            "Missing dataset task description. Set --dataset.single_task=... or add dataset.single_task in YAML."
        )

    resume = args.resume if args.resume is not None else _coerce_bool(raw_cfg.get("resume"), False)
    auto_increment_root = _coerce_bool(
        args.dataset_auto_increment_root
        if args.dataset_auto_increment_root is not None
        else raw_dataset.get("auto_increment_root"),
        True,
    )

    root_value = args.dataset_root if args.dataset_root is not None else raw_dataset.get("root")
    if not root_value:
        root_value = str(DEFAULT_DATASET_ROOT)
        print(f"[INFO] dataset.root not set; defaulting to: {root_value}")
    root = _ensure_local_root_is_writable(root_value=root_value, resume=resume, auto_increment_root=auto_increment_root)

    fps_default = raw_dataset.get("fps", top_level_fps if top_level_fps is not None else 30)
    fps = _coerce_int(args.dataset_fps if args.dataset_fps is not None else fps_default, 30)
    episode_time_s = _coerce_float(
        args.dataset_episode_time_s if args.dataset_episode_time_s is not None else raw_dataset.get("episode_time_s"),
        30.0,
    )
    reset_time_s = _coerce_float(
        args.dataset_reset_time_s if args.dataset_reset_time_s is not None else raw_dataset.get("reset_time_s"),
        15.0,
    )
    num_episodes = _coerce_int(
        args.dataset_num_episodes if args.dataset_num_episodes is not None else raw_dataset.get("num_episodes"),
        1,
    )

    return DatasetOptions(
        repo_id=str(repo_id),
        single_task=str(single_task),
        root=root,
        fps=fps,
        episode_time_s=episode_time_s,
        reset_time_s=reset_time_s,
        num_episodes=num_episodes,
        video=_coerce_bool(args.dataset_video if args.dataset_video is not None else raw_dataset.get("video"), True),
        push_to_hub=_coerce_bool(
            args.dataset_push_to_hub if args.dataset_push_to_hub is not None else raw_dataset.get("push_to_hub"),
            False,
        ),
        private=_coerce_bool(
            args.dataset_private if args.dataset_private is not None else raw_dataset.get("private"),
            False,
        ),
        tags=_parse_tags(args.dataset_tags if args.dataset_tags is not None else raw_dataset.get("tags")),
        num_image_writer_processes=_coerce_int(
            args.dataset_num_image_writer_processes
            if args.dataset_num_image_writer_processes is not None
            else raw_dataset.get("num_image_writer_processes"),
            0,
        ),
        num_image_writer_threads_per_camera=_coerce_int(
            args.dataset_num_image_writer_threads_per_camera
            if args.dataset_num_image_writer_threads_per_camera is not None
            else raw_dataset.get("num_image_writer_threads_per_camera"),
            4,
        ),
        video_encoding_batch_size=_coerce_int(
            args.dataset_video_encoding_batch_size
            if args.dataset_video_encoding_batch_size is not None
            else raw_dataset.get("video_encoding_batch_size"),
            1,
        ),
        vcodec=str(args.dataset_vcodec if args.dataset_vcodec is not None else raw_dataset.get("vcodec", "libsvtav1")),
        streaming_encoding=_coerce_bool(
            args.dataset_streaming_encoding
            if args.dataset_streaming_encoding is not None
            else raw_dataset.get("streaming_encoding"),
            False,
        ),
        encoder_queue_maxsize=_coerce_int(
            args.dataset_encoder_queue_maxsize
            if args.dataset_encoder_queue_maxsize is not None
            else raw_dataset.get("encoder_queue_maxsize"),
            30,
        ),
        encoder_threads=(
            args.dataset_encoder_threads
            if args.dataset_encoder_threads is not None
            else (
                int(raw_dataset.get("encoder_threads"))
                if raw_dataset.get("encoder_threads") is not None
                else None
            )
        ),
        resume=resume,
        auto_increment_root=auto_increment_root,
        play_sounds=_coerce_bool(args.play_sounds if args.play_sounds is not None else raw_cfg.get("play_sounds"), False),
        voice_lang=str(args.voice_lang if args.voice_lang is not None else raw_cfg.get("voice_lang", "zh")).strip().lower(),
        voice_rate=_coerce_int(args.voice_rate if args.voice_rate is not None else raw_cfg.get("voice_rate"), 0),
        voice_engine=_coerce_voice_engine(
            args.voice_engine if args.voice_engine is not None else raw_cfg.get("voice_engine", "auto")
        ),
        voice_piper_model=(
            str(args.voice_piper_model if args.voice_piper_model is not None else raw_cfg.get("voice_piper_model"))
            if (args.voice_piper_model is not None or raw_cfg.get("voice_piper_model") is not None)
            else None
        ),
        voice_piper_binary=(
            str(args.voice_piper_binary if args.voice_piper_binary is not None else raw_cfg.get("voice_piper_binary"))
            if (args.voice_piper_binary is not None or raw_cfg.get("voice_piper_binary") is not None)
            else None
        ),
        voice_piper_speaker=(
            int(args.voice_piper_speaker if args.voice_piper_speaker is not None else raw_cfg.get("voice_piper_speaker"))
            if (args.voice_piper_speaker is not None or raw_cfg.get("voice_piper_speaker") is not None)
            else None
        ),
    )


def _resolve_display_options(raw_cfg: dict[str, Any], args: argparse.Namespace) -> DisplayOptions:
    display_data = _coerce_bool(
        args.display_data if args.display_data is not None else raw_cfg.get("display_data"),
        False,
    )
    display_ip = args.display_ip if args.display_ip is not None else raw_cfg.get("display_ip")
    display_port_raw = args.display_port if args.display_port is not None else raw_cfg.get("display_port")
    display_port = int(display_port_raw) if display_port_raw is not None else None
    if (display_ip is None) != (display_port is None):
        raise ValueError(
            "Please set both display_ip and display_port together, or omit both for local viewer."
        )
    display_compressed_images = _coerce_bool(
        args.display_compressed_images
        if args.display_compressed_images is not None
        else raw_cfg.get("display_compressed_images"),
        False,
    )
    return DisplayOptions(
        display_data=display_data,
        display_ip=display_ip,
        display_port=display_port,
        display_compressed_images=display_compressed_images,
    )


def _build_features(
    camera_configs: dict[str, OpenCVCameraConfig],
    joint_names: list[str],
    use_video: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    vector_names = [f"{side}_{joint_name}.pos" for side in ("left", "right") for joint_name in joint_names]
    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (len(vector_names),),
            "names": vector_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(vector_names),),
            "names": vector_names,
        },
    }
    for cam_name, cam_cfg in camera_configs.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video" if use_video else "image",
            "shape": (int(cam_cfg.height), int(cam_cfg.width), 3),
            "names": ["height", "width", "channels"],
        }
    return features, vector_names


def _create_or_resume_dataset(
    opts: DatasetOptions,
    features: dict[str, dict[str, Any]],
    num_cameras: int,
    robot_type: str,
) -> LeRobotDataset:
    if opts.resume:
        dataset = LeRobotDataset(
            opts.repo_id,
            root=opts.root,
            batch_encoding_size=opts.video_encoding_batch_size,
            vcodec=opts.vcodec,
            streaming_encoding=opts.streaming_encoding,
            encoder_queue_maxsize=opts.encoder_queue_maxsize,
            encoder_threads=opts.encoder_threads,
        )
        if num_cameras > 0:
            dataset.start_image_writer(
                num_processes=opts.num_image_writer_processes,
                num_threads=opts.num_image_writer_threads_per_camera * num_cameras,
            )
        return dataset

    return LeRobotDataset.create(
        repo_id=opts.repo_id,
        fps=opts.fps,
        features=features,
        root=opts.root,
        robot_type=robot_type,
        use_videos=opts.video,
        image_writer_processes=opts.num_image_writer_processes,
        image_writer_threads=opts.num_image_writer_threads_per_camera * num_cameras,
        batch_encoding_size=opts.video_encoding_batch_size,
        vcodec=opts.vcodec,
        streaming_encoding=opts.streaming_encoding,
        encoder_queue_maxsize=opts.encoder_queue_maxsize,
        encoder_threads=opts.encoder_threads,
    )


def _prefixed_camera_configs(robot_cfg: BiSonglingFollowerConfig) -> dict[str, OpenCVCameraConfig]:
    cameras: dict[str, OpenCVCameraConfig] = {}
    for name, cfg in robot_cfg.left_arm_config.cameras.items():
        cameras[f"left_{name}"] = cfg
    for name, cfg in robot_cfg.right_arm_config.cameras.items():
        cameras[f"right_{name}"] = cfg
    return cameras


def _prefixed_camera_handles(robot: BiSonglingFollower) -> dict[str, Any]:
    handles: dict[str, Any] = {}
    for name, cam in robot.left_arm.cameras.items():
        handles[f"left_{name}"] = cam
    for name, cam in robot.right_arm.cameras.items():
        handles[f"right_{name}"] = cam
    return handles


def _safe_disconnect_camera(camera: Any) -> None:
    disconnect = getattr(camera.disconnect, "__wrapped__", None)
    try:
        if callable(disconnect):
            disconnect(camera)
        else:
            camera.disconnect()
    except Exception:
        pass


def _defer_camera_connect(robot: BiSonglingFollower) -> tuple[dict[str, Any], dict[str, Any]]:
    left_cameras = dict(robot.left_arm.cameras)
    right_cameras = dict(robot.right_arm.cameras)
    robot.left_arm.cameras = {}
    robot.right_arm.cameras = {}
    robot.cameras = {}
    return left_cameras, right_cameras


def _attach_camera_maps(
    robot: BiSonglingFollower,
    *,
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
) -> None:
    robot.left_arm.cameras = dict(left_cameras)
    robot.right_arm.cameras = dict(right_cameras)
    robot.cameras = {**robot.left_arm.cameras, **robot.right_arm.cameras}


def _connect_camera_with_retry(
    camera: Any,
    *,
    camera_name: str,
    attempts: int = CAMERA_CONNECT_RETRIES,
    warmup_s: float = CAMERA_CONNECT_RETRY_WARMUP_S,
    settle_s: float = CAMERA_CONNECT_RETRY_SETTLE_S,
) -> None:
    last_exc: Exception | None = None
    original_warmup = float(getattr(camera, "warmup_s", 0.0) or 0.0)
    try:
        for attempt in range(1, max(int(attempts), 1) + 1):
            _safe_disconnect_camera(camera)
            try:
                setattr(camera, "warmup_s", max(original_warmup, float(warmup_s)))
            except Exception:
                pass
            try:
                camera.connect(warmup=True)
                print(
                    f"[INFO] Camera '{camera_name}' connected on attempt {attempt}/{max(int(attempts), 1)}."
                )
                return
            except Exception as exc:
                last_exc = exc
                print(
                    f"[WARN] Camera '{camera_name}' connect attempt {attempt}/{max(int(attempts), 1)} failed: {exc}"
                )
                _safe_disconnect_camera(camera)
                if attempt < max(int(attempts), 1) and settle_s > 0:
                    time.sleep(settle_s)
    finally:
        try:
            setattr(camera, "warmup_s", original_warmup)
        except Exception:
            pass

    raise RuntimeError(
        f"Camera '{camera_name}' failed to connect after {max(int(attempts), 1)} attempts."
        + (f" Last error: {last_exc}" if last_exc is not None else "")
    )


def _connect_recording_robot(robot: BiSonglingFollower) -> None:
    left_cameras, right_cameras = _defer_camera_connect(robot)
    try:
        robot.connect(calibrate=False)
        _attach_camera_maps(robot, left_cameras=left_cameras, right_cameras=right_cameras)
        connected_left: dict[str, Any] = {}
        connected_right: dict[str, Any] = {}

        for side, camera_items, bucket in (
            ("left", left_cameras.items(), connected_left),
            ("right", right_cameras.items(), connected_right),
        ):
            for camera_key, camera in camera_items:
                camera_name = f"{side}_{camera_key}"
                _connect_camera_with_retry(camera, camera_name=camera_name)
                bucket[camera_key] = camera

        _attach_camera_maps(robot, left_cameras=connected_left, right_cameras=connected_right)
    except Exception:
        _attach_camera_maps(robot, left_cameras={}, right_cameras={})
        try:
            robot.disconnect()
        except Exception:
            pass
        raise


def _frame_signature(frame: np.ndarray) -> int:
    sample = frame[::16, ::16]
    return int(abs(hash(sample.tobytes())) % (2**31))


def _read_camera_frames_with_health(
    cameras: dict[str, Any],
    health_state: CameraHealthState,
    camera_max_age_ms: int,
    camera_retry_timeout_ms: int,
    camera_reconnect_stale_count: int,
    camera_freeze_identical_count: int,
    camera_reconnect_cooldown_s: float,
    camera_fail_on_freeze: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    frames: dict[str, np.ndarray] = {}
    health: dict[str, dict[str, float]] = {}

    for cam_key, cam in cameras.items():
        frame = None
        used_cached = False
        reconnected = False
        reconnect_reason = ""

        try:
            frame = cam.read_latest(max_age_ms=camera_max_age_ms)
        except Exception:
            try:
                frame = cam.async_read(timeout_ms=max(camera_retry_timeout_ms, 1))
            except Exception:
                frame = health_state.last_frames.get(cam_key)
                used_cached = frame is not None

        if frame is None:
            raise RuntimeError(f"Camera '{cam_key}' returned no frame. Check cable/power/device path, then retry.")

        if used_cached:
            health_state.stale_counts[cam_key] = health_state.stale_counts.get(cam_key, 0) + 1
        else:
            health_state.stale_counts[cam_key] = 0

        sig = _frame_signature(frame)
        if health_state.last_signatures.get(cam_key) == sig:
            health_state.identical_counts[cam_key] = health_state.identical_counts.get(cam_key, 0) + 1
        else:
            health_state.identical_counts[cam_key] = 0
        health_state.last_signatures[cam_key] = sig

        stale_trigger = camera_reconnect_stale_count > 0 and health_state.stale_counts[cam_key] >= camera_reconnect_stale_count
        freeze_trigger = camera_freeze_identical_count > 0 and health_state.identical_counts[cam_key] >= camera_freeze_identical_count
        should_reconnect = stale_trigger or freeze_trigger

        if should_reconnect:
            now = time.time()
            cooldown_ok = (now - health_state.last_reconnect_ts.get(cam_key, 0.0)) >= max(camera_reconnect_cooldown_s, 0.0)
            if cooldown_ok:
                reconnect_reason = (
                    f"stale_count={health_state.stale_counts[cam_key]}"
                    if stale_trigger
                    else f"identical_count={health_state.identical_counts[cam_key]}"
                )
                print(f"[WARN] Camera '{cam_key}' seems stalled ({reconnect_reason}), reconnecting...")
                try:
                    cam.disconnect()
                except Exception:
                    pass
                try:
                    cam.connect(warmup=True)
                    frame = cam.async_read(timeout_ms=max(camera_retry_timeout_ms, 1))
                    reconnected = True
                    health_state.stale_counts[cam_key] = 0
                    health_state.identical_counts[cam_key] = 0
                    health_state.last_signatures[cam_key] = _frame_signature(frame)
                    print(f"[INFO] Camera '{cam_key}' reconnected.")
                except Exception as exc:
                    print(f"[WARN] Camera '{cam_key}' reconnect failed: {exc}")
                finally:
                    health_state.last_reconnect_ts[cam_key] = time.time()

            if camera_fail_on_freeze and not reconnected:
                raise RuntimeError(
                    f"Camera '{cam_key}' stalled ({reconnect_reason or 'stale/freeze trigger'}) and reconnect did not recover."
                )

        health_state.last_frames[cam_key] = frame
        frames[cam_key] = frame
        health[cam_key] = {
            "used_cached": float(1.0 if used_cached else 0.0),
            "stale_count": float(health_state.stale_counts.get(cam_key, 0)),
            "identical_count": float(health_state.identical_counts.get(cam_key, 0)),
            "reconnected": float(1.0 if reconnected else 0.0),
        }

    return frames, health


def _side_command_echo_status(port: Any, *, max_age_s: float) -> CommandEchoStatus:
    now = time.time()
    command_ts = float(getattr(port, "command_feedback_timestamp", 0.0) or 0.0)
    age_s = (now - command_ts) if command_ts > 0.0 else None
    is_fresh = command_ts > 0.0 and age_s is not None and age_s <= max(max_age_s, 0.0)
    is_valid = bool(getattr(port, "command_feedback_valid", False))
    commanded_seen = getattr(port, "commanded_position_seen", {})
    if not isinstance(commanded_seen, dict):
        commanded_seen = {}
    missing_joints = [joint_name for joint_name in DEFAULT_JOINT_NAMES if not bool(commanded_seen.get(joint_name, False))]
    has_control = is_valid and is_fresh and not missing_joints
    return CommandEchoStatus(
        has_control=has_control,
        is_valid=is_valid,
        is_fresh=is_fresh,
        age_s=age_s,
        missing_joints=missing_joints,
    )


def _side_has_command_echo(port: Any, *, max_age_s: float) -> bool:
    return _side_command_echo_status(port, max_age_s=max_age_s).has_control


def _side_control_reason(status: CommandEchoStatus) -> str:
    if status.has_control:
        return "ok"
    if not status.is_valid:
        return "invalid"
    if not status.is_fresh:
        return "stale"
    if status.missing_joints:
        return "missing_joint_echo"
    return "unknown"


def _side_has_control_with_stale_grace(status: CommandEchoStatus, *, stale_grace_s: float) -> bool:
    if status.has_control:
        return True
    grace_s = max(float(stale_grace_s), 0.0)
    if grace_s <= 0.0:
        return False
    if not status.is_valid or bool(status.missing_joints):
        return False
    if status.age_s is None:
        return False
    return float(status.age_s) <= grace_s


def _format_side_control_debug(side: str, status: CommandEchoStatus) -> str:
    age_text = "na" if status.age_s is None else f"{float(status.age_s):.3f}s"
    missing_text = ",".join(status.missing_joints) if status.missing_joints else "-"
    return (
        f"{side}_control: reason={_side_control_reason(status)} "
        f"valid={status.is_valid} fresh={status.is_fresh} age={age_text} "
        f"missing_joints={missing_text}"
    )


def _append_loop_range(
    *,
    summary: dict[str, Any] | None,
    key: str,
    loop_index: int,
    reason: str,
    left_status: CommandEchoStatus,
    right_status: CommandEchoStatus,
) -> None:
    if summary is None:
        return
    ranges = summary.setdefault(key, [])
    if not isinstance(ranges, list):
        ranges = []
        summary[key] = ranges

    entry = {
        "start_loop": int(loop_index),
        "end_loop": int(loop_index),
        "reason": str(reason),
        "left_has_control": bool(left_status.has_control),
        "right_has_control": bool(right_status.has_control),
        "left_reason": _side_control_reason(left_status),
        "right_reason": _side_control_reason(right_status),
    }

    if ranges and isinstance(ranges[-1], dict):
        prev = ranges[-1]
        prev_end = int(prev.get("end_loop", -2))
        if (
            prev_end + 1 == entry["start_loop"]
            and str(prev.get("reason", "")) == entry["reason"]
            and bool(prev.get("left_has_control", False)) == entry["left_has_control"]
            and bool(prev.get("right_has_control", False)) == entry["right_has_control"]
            and str(prev.get("left_reason", "")) == entry["left_reason"]
            and str(prev.get("right_reason", "")) == entry["right_reason"]
        ):
            prev["end_loop"] = entry["end_loop"]
            return

    ranges.append(entry)


def _record_missing_label_loop(
    *,
    summary: dict[str, Any] | None,
    loop_index: int,
    left_status: CommandEchoStatus,
    right_status: CommandEchoStatus,
) -> None:
    if summary is None:
        return

    summary["missing_control_loops"] = int(summary.get("missing_control_loops", 0)) + 1
    if not left_status.has_control:
        summary["missing_left_control_loops"] = int(summary.get("missing_left_control_loops", 0)) + 1
    if not right_status.has_control:
        summary["missing_right_control_loops"] = int(summary.get("missing_right_control_loops", 0)) + 1

    _append_loop_range(
        summary=summary,
        key="missing_label_loop_ranges",
        loop_index=loop_index,
        reason="missing_control",
        left_status=left_status,
        right_status=right_status,
    )


def _record_dropped_loop(
    *,
    summary: dict[str, Any] | None,
    loop_index: int,
    drop_reason: str,
    left_status: CommandEchoStatus,
    right_status: CommandEchoStatus,
) -> None:
    if summary is None:
        return

    summary["dropped_missing_control_frames"] = int(summary.get("dropped_missing_control_frames", 0)) + 1
    if not left_status.has_control:
        summary["dropped_left_missing_control_frames"] = int(summary.get("dropped_left_missing_control_frames", 0)) + 1
    if not right_status.has_control:
        summary["dropped_right_missing_control_frames"] = int(summary.get("dropped_right_missing_control_frames", 0)) + 1

    _append_loop_range(
        summary=summary,
        key="dropped_loop_ranges",
        loop_index=loop_index,
        reason=drop_reason,
        left_status=left_status,
        right_status=right_status,
    )


def _clone_camera_frames(camera_frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {cam_key: np.asarray(image).copy() for cam_key, image in camera_frames.items()}


def _drop_pending_buffer(
    *,
    pending_observations: deque[BufferedObservation],
    summary: dict[str, Any] | None,
    reason: str,
) -> int:
    dropped = 0
    while pending_observations:
        pending = pending_observations.popleft()
        dropped += 1
        _record_dropped_loop(
            summary=summary,
            loop_index=pending.loop_index,
            drop_reason=reason,
            left_status=pending.left_status,
            right_status=pending.right_status,
        )
    return dropped


def _state_vector(left_positions: dict[str, float], right_positions: dict[str, float], joint_names: list[str]) -> np.ndarray:
    return np.asarray(
        [float(left_positions[joint_name]) for joint_name in joint_names]
        + [float(right_positions[joint_name]) for joint_name in joint_names],
        dtype=np.float32,
    )


def _action_vector(
    left_commanded: dict[str, float],
    right_commanded: dict[str, float],
    joint_names: list[str],
) -> np.ndarray:
    values: list[float] = []
    for joint_name in joint_names:
        values.append(float(left_commanded[joint_name]))
    for joint_name in joint_names:
        values.append(float(right_commanded[joint_name]))
    return np.asarray(values, dtype=np.float32)


def _say_event(event: str, opts: DatasetOptions, *, episode_idx: int | None = None, blocking: bool = False) -> None:
    now = time.monotonic()
    min_gap_s = {
        "quick_save": 0.45,
        "rerecord_episode": 0.45,
        "saved_episode": 0.20,
        "reset_environment": 0.20,
        "stop_recording": 0.20,
    }.get(event, 0.0)
    with _VOICE_EVENT_TS_LOCK:
        last_ts = float(_LAST_VOICE_EVENT_TS.get(event, 0.0) or 0.0)
        if min_gap_s > 0.0 and (now - last_ts) < min_gap_s:
            return
        _LAST_VOICE_EVENT_TS[event] = now

    spoken_episode_idx = int(episode_idx or 0) + 1
    if opts.voice_lang == "zh":
        table = {
            "start_recording": "开始录制。",
            "recording_episode": f"第 {spoken_episode_idx} 段。" if spoken_episode_idx == 1 else f"第 {spoken_episode_idx} 段，开始录制。",
            "rerecord_episode": "重录当前片段。",
            "quick_save": "快速保存当前片段。",
            "saved_episode": f"第 {spoken_episode_idx} 段，保存完成。",
            "reset_environment": "请重新布置场景。",
            "stop_recording": "录制结束。",
        }
    else:
        table = {
            "start_recording": "Start recording.",
            "recording_episode": f"Episode {spoken_episode_idx}." if spoken_episode_idx == 1 else f"Recording episode {spoken_episode_idx}.",
            "rerecord_episode": "Re-record current episode.",
            "quick_save": "Quick save current segment.",
            "saved_episode": f"Episode {spoken_episode_idx} saved.",
            "reset_environment": "Reset the environment.",
            "stop_recording": "Recording finished.",
        }
    text = table.get(event, event)
    _safe_log_say(
        text,
        opts.play_sounds,
        blocking=blocking,
        lang=opts.voice_lang,
        voice_engine=opts.voice_engine,
        piper_binary=opts.voice_piper_binary,
        piper_model=opts.voice_piper_model,
        piper_speaker=opts.voice_piper_speaker,
    )


def _summary_path(root: Path) -> Path:
    return root / "meta" / CAPTURE_SUMMARY_FILENAME


def _finalize_episode_summary(summary: dict[str, Any]) -> dict[str, Any]:
    def _normalize_loop_ranges(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                start_loop = int(item.get("start_loop", 0))
                end_loop = int(item.get("end_loop", start_loop))
            except Exception:
                continue
            if end_loop < start_loop:
                start_loop, end_loop = end_loop, start_loop
            normalized.append(
                {
                    "start_loop": start_loop,
                    "end_loop": end_loop,
                    "reason": str(item.get("reason", "unknown")),
                    "left_has_control": bool(item.get("left_has_control", False)),
                    "right_has_control": bool(item.get("right_has_control", False)),
                    "left_reason": str(item.get("left_reason", "unknown")),
                    "right_reason": str(item.get("right_reason", "unknown")),
                }
            )
        return normalized

    frames = int(summary.get("frames", 0))
    total_loops = int(summary.get("total_loops", 0))
    left_control = int(summary.get("left_control_frames", 0))
    right_control = int(summary.get("right_control_frames", 0))
    missing_loops = int(summary.get("missing_control_loops", 0))
    missing_left_loops = int(summary.get("missing_left_control_loops", 0))
    missing_right_loops = int(summary.get("missing_right_control_loops", 0))
    dropped_missing = int(summary.get("dropped_missing_control_frames", 0))
    dropped_left = int(summary.get("dropped_left_missing_control_frames", 0))
    dropped_right = int(summary.get("dropped_right_missing_control_frames", 0))
    paired_from_buffer = int(summary.get("paired_from_buffer_frames", 0))
    max_buffered = int(summary.get("max_buffered_observation_frames", 0))
    dropped_overflow = int(summary.get("buffer_overflow_dropped_frames", 0))
    dropped_unpaired = int(summary.get("episode_end_unpaired_dropped_frames", 0))
    summary["frames"] = frames
    summary["total_loops"] = max(total_loops, 0)
    summary["left_fallback_frames"] = 0
    summary["right_fallback_frames"] = 0
    summary["missing_control_loops"] = max(missing_loops, 0)
    summary["missing_left_control_loops"] = max(missing_left_loops, 0)
    summary["missing_right_control_loops"] = max(missing_right_loops, 0)
    summary["left_control_ratio"] = (left_control / frames) if frames > 0 else 0.0
    summary["right_control_ratio"] = (right_control / frames) if frames > 0 else 0.0
    summary["dropped_missing_control_frames"] = max(dropped_missing, 0)
    summary["dropped_left_missing_control_frames"] = max(dropped_left, 0)
    summary["dropped_right_missing_control_frames"] = max(dropped_right, 0)
    summary["paired_from_buffer_frames"] = max(paired_from_buffer, 0)
    summary["max_buffered_observation_frames"] = max(max_buffered, 0)
    summary["buffer_overflow_dropped_frames"] = max(dropped_overflow, 0)
    summary["episode_end_unpaired_dropped_frames"] = max(dropped_unpaired, 0)
    summary["missing_label_loop_ranges"] = _normalize_loop_ranges(summary.get("missing_label_loop_ranges"))
    summary["dropped_loop_ranges"] = _normalize_loop_ranges(summary.get("dropped_loop_ranges"))
    return summary


def _write_capture_summary(
    *,
    root: Path,
    repo_id: str,
    joint_names: list[str],
    episode_summaries: list[dict[str, Any]],
    command_freshness_s: float,
    stale_grace_s: float,
    missing_label_policy: str,
    buffered_observation_max_frames: int,
    episode_start_min_fresh_frames: int,
) -> Path:
    summary_path = _summary_path(root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    training_action_source = (
        "Leader/master control echo only. Missing-label loops can buffer observation and pair when fresh labels recover."
        if missing_label_policy == "buffer-observation"
        else "Leader/master control echo only. Missing/stale label loops are dropped."
    )
    payload = {
        "schema_version": 2,
        "repo_id": repo_id,
        "dataset_root": str(root),
        "joint_names_per_side": list(joint_names),
        "capture_policy": {
            "command_freshness_s": float(command_freshness_s),
            "stale_grace_s": float(stale_grace_s),
            "missing_label_policy": str(missing_label_policy),
            "buffered_observation_max_frames": int(buffered_observation_max_frames),
            "episode_start_min_fresh_frames": int(episode_start_min_fresh_frames),
            "loop_index_base": 0,
            "loop_index_note": "Loop index is per-episode, incremented once per recorder control-loop iteration.",
        },
        "units": {
            "arm_joint_storage": "degree",
            "gripper_storage": "mm",
            "protocol_reference": "Piper-style Songling integrated CAN: follower feedback + leader control echo.",
            "training_action_source": training_action_source,
        },
        "can_frame_ids": {
            "observation": [hex(frame_id) for frame_id in OBSERVATION_FRAME_IDS],
            "action": [hex(frame_id) for frame_id in ACTION_FRAME_IDS],
        },
        "episodes": [_finalize_episode_summary(dict(item)) for item in episode_summaries],
        "seen_ids": {
            "left": [],
            "right": [],
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _print_camera_summary(camera_health: dict[str, dict[str, float]]) -> None:
    if not camera_health:
        return
    for cam_key, stats in camera_health.items():
        print(
            f"[INFO] {cam_key}: stale={int(stats['stale_count'])} "
            f"identical={int(stats['identical_count'])} "
            f"cached={int(stats['used_cached'])} reconnect={int(stats['reconnected'])}"
        )


def _consume_keyboard_requests(
    events: dict[str, Any],
    consumed: dict[str, int],
) -> tuple[bool, bool, bool]:
    quick_count = int(events.get("quick_save_request_count", 0) or 0)
    rerecord_count = int(events.get("rerecord_request_count", 0) or 0)
    stop_count = int(events.get("stop_request_count", 0) or 0)

    quick_requested = quick_count > int(consumed.get("quick_save_request_count", 0))
    rerecord_requested = rerecord_count > int(consumed.get("rerecord_request_count", 0))
    stop_requested = stop_count > int(consumed.get("stop_request_count", 0))

    if quick_requested:
        consumed["quick_save_request_count"] = quick_count
    if rerecord_requested:
        consumed["rerecord_request_count"] = rerecord_count
    if stop_requested:
        consumed["stop_request_count"] = stop_count

    if quick_requested or rerecord_requested or stop_requested:
        events["exit_early"] = False

    return quick_requested, rerecord_requested, stop_requested


def main() -> None:
    _patch_multiprocess_resource_tracker()

    parser = argparse.ArgumentParser(
        description="Record Songling integrated-chain data into LeRobot format."
    )
    parser.add_argument("--config-path", "--config_path", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and exit.")
    parser.add_argument("--joint-names", default=",".join(DEFAULT_JOINT_NAMES))
    parser.add_argument(
        "--command-freshness-s",
        type=float,
        default=DEFAULT_COMMAND_FRESHNESS_S,
        help="Maximum allowed age of leader control echo before it is treated as missing/stale.",
    )
    parser.add_argument(
        "--stale-grace-s",
        type=float,
        default=DEFAULT_STALE_GRACE_S,
        help=(
            "Optional grace window (seconds) to treat stale command echo as usable "
            "when feedback is valid and all joint echoes have been seen. "
            "Set >0 to reduce drops during short command-echo pauses."
        ),
    )
    parser.add_argument(
        "--missing-label-policy",
        choices=MISSING_LABEL_POLICY_CHOICES,
        default=DEFAULT_MISSING_LABEL_POLICY,
        help=(
            "How to handle loops with missing/stale control label. "
            "'drop' drops those loops; 'buffer-observation' caches observations and pairs them once label recovers."
        ),
    )
    parser.add_argument(
        "--buffered-observation-max-frames",
        type=int,
        default=DEFAULT_BUFFERED_OBSERVATION_MAX_FRAMES,
        help="Max pending observation loops to keep when --missing-label-policy=buffer-observation.",
    )
    parser.add_argument(
        "--episode-start-on-first-action",
        type=_parse_cli_bool,
        default=DEFAULT_START_ON_FIRST_ACTION,
        help=(
            "Only start the episode timer after the first fresh leader action label is observed. "
            "This avoids recording long pre-motion warmup and prevents pre-action observation buffering."
        ),
    )
    parser.add_argument(
        "--episode-start-min-fresh-frames",
        "--episode_start_min_fresh_frames",
        type=int,
        default=DEFAULT_EPISODE_START_MIN_FRESH_FRAMES,
        help=(
            "When waiting for episode start on first action, require this many consecutive loops "
            "with fresh leader action labels before recording starts."
        ),
    )
    parser.add_argument(
        "--episode-end-tail-s",
        type=float,
        default=DEFAULT_EPISODE_END_TAIL_S,
        help=(
            "After quick save or timeout, keep recording this many seconds of final stable frames "
            "before closing the episode. Set <=0 to disable."
        ),
    )
    parser.add_argument("--camera-max-age-ms", type=int, default=500)
    parser.add_argument("--camera-retry-timeout-ms", type=int, default=60)
    parser.add_argument("--camera-reconnect-stale-count", type=int, default=12)
    parser.add_argument("--camera-freeze-identical-count", type=int, default=120)
    parser.add_argument("--camera-reconnect-cooldown-s", type=float, default=2.0)
    parser.add_argument("--camera-fail-on-freeze", type=_parse_cli_bool, default=False)
    parser.add_argument("--display_data", "--display-data", dest="display_data", type=_parse_cli_bool, default=None)
    parser.add_argument("--display_ip", "--display-ip", dest="display_ip", default=None)
    parser.add_argument("--display_port", "--display-port", dest="display_port", type=int, default=None)
    parser.add_argument(
        "--display_compressed_images",
        "--display-compressed-images",
        dest="display_compressed_images",
        type=_parse_cli_bool,
        default=None,
    )
    parser.add_argument("--left-interface", "--robot.left_arm_config.channel", dest="left_interface", default=None)
    parser.add_argument("--right-interface", "--robot.right_arm_config.channel", dest="right_interface", default=None)
    parser.add_argument("--left-bitrate", "--robot.left_arm_config.bitrate", dest="left_bitrate", type=int, default=None)
    parser.add_argument("--right-bitrate", "--robot.right_arm_config.bitrate", dest="right_bitrate", type=int, default=None)
    parser.add_argument(
        "--left-data-bitrate",
        "--robot.left_arm_config.can_data_bitrate",
        dest="left_data_bitrate",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--right-data-bitrate",
        "--robot.right_arm_config.can_data_bitrate",
        dest="right_data_bitrate",
        type=int,
        default=None,
    )
    parser.add_argument("--left-use-fd", "--robot.left_arm_config.use_can_fd", dest="left_use_fd", type=_parse_cli_bool, default=None)
    parser.add_argument("--right-use-fd", "--robot.right_arm_config.use_can_fd", dest="right_use_fd", type=_parse_cli_bool, default=None)
    parser.add_argument("--left-high", "--robot.left_arm_config.cameras.high.index_or_path", dest="left_high", default=None)
    parser.add_argument("--left-elbow", "--robot.left_arm_config.cameras.elbow.index_or_path", dest="left_elbow", default=None)
    parser.add_argument("--right-elbow", "--robot.right_arm_config.cameras.elbow.index_or_path", dest="right_elbow", default=None)
    parser.add_argument("--dataset.repo_id", "--dataset-repo-id", dest="dataset_repo_id", default=None)
    parser.add_argument("--dataset.single_task", "--dataset-single-task", dest="dataset_single_task", default=None)
    parser.add_argument("--dataset.root", "--dataset-root", dest="dataset_root", default=None)
    parser.add_argument("--dataset.fps", "--dataset-fps", dest="dataset_fps", type=int, default=None)
    parser.add_argument("--dataset.episode_time_s", "--dataset-episode-time-s", dest="dataset_episode_time_s", type=float, default=None)
    parser.add_argument("--dataset.reset_time_s", "--dataset-reset-time-s", dest="dataset_reset_time_s", type=float, default=None)
    parser.add_argument("--dataset.num_episodes", "--dataset-num-episodes", dest="dataset_num_episodes", type=int, default=None)
    parser.add_argument("--dataset.video", "--dataset-video", dest="dataset_video", type=_parse_cli_bool, default=None)
    parser.add_argument("--dataset.push_to_hub", "--dataset-push-to-hub", dest="dataset_push_to_hub", type=_parse_cli_bool, default=None)
    parser.add_argument("--dataset.private", "--dataset-private", dest="dataset_private", type=_parse_cli_bool, default=None)
    parser.add_argument("--dataset.tags", "--dataset-tags", dest="dataset_tags", default=None)
    parser.add_argument(
        "--dataset.num_image_writer_processes",
        "--dataset-num-image-writer-processes",
        dest="dataset_num_image_writer_processes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dataset.num_image_writer_threads_per_camera",
        "--dataset-num-image-writer-threads-per-camera",
        dest="dataset_num_image_writer_threads_per_camera",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dataset.video_encoding_batch_size",
        "--dataset-video-encoding-batch-size",
        dest="dataset_video_encoding_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument("--dataset.vcodec", "--dataset-vcodec", dest="dataset_vcodec", default=None)
    parser.add_argument(
        "--dataset.streaming_encoding",
        "--dataset-streaming-encoding",
        dest="dataset_streaming_encoding",
        type=_parse_cli_bool,
        default=None,
    )
    parser.add_argument(
        "--dataset.encoder_queue_maxsize",
        "--dataset-encoder-queue-maxsize",
        dest="dataset_encoder_queue_maxsize",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dataset.encoder_threads",
        "--dataset-encoder-threads",
        dest="dataset_encoder_threads",
        type=_parse_optional_int,
        default=None,
    )
    parser.add_argument("--resume", type=_parse_cli_bool, default=None)
    parser.add_argument(
        "--dataset.auto_increment_root",
        "--dataset-auto-increment-root",
        dest="dataset_auto_increment_root",
        type=_parse_cli_bool,
        default=None,
    )
    parser.add_argument("--play_sounds", "--play-sounds", dest="play_sounds", type=_parse_cli_bool, default=None)
    parser.add_argument("--voice-lang", "--voice_lang", dest="voice_lang", choices=["en", "zh"], default=None)
    parser.add_argument("--voice-rate", "--voice_rate", dest="voice_rate", type=int, default=None)
    parser.add_argument("--voice-engine", "--voice_engine", dest="voice_engine", default=None)
    parser.add_argument("--voice-piper-model", "--voice_piper_model", dest="voice_piper_model", default=None)
    parser.add_argument("--voice-piper-binary", "--voice_piper_binary", dest="voice_piper_binary", default=None)
    parser.add_argument("--voice-piper-speaker", "--voice_piper_speaker", dest="voice_piper_speaker", type=int, default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        print("[WARN] Ignoring unsupported overrides: " + " ".join(unknown))
    if args.episode_start_min_fresh_frames < 1:
        raise ValueError("--episode-start-min-fresh-frames must be >= 1.")

    raw_cfg = _load_songling_yaml(args.config_path)
    dataset_opts = _resolve_dataset_options(raw_cfg, args)
    display_opts = _resolve_display_options(raw_cfg, args)
    joint_names = _parse_csv(args.joint_names)
    if not joint_names:
        raise ValueError("No joint names resolved. Please set --joint-names.")

    register_third_party_plugins()
    robot_cfg = _build_runtime_robot_config(raw_cfg, args, fallback_fps=dataset_opts.fps)
    _ensure_required_can_interfaces_up(
        [str(robot_cfg.left_arm_config.channel), str(robot_cfg.right_arm_config.channel)]
    )
    camera_configs = _prefixed_camera_configs(robot_cfg)
    features, vector_names = _build_features(camera_configs=camera_configs, joint_names=joint_names, use_video=dataset_opts.video)

    print("=" * 50)
    print("Songling ALOHA Dataset Record")
    print("=" * 50)
    print(f"Config: {args.config_path}")
    print(f"Dataset repo_id: {dataset_opts.repo_id}")
    print(f"Dataset root: {dataset_opts.root}")
    print(f"Robot type: bi_songling_follower")
    print(f"FPS: {dataset_opts.fps}")
    print(f"Episodes: {dataset_opts.num_episodes} (episode_time_s={dataset_opts.episode_time_s})")
    print(f"Reset time: {dataset_opts.reset_time_s}s")
    print(
        f"Voice prompts: {'on' if dataset_opts.play_sounds else 'off'} "
        f"(lang={dataset_opts.voice_lang}, engine={dataset_opts.voice_engine})"
    )
    if dataset_opts.voice_engine in {"auto", "piper"}:
        print(
            "Piper config: "
            f"model={dataset_opts.voice_piper_model or 'auto-discovery'}, "
            f"binary={dataset_opts.voice_piper_binary or 'piper'}, "
            f"speaker={dataset_opts.voice_piper_speaker if dataset_opts.voice_piper_speaker is not None else 'default'}"
        )
    if args.missing_label_policy == "buffer-observation":
        print(
            "Action source policy: leader/master control echo only. "
            "When label is missing/stale, observation is buffered and paired after label recovers."
        )
        print(f"Buffered observation max frames: {max(int(args.buffered_observation_max_frames), 0)}")
    else:
        print(
            "Action source policy: leader/master control echo only. "
            "If either side is missing or stale, that loop is dropped and never written."
        )
    if args.episode_start_on_first_action:
        if int(args.episode_start_min_fresh_frames) == 1:
            start_timing_msg = "start timer on first fresh leader action"
        else:
            start_timing_msg = (
                "start timer after "
                f"{int(args.episode_start_min_fresh_frames)} consecutive loops with fresh leader action labels"
            )
    else:
        start_timing_msg = "start timer immediately when episode loop begins"
    print("Episode timing: " + start_timing_msg + f", end tail={max(float(args.episode_end_tail_s), 0.0):.2f}s")
    print(
        "Control label timing: "
        f"freshness={float(max(args.command_freshness_s, 0.0)):.3f}s, "
        f"stale_grace={float(max(args.stale_grace_s, 0.0)):.3f}s"
    )
    print(f"Left pair CAN: {robot_cfg.left_arm_config.channel}")
    print(f"Right pair CAN: {robot_cfg.right_arm_config.channel}")
    print("Camera keys: " + ", ".join(sorted(camera_configs)))
    print(
        "Display: "
        + (
            f"on ({display_opts.display_ip}:{display_opts.display_port})"
            if (display_opts.display_data and display_opts.display_ip and display_opts.display_port)
            else ("on (local viewer)" if display_opts.display_data else "off")
        )
    )
    print()

    if args.dry_run:
        return

    _validate_camera_devices(camera_configs)
    robot = BiSonglingFollower(robot_cfg)
    if display_opts.display_data:
        try:
            import rerun as rr  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "rerun-sdk is required when --display_data=true. Install with `pip install rerun-sdk`."
            ) from exc
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(
            session_name="songling_aloha_record",
            ip=display_opts.display_ip,
            port=display_opts.display_port,
        )
    else:
        rr = None

    dataset: LeRobotDataset | None = None
    listener = None
    camera_health_state = CameraHealthState()
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "quick_save_request_count": 0,
        "rerecord_request_count": 0,
        "stop_request_count": 0,
    }
    episode_summaries: list[dict[str, Any]] = []
    current_episode_summary: dict[str, Any] | None = None
    warned_left_missing_control = False
    warned_right_missing_control = False
    saved_episodes = 0
    frames_total = 0
    display_frames_total = 0
    interrupted = False
    pending_observations: deque[BufferedObservation] = deque()

    try:
        _connect_recording_robot(robot)
        cameras = _prefixed_camera_handles(robot)
        dataset = _create_or_resume_dataset(
            opts=dataset_opts,
            features=features,
            num_cameras=len(cameras),
            robot_type=robot.robot_type,
        )
        listener, events = init_keyboard_listener()
        keyboard_request_state = {
            "quick_save_request_count": 0,
            "rerecord_request_count": 0,
            "stop_request_count": 0,
        }
        if listener is not None:
            print(
                "[INFO] Keyboard controls: Right Arrow / D=quick save, Left Arrow / A=re-record, Esc / Q=stop recording."
            )

        _prewarm_voice_prompts(dataset_opts)
        _say_event("start_recording", dataset_opts, blocking=True)
        with VideoEncodingManager(dataset):
            while saved_episodes < dataset_opts.num_episodes and not events["stop_recording"]:
                episode_idx = dataset.num_episodes
                current_episode_summary = {
                    "episode_index": int(episode_idx),
                    "frames": 0,
                    "total_loops": 0,
                    "left_control_frames": 0,
                    "right_control_frames": 0,
                    "missing_control_loops": 0,
                    "missing_left_control_loops": 0,
                    "missing_right_control_loops": 0,
                    "dropped_missing_control_frames": 0,
                    "dropped_left_missing_control_frames": 0,
                    "dropped_right_missing_control_frames": 0,
                    "paired_from_buffer_frames": 0,
                    "max_buffered_observation_frames": 0,
                    "buffer_overflow_dropped_frames": 0,
                    "episode_end_unpaired_dropped_frames": 0,
                    "missing_label_loop_ranges": [],
                    "dropped_loop_ranges": [],
                }
                pending_observations = deque()
                _say_event("recording_episode", dataset_opts, episode_idx=episode_idx, blocking=True)
                print(f"\n[INFO] Recording episode {episode_idx} ...")
                capture_start_t: float | None = None if args.episode_start_on_first_action else time.perf_counter()
                tail_end_t: float | None = None
                required_start_fresh_frames = max(int(args.episode_start_min_fresh_frames), 1)
                start_fresh_streak = 0
                ep_frames = 0
                quick_save_requested = False
                last_log_s = time.time()
                last_wait_log_s = 0.0

                while True:
                    quick_key_requested, rerecord_key_requested, stop_key_requested = _consume_keyboard_requests(
                        events,
                        keyboard_request_state,
                    )
                    if rerecord_key_requested:
                        events["rerecord_episode"] = True
                    if stop_key_requested:
                        events["stop_recording"] = True

                    if quick_key_requested or rerecord_key_requested or stop_key_requested or events["exit_early"]:
                        quick_save_requested = (
                            (quick_key_requested or (events["exit_early"] and not events["rerecord_episode"]))
                            and not events["rerecord_episode"]
                            and not events["stop_recording"]
                        )
                        events["exit_early"] = False
                        if quick_save_requested and capture_start_t is None:
                            print(
                                f"[WARN] ep={episode_idx} quick save requested before the first fresh action label; "
                                "ignoring save request and waiting for task start."
                            )
                            quick_save_requested = False
                        elif quick_save_requested:
                            tail_s = max(float(args.episode_end_tail_s), 0.0)
                            if tail_s <= 0.0:
                                break
                            if tail_end_t is None:
                                tail_end_t = time.perf_counter() + tail_s
                                print(
                                    f"[INFO] ep={episode_idx} quick save requested; keep the finished pose stable "
                                    f"for {tail_s:.2f}s before save."
                                )
                        if events["rerecord_episode"] or events["stop_recording"]:
                            break

                    now_perf = time.perf_counter()
                    if capture_start_t is not None and tail_end_t is None:
                        elapsed_s = now_perf - capture_start_t
                        if elapsed_s >= dataset_opts.episode_time_s:
                            tail_s = max(float(args.episode_end_tail_s), 0.0)
                            if tail_s <= 0.0:
                                break
                            tail_end_t = now_perf + tail_s
                            print(
                                f"[INFO] ep={episode_idx} main capture window reached; keep the finished pose stable "
                                f"for {tail_s:.2f}s before save."
                            )
                    if tail_end_t is not None and now_perf >= tail_end_t:
                        break

                    loop_start = time.perf_counter()

                    left_positions = robot.left_arm.bus.get_positions(poll=True)
                    right_positions = robot.right_arm.bus.get_positions(poll=True)
                    left_commanded = robot.left_arm.bus.get_commanded_positions(poll=False)
                    right_commanded = robot.right_arm.bus.get_commanded_positions(poll=False)
                    left_status = _side_command_echo_status(
                        robot.left_arm.bus,
                        max_age_s=args.command_freshness_s,
                    )
                    right_status = _side_command_echo_status(
                        robot.right_arm.bus,
                        max_age_s=args.command_freshness_s,
                    )
                    left_has_control_raw = left_status.has_control
                    right_has_control_raw = right_status.has_control
                    left_has_control = _side_has_control_with_stale_grace(
                        left_status, stale_grace_s=args.stale_grace_s
                    )
                    right_has_control = _side_has_control_with_stale_grace(
                        right_status, stale_grace_s=args.stale_grace_s
                    )

                    camera_frames, camera_health = _read_camera_frames_with_health(
                        cameras=cameras,
                        health_state=camera_health_state,
                        camera_max_age_ms=args.camera_max_age_ms,
                        camera_retry_timeout_ms=args.camera_retry_timeout_ms,
                        camera_reconnect_stale_count=args.camera_reconnect_stale_count,
                        camera_freeze_identical_count=args.camera_freeze_identical_count,
                        camera_reconnect_cooldown_s=args.camera_reconnect_cooldown_s,
                        camera_fail_on_freeze=args.camera_fail_on_freeze,
                    )

                    observation_state = _state_vector(left_positions, right_positions, joint_names)
                    missing_label = not left_has_control or not right_has_control

                    if not left_has_control and not warned_left_missing_control:
                        missing_hint = (
                            "Frames without a fresh leader action label will be buffered until labels recover."
                            if args.missing_label_policy == "buffer-observation"
                            else "Frames without a fresh leader action label will be dropped."
                        )
                        print(
                            "[WARN] Left side leader control echo not observed yet. "
                            + missing_hint
                        )
                        warned_left_missing_control = True
                    if not right_has_control and not warned_right_missing_control:
                        missing_hint = (
                            "Frames without a fresh leader action label will be buffered until labels recover."
                            if args.missing_label_policy == "buffer-observation"
                            else "Frames without a fresh leader action label will be dropped."
                        )
                        print(
                            "[WARN] Right side leader control echo not observed yet. "
                            + missing_hint
                        )
                        warned_right_missing_control = True

                    waiting_for_first_action = bool(args.episode_start_on_first_action and capture_start_t is None)
                    if waiting_for_first_action:
                        if missing_label:
                            start_fresh_streak = 0
                        else:
                            start_fresh_streak += 1
                            if start_fresh_streak >= required_start_fresh_frames:
                                capture_start_t = time.perf_counter()
                                last_log_s = time.time()
                                if required_start_fresh_frames == 1:
                                    print(
                                        f"[INFO] ep={episode_idx} first fresh leader action detected; "
                                        "start episode timer and frame capture."
                                    )
                                else:
                                    print(
                                        f"[INFO] ep={episode_idx} leader action labels stable for "
                                        f"{start_fresh_streak} consecutive loops; "
                                        "start episode timer and frame capture."
                                    )

                        if capture_start_t is None:
                            now_s = time.time()
                            if now_s - last_wait_log_s >= 1.0:
                                if missing_label:
                                    print(
                                        f"[INFO] ep={episode_idx} waiting for the first fresh leader action label "
                                        f"(left={left_has_control}, right={right_has_control})."
                                    )
                                    print(
                                        "[INFO] control gate: "
                                        f"freshness_window={float(max(args.command_freshness_s, 0.0)):.3f}s "
                                        f"stale_grace={float(max(args.stale_grace_s, 0.0)):.3f}s"
                                    )
                                    print(f"[INFO] {_format_side_control_debug('left', left_status)}")
                                    print(f"[INFO] {_format_side_control_debug('right', right_status)}")
                                elif (left_has_control and not left_has_control_raw) or (
                                    right_has_control and not right_has_control_raw
                                ):
                                    print(
                                        "[INFO] control gate: using stale_grace to keep recording continuity "
                                        f"(left_raw={left_has_control_raw}, left_effective={left_has_control}, "
                                        f"right_raw={right_has_control_raw}, right_effective={right_has_control})."
                                    )
                                else:
                                    print(
                                        f"[INFO] ep={episode_idx} waiting for stable fresh leader action labels "
                                        f"({start_fresh_streak}/{required_start_fresh_frames})."
                                    )
                                _print_camera_summary(camera_health)
                                last_wait_log_s = now_s
                            if rr is not None:
                                display_frames_total += 1
                                rr.set_time("frame", sequence=display_frames_total)
                                for cam_key, image in camera_frames.items():
                                    entity = (
                                        rr.Image(image).compress()
                                        if display_opts.display_compressed_images
                                        else rr.Image(image)
                                    )
                                    rr.log(f"observation.images.{cam_key}", entity)
                                for idx, name in enumerate(vector_names):
                                    rr.log(f"observation.state/{name}", rr.Scalars(float(observation_state[idx])))
                                rr.log("status/left_cmd_echo", rr.Scalars(float(1.0 if left_has_control else 0.0)))
                                rr.log("status/right_cmd_echo", rr.Scalars(float(1.0 if right_has_control else 0.0)))
                                rr.log("status/left_obs_hz", rr.Scalars(float(robot.left_arm.bus.joint_feedback_hz)))
                                rr.log("status/right_obs_hz", rr.Scalars(float(robot.right_arm.bus.joint_feedback_hz)))
                                rr.log("status/left_cmd_hz", rr.Scalars(float(robot.left_arm.bus.command_feedback_hz)))
                                rr.log("status/right_cmd_hz", rr.Scalars(float(robot.right_arm.bus.command_feedback_hz)))
                            dt = time.perf_counter() - loop_start
                            precise_sleep(max((1.0 / dataset_opts.fps) - dt, 0.0))
                            continue

                    loop_index = int(current_episode_summary.get("total_loops", 0)) if current_episode_summary is not None else 0
                    if current_episode_summary is not None:
                        current_episode_summary["total_loops"] = int(current_episode_summary["total_loops"]) + 1

                    if missing_label:
                        _record_missing_label_loop(
                            summary=current_episode_summary,
                            loop_index=loop_index,
                            left_status=left_status,
                            right_status=right_status,
                        )
                        if args.missing_label_policy == "buffer-observation":
                            pending_observations.append(
                                BufferedObservation(
                                    loop_index=loop_index,
                                    observation_state=np.asarray(observation_state).copy(),
                                    camera_frames=_clone_camera_frames(camera_frames),
                                    left_status=left_status,
                                    right_status=right_status,
                                )
                            )
                            if current_episode_summary is not None:
                                current_episode_summary["max_buffered_observation_frames"] = max(
                                    int(current_episode_summary.get("max_buffered_observation_frames", 0)),
                                    len(pending_observations),
                                )
                            max_pending = max(int(args.buffered_observation_max_frames), 0)
                            overflow_dropped = 0
                            while len(pending_observations) > max_pending:
                                overflow = pending_observations.popleft()
                                overflow_dropped += 1
                                _record_dropped_loop(
                                    summary=current_episode_summary,
                                    loop_index=overflow.loop_index,
                                    drop_reason="buffer_overflow",
                                    left_status=overflow.left_status,
                                    right_status=overflow.right_status,
                                )
                            if overflow_dropped > 0 and current_episode_summary is not None:
                                current_episode_summary["buffer_overflow_dropped_frames"] = int(
                                    current_episode_summary.get("buffer_overflow_dropped_frames", 0)
                                ) + overflow_dropped
                        else:
                            _record_dropped_loop(
                                summary=current_episode_summary,
                                loop_index=loop_index,
                                drop_reason="missing_control",
                                left_status=left_status,
                                right_status=right_status,
                            )

                        if rr is not None:
                            display_frames_total += 1
                            rr.set_time("frame", sequence=display_frames_total)
                            for cam_key, image in camera_frames.items():
                                entity = rr.Image(image).compress() if display_opts.display_compressed_images else rr.Image(image)
                                rr.log(f"observation.images.{cam_key}", entity)
                            for idx, name in enumerate(vector_names):
                                rr.log(f"observation.state/{name}", rr.Scalars(float(observation_state[idx])))
                            rr.log("status/left_cmd_echo", rr.Scalars(float(1.0 if left_has_control else 0.0)))
                            rr.log("status/right_cmd_echo", rr.Scalars(float(1.0 if right_has_control else 0.0)))
                            rr.log("status/left_obs_hz", rr.Scalars(float(robot.left_arm.bus.joint_feedback_hz)))
                            rr.log("status/right_obs_hz", rr.Scalars(float(robot.right_arm.bus.joint_feedback_hz)))
                            rr.log("status/left_cmd_hz", rr.Scalars(float(robot.left_arm.bus.command_feedback_hz)))
                            rr.log("status/right_cmd_hz", rr.Scalars(float(robot.right_arm.bus.command_feedback_hz)))
                        dt = time.perf_counter() - loop_start
                        precise_sleep(max((1.0 / dataset_opts.fps) - dt, 0.0))
                        continue

                    action_state = _action_vector(
                        left_commanded=left_commanded,
                        right_commanded=right_commanded,
                        joint_names=joint_names,
                    )

                    if args.missing_label_policy == "buffer-observation" and pending_observations:
                        flush_count = 0
                        while pending_observations:
                            pending = pending_observations.popleft()
                            buffered_frame: dict[str, Any] = {
                                "observation.state": pending.observation_state,
                                "action": action_state,
                                "task": dataset_opts.single_task,
                            }
                            for cam_key, image in pending.camera_frames.items():
                                buffered_frame[f"observation.images.{cam_key}"] = image
                            dataset.add_frame(buffered_frame)
                            flush_count += 1
                            ep_frames += 1
                            frames_total += 1
                            if current_episode_summary is not None:
                                current_episode_summary["frames"] = int(current_episode_summary["frames"]) + 1
                                current_episode_summary["left_control_frames"] = int(
                                    current_episode_summary["left_control_frames"]
                                ) + 1
                                current_episode_summary["right_control_frames"] = int(
                                    current_episode_summary["right_control_frames"]
                                ) + 1
                                current_episode_summary["paired_from_buffer_frames"] = int(
                                    current_episode_summary.get("paired_from_buffer_frames", 0)
                                ) + 1
                        if flush_count > 0:
                            print(
                                f"[INFO] ep={episode_idx} recovered control; paired {flush_count} buffered observation loop(s)."
                            )

                    frame: dict[str, Any] = {
                        "observation.state": observation_state,
                        "action": action_state,
                        "task": dataset_opts.single_task,
                    }
                    for cam_key, image in camera_frames.items():
                        frame[f"observation.images.{cam_key}"] = image

                    dataset.add_frame(frame)
                    ep_frames += 1
                    frames_total += 1
                    display_frames_total += 1

                    if current_episode_summary is not None:
                        current_episode_summary["frames"] = int(current_episode_summary["frames"]) + 1
                        current_episode_summary["left_control_frames"] = int(
                            current_episode_summary["left_control_frames"]
                        ) + int(left_has_control)
                        current_episode_summary["right_control_frames"] = int(
                            current_episode_summary["right_control_frames"]
                        ) + int(right_has_control)

                    if rr is not None:
                        rr.set_time("frame", sequence=display_frames_total)
                        for cam_key, image in camera_frames.items():
                            entity = rr.Image(image).compress() if display_opts.display_compressed_images else rr.Image(image)
                            rr.log(f"observation.images.{cam_key}", entity)
                        for idx, name in enumerate(vector_names):
                            rr.log(f"observation.state/{name}", rr.Scalars(float(observation_state[idx])))
                            rr.log(f"action/{name}", rr.Scalars(float(action_state[idx])))
                        rr.log("status/left_cmd_echo", rr.Scalars(float(1.0 if left_has_control else 0.0)))
                        rr.log("status/right_cmd_echo", rr.Scalars(float(1.0 if right_has_control else 0.0)))
                        rr.log("status/left_obs_hz", rr.Scalars(float(robot.left_arm.bus.joint_feedback_hz)))
                        rr.log("status/right_obs_hz", rr.Scalars(float(robot.right_arm.bus.joint_feedback_hz)))
                        rr.log("status/left_cmd_hz", rr.Scalars(float(robot.left_arm.bus.command_feedback_hz)))
                        rr.log("status/right_cmd_hz", rr.Scalars(float(robot.right_arm.bus.command_feedback_hz)))

                    now_s = time.time()
                    if now_s - last_log_s >= 1.0:
                        print(
                            f"[INFO] ep={episode_idx} frame={ep_frames} "
                            f"left_cmd_echo={left_has_control} right_cmd_echo={right_has_control} "
                            f"left_obs_hz={robot.left_arm.bus.joint_feedback_hz:.1f} "
                            f"right_obs_hz={robot.right_arm.bus.joint_feedback_hz:.1f} "
                            f"left_cmd_hz={robot.left_arm.bus.command_feedback_hz:.1f} "
                            f"right_cmd_hz={robot.right_arm.bus.command_feedback_hz:.1f}"
                        )
                        _print_camera_summary(camera_health)
                        last_log_s = now_s

                    dt = time.perf_counter() - loop_start
                    precise_sleep(max((1.0 / dataset_opts.fps) - dt, 0.0))

                if args.missing_label_policy == "buffer-observation" and pending_observations and not events["rerecord_episode"]:
                    dropped_unpaired = _drop_pending_buffer(
                        pending_observations=pending_observations,
                        summary=current_episode_summary,
                        reason="episode_end_unpaired",
                    )
                    if dropped_unpaired > 0 and current_episode_summary is not None:
                        current_episode_summary["episode_end_unpaired_dropped_frames"] = int(
                            current_episode_summary.get("episode_end_unpaired_dropped_frames", 0)
                        ) + dropped_unpaired
                        print(
                            f"[INFO] ep={episode_idx} end-of-episode dropped {dropped_unpaired} unpaired buffered observation loop(s)."
                        )

                if events["rerecord_episode"]:
                    _say_event("rerecord_episode", dataset_opts)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    pending_observations.clear()
                    dataset.clear_episode_buffer()
                    current_episode_summary = None
                    print(f"[INFO] Discarded episode {episode_idx}, re-recording.")
                    continue

                if dataset.episode_buffer is None or dataset.episode_buffer.get("size", 0) == 0:
                    current_episode_summary = None
                    print(f"[WARN] Episode {episode_idx} has no frames; skipping save.")
                    if events["stop_recording"]:
                        break
                    continue

                if quick_save_requested:
                    _say_event("quick_save", dataset_opts)

                dataset.save_episode()
                saved_episodes += 1
                if current_episode_summary is not None:
                    episode_summaries.append(_finalize_episode_summary(current_episode_summary))
                    current_episode_summary = None
                _say_event("saved_episode", dataset_opts, episode_idx=episode_idx, blocking=True)
                print(f"[INFO] Saved episode {episode_idx} ({ep_frames} frames).")

                if not events["stop_recording"] and saved_episodes < dataset_opts.num_episodes and dataset_opts.reset_time_s > 0:
                    time.sleep(VOICE_EVENT_GAP_S)
                    _say_event("reset_environment", dataset_opts, blocking=True)
                    print(f"[INFO] Reset window: {dataset_opts.reset_time_s:.1f}s (Right Arrow to skip)")
                    reset_end = time.perf_counter() + dataset_opts.reset_time_s
                    while time.perf_counter() < reset_end:
                        quick_key_requested, rerecord_key_requested, stop_key_requested = _consume_keyboard_requests(
                            events,
                            keyboard_request_state,
                        )
                        if rerecord_key_requested:
                            events["rerecord_episode"] = True
                        if stop_key_requested:
                            events["stop_recording"] = True
                        if quick_key_requested or rerecord_key_requested or stop_key_requested or events["exit_early"]:
                            events["exit_early"] = False
                            break
                        robot.left_arm.bus.poll(max_msgs=robot.left_arm.bus.config.poll_max_msgs)
                        robot.right_arm.bus.poll(max_msgs=robot.right_arm.bus.config.poll_max_msgs)
                        precise_sleep(0.01)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Received Ctrl+C, stopping recording loop.")
        if args.missing_label_policy == "buffer-observation" and pending_observations:
            dropped_unpaired = _drop_pending_buffer(
                pending_observations=pending_observations,
                summary=current_episode_summary,
                reason="episode_end_unpaired",
            )
            if dropped_unpaired > 0 and current_episode_summary is not None:
                current_episode_summary["episode_end_unpaired_dropped_frames"] = int(
                    current_episode_summary.get("episode_end_unpaired_dropped_frames", 0)
                ) + dropped_unpaired
                print(f"[INFO] Dropped {dropped_unpaired} unpaired buffered observation loop(s) before exit.")
        if dataset is not None and dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
            dataset.save_episode()
            saved_episodes += 1
            if current_episode_summary is not None:
                episode_summaries.append(_finalize_episode_summary(current_episode_summary))
                current_episode_summary = None
            print("[INFO] Saved partial episode buffer before exit.")
    finally:
        _say_event("stop_recording", dataset_opts, blocking=True)
        if dataset is not None:
            dataset.finalize()
            summary_path = _write_capture_summary(
                root=dataset_opts.root,
                repo_id=dataset_opts.repo_id,
                joint_names=joint_names,
                episode_summaries=episode_summaries,
                command_freshness_s=float(max(args.command_freshness_s, 0.0)),
                stale_grace_s=float(max(args.stale_grace_s, 0.0)),
                missing_label_policy=args.missing_label_policy,
                buffered_observation_max_frames=max(int(args.buffered_observation_max_frames), 0),
                episode_start_min_fresh_frames=max(int(args.episode_start_min_fresh_frames), 1),
            )
            print(f"[INFO] Wrote action-source summary: {summary_path}")
            if dataset_opts.push_to_hub:
                dataset.push_to_hub(tags=dataset_opts.tags, private=dataset_opts.private)
                print("[INFO] Dataset pushed to Hugging Face Hub.")
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if rr is not None:
            try:
                rr.rerun_shutdown()
            except Exception:
                pass
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception:
            pass

        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        print(f"Episodes saved: {saved_episodes}")
        print(f"Frames written: {frames_total}")
        print(f"Dataset root: {dataset_opts.root}")
        if interrupted:
            print("[INFO] Recording stopped by user.")


if __name__ == "__main__":
    main()
