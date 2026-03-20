#!/usr/bin/env python

"""Record Songling ALOHA dataset with 3 cameras + raw CAN parsing.

This path bypasses Damiao handshake and records directly from:
- three UVC cameras (`left_high`, `left_elbow`, `right_elbow`)
- two CAN interfaces (left and right integrated chains)

Output is a standard LeRobot dataset created with `LeRobotDataset.create`.
"""

from __future__ import annotations

import argparse
import ast
import glob
import io
import json
import math
import os
import platform
import re
import shutil
import shlex
import subprocess
import tempfile
import threading
import time
import wave
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.songling_follower.protocol import (
    ARM_GRIPPER_CTRL,
    ARM_GRIPPER_FEEDBACK,
    ARM_JOINT_CTRL_12,
    ARM_JOINT_CTRL_34,
    ARM_JOINT_CTRL_56,
    ARM_JOINT_FEEDBACK_12,
    ARM_JOINT_FEEDBACK_34,
    ARM_JOINT_FEEDBACK_56,
    decode_gripper_feedback as decode_piper_gripper_feedback,
    decode_joint_feedback as decode_piper_joint_feedback,
)
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
try:
    from songling_protocol import (
        DEFAULT_SONGLING_STATE_IDS,
        decode_known_state,
        is_known_state_id,
    )
except ModuleNotFoundError:
    from examples.songling_aloha.songling_protocol import (
        DEFAULT_SONGLING_STATE_IDS,
        decode_known_state,
        is_known_state_id,
    )

CAMERA_KEYS = ("left_high", "left_elbow", "right_elbow")
DEFAULT_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
PIPER_OBSERVATION_FRAME_IDS = (
    ARM_JOINT_FEEDBACK_12,
    ARM_JOINT_FEEDBACK_34,
    ARM_JOINT_FEEDBACK_56,
    ARM_GRIPPER_FEEDBACK,
)
PIPER_ACTION_FRAME_IDS = (
    ARM_JOINT_CTRL_12,
    ARM_JOINT_CTRL_34,
    ARM_JOINT_CTRL_56,
    ARM_GRIPPER_CTRL,
)
DEFAULT_OBSERVATION_IDS = ",".join(hex(v) for v in PIPER_OBSERVATION_FRAME_IDS)
DEFAULT_ACTION_IDS = ",".join(hex(v) for v in PIPER_ACTION_FRAME_IDS)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "songling_aloha"
CAPTURE_SUMMARY_FILENAME = "songling_recording_summary.json"
LOCAL_PIPER_BINARY = PROJECT_ROOT / "third_party" / "piper" / "piper" / "piper"
LOCAL_PIPER_MODELS_DIR = PROJECT_ROOT / "third_party" / "piper" / "models"
VOICE_ENGINE_CHOICES = ("auto", "piper", "spd-say", "espeak-ng", "espeak", "tone", "log-say", "bell", "off")
_VOICE_WARN_ONCE_KEYS: set[str] = set()
_VOICE_LANG_SUPPORT_CACHE: dict[tuple[str, str], bool] = {}
_TONE_WAV_CACHE: dict[str, bytes] = {}
_PIPER_MODEL_DISCOVERY_CACHE: dict[str, str | None] = {}
_VOICE_PLAY_LOCK = threading.Lock()


def _patch_multiprocess_resource_tracker() -> None:
    """Work around multiprocess<->Python3.12 RLock private API mismatch."""
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
class CANDecodeOptions:
    byte_offset: int
    byte_length: int
    signed: bool
    endian: str
    scale: float
    bias: float


@dataclass
class DisplayOptions:
    display_data: bool
    display_ip: str | None
    display_port: int | None
    display_compressed_images: bool


@dataclass
class SideCANRuntime:
    interface: str
    bitrate: int
    data_bitrate: int | None
    use_fd: bool
    observation_ids: list[int]
    action_ids: list[int]


@dataclass
class CANBusState:
    latest: dict[int, float] = field(default_factory=dict)
    latest_ts: dict[int, float] = field(default_factory=dict)
    latest_payload: dict[int, bytes] = field(default_factory=dict)
    seen_ids: set[int] = field(default_factory=set)
    total_msgs: int = 0
    recent_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=4096))


def _parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false.")


def _capture_summary_path(root: Path) -> Path:
    return root / "meta" / CAPTURE_SUMMARY_FILENAME


def _load_capture_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_hex_ids(values: Any) -> set[int]:
    if not isinstance(values, list):
        return set()

    out: set[int] = set()
    for value in values:
        try:
            out.add(int(str(value), 0))
        except Exception:
            continue
    return out


def _format_hex_ids(values: set[int]) -> list[str]:
    return [hex(v) for v in sorted(values)]


def _finalize_action_source_summary(summary: dict[str, Any]) -> dict[str, Any]:
    frames = int(summary.get("frames", 0))
    left_control = int(summary.get("left_control_frames", 0))
    right_control = int(summary.get("right_control_frames", 0))
    if frames < 0:
        frames = 0

    summary["frames"] = frames
    summary["left_fallback_frames"] = max(frames - left_control, 0)
    summary["right_fallback_frames"] = max(frames - right_control, 0)
    summary["left_control_ratio"] = (left_control / frames) if frames > 0 else 0.0
    summary["right_control_ratio"] = (right_control / frames) if frames > 0 else 0.0
    return summary


def _write_capture_summary(
    *,
    root: Path,
    repo_id: str,
    joint_names: list[str],
    left_can: SideCANRuntime,
    right_can: SideCANRuntime,
    episode_summaries: list[dict[str, Any]],
    left_seen_ids: set[int],
    right_seen_ids: set[int],
) -> Path:
    summary_path = _capture_summary_path(root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_capture_summary(summary_path)
    existing_episodes = existing.get("episodes")
    episode_map: dict[int, dict[str, Any]] = {}
    if isinstance(existing_episodes, list):
        for entry in existing_episodes:
            if isinstance(entry, dict) and "episode_index" in entry:
                try:
                    episode_map[int(entry["episode_index"])] = dict(entry)
                except Exception:
                    continue

    for entry in episode_summaries:
        episode_map[int(entry["episode_index"])] = _finalize_action_source_summary(dict(entry))

    existing_seen = existing.get("seen_ids") if isinstance(existing.get("seen_ids"), dict) else {}
    merged_left_seen = left_seen_ids | _parse_hex_ids(existing_seen.get("left"))
    merged_right_seen = right_seen_ids | _parse_hex_ids(existing_seen.get("right"))

    payload = {
        "schema_version": 1,
        "repo_id": repo_id,
        "dataset_root": str(root),
        "joint_names_per_side": list(joint_names),
        "units": {
            "arm_joint_storage": "degree",
            "gripper_storage": "mm",
            "protocol_reference": "Piper-style Songling CAN: joints are 0.001 degree, gripper is 0.001 mm.",
            "mobile_aloha_projection": "Convert joints degree->rad and gripper mm->[0,1] by dividing with the verified gripper max stroke.",
        },
        "can_frame_ids": {
            "left_observation": [hex(v) for v in left_can.observation_ids],
            "right_observation": [hex(v) for v in right_can.observation_ids],
            "left_action": [hex(v) for v in left_can.action_ids],
            "right_action": [hex(v) for v in right_can.action_ids],
        },
        "episodes": [episode_map[idx] for idx in sorted(episode_map)],
        "seen_ids": {
            "left": _format_hex_ids(merged_left_seen),
            "right": _format_hex_ids(merged_right_seen),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _parse_optional_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    return int(value)


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_tags(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError(f"--dataset.tags expects list syntax when using brackets, got: {text}")
        return [str(v) for v in parsed if str(v).strip()]
    return [item for item in _parse_csv(text) if item]


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


def _parse_id_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in _parse_csv(raw):
        out.append(int(item, 0))
    return out


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at `{path}` in YAML config, got: {type(value)}")
    return value


def _next_available_directory(root_path: Path) -> Path:
    if not root_path.exists():
        return root_path

    parent = root_path.parent
    stem = root_path.name
    match = re.match(r"^(.*?)(?:_(\d+))?$", stem)
    if match:
        base = match.group(1)
        num = match.group(2)
    else:
        base = stem
        num = None

    width = max(len(num), 3) if num is not None else 3
    idx = int(num) if num is not None else 0
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

    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        raise PermissionError(
            f"Cannot create/access parent directory for dataset root: {parent}\n"
            "Please choose a writable local path under your user directory."
        ) from None

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


def _clamp_voice_rate(rate: int) -> int:
    return max(-100, min(100, int(rate)))


def _coerce_voice_engine(value: Any) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "disabled": "off",
        "spd": "spd-say",
        "espeakng": "espeak-ng",
    }
    engine = aliases.get(normalized, normalized)
    if engine not in VOICE_ENGINE_CHOICES:
        raise ValueError(
            f"Unsupported voice engine: {value}. "
            f"Use one of: {', '.join(VOICE_ENGINE_CHOICES)}."
        )
    return engine


def _warn_voice_once(key: str, message: str) -> None:
    if key in _VOICE_WARN_ONCE_KEYS:
        return
    _VOICE_WARN_ONCE_KEYS.add(key)
    print(message)


def _probe_command_output(cmd: list[str], timeout_s: float = 2.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return False, str(exc)

    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip().lower()
    return proc.returncode == 0, combined


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
    cache_key = lang
    if cache_key in _PIPER_MODEL_DISCOVERY_CACHE:
        return _PIPER_MODEL_DISCOVERY_CACHE[cache_key]

    env_candidates: list[str] = []
    if lang == "zh":
        env_candidates.extend(
            [
                os.environ.get("PIPER_MODEL_ZH", ""),
                os.environ.get("PIPER_MODEL_ZH_CN", ""),
            ]
        )
    elif lang == "en":
        env_candidates.extend(
            [
                os.environ.get("PIPER_MODEL_EN", ""),
                os.environ.get("PIPER_MODEL_EN_US", ""),
            ]
        )
    env_candidates.append(os.environ.get("PIPER_MODEL", ""))

    for raw in env_candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists() and path.is_file():
            _PIPER_MODEL_DISCOVERY_CACHE[cache_key] = str(path)
            return str(path)

    roots = [
        LOCAL_PIPER_MODELS_DIR,
        Path.home() / ".local/share/piper",
        Path.home() / ".cache/piper",
        Path("/usr/share/piper"),
        Path("/opt/piper/models"),
    ]
    if lang == "zh":
        patterns = ("*zh*.onnx", "*cmn*.onnx", "*chinese*.onnx")
    else:
        patterns = ("*en*.onnx", "*english*.onnx")

    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                matches = sorted(root.rglob(pattern))
            except Exception:
                matches = []
            if matches:
                _PIPER_MODEL_DISCOVERY_CACHE[cache_key] = str(matches[0])
                return str(matches[0])

    _PIPER_MODEL_DISCOVERY_CACHE[cache_key] = None
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


def _engine_supports_language(
    engine: str,
    lang: str,
    *,
    piper_binary: str | None = None,
    piper_model: str | None = None,
) -> bool:
    engine_key = engine
    if engine == "piper":
        engine_key = f"piper|{piper_binary or ''}|{piper_model or ''}"
    key = (engine_key, lang)
    if key in _VOICE_LANG_SUPPORT_CACHE:
        return _VOICE_LANG_SUPPORT_CACHE[key]

    supported = False
    if lang == "en":
        if engine in {"off", "bell", "tone", "log-say"}:
            supported = True
        elif engine == "piper":
            bin_path = _resolve_piper_binary(piper_binary)
            model_path = _resolve_piper_model_for_lang(lang, piper_model)
            supported = bin_path is not None and model_path is not None
        else:
            supported = shutil.which(engine) is not None
    elif lang == "zh":
        if engine in {"off", "bell", "tone"}:
            supported = True
        elif engine == "log-say":
            # log_say cannot select language on Linux; treat zh support as unknown/false.
            supported = False
        elif engine == "piper":
            bin_path = _resolve_piper_binary(piper_binary)
            model_path = _resolve_piper_model_for_lang(lang, piper_model)
            supported = bin_path is not None and model_path is not None
        elif engine == "spd-say":
            if shutil.which("spd-say") is not None:
                ok, output = _probe_command_output(["spd-say", "-L"])
                if ok:
                    supported = bool(re.search(r"\b(zh|zh-cn|zh_cn|cmn|chinese|mandarin)\b", output))
        elif engine in {"espeak-ng", "espeak"}:
            if shutil.which(engine) is not None:
                ok, output = _probe_command_output([engine, "--voices"])
                if ok:
                    supported = bool(re.search(r"\b(zh|cmn|yue|zhy)\b", output))

    _VOICE_LANG_SUPPORT_CACHE[key] = supported
    return supported


def _has_any_tts_for_language(
    lang: str,
    requested_engine: str,
    *,
    piper_binary: str | None = None,
    piper_model: str | None = None,
) -> bool:
    for engine in _ordered_voice_engines(requested_engine=requested_engine):
        if engine in {"off", "bell", "tone"}:
            continue
        if _engine_supports_language(engine, lang, piper_binary=piper_binary, piper_model=piper_model):
            return True
    return False


def _voice_bell_count(event: str | None) -> int:
    table = {
        "start_recording": 2,
        "recording_episode": 1,
        "rerecord_episode": 3,
        "quick_save": 2,
        "saved_episode": 1,
        "reset_environment": 2,
        "stop_recording": 3,
    }
    return table.get(event or "", 1)


def _voice_tone_pattern(event: str | None) -> list[tuple[int, float]]:
    # (frequency_hz, duration_seconds). frequency<=0 means silence.
    table: dict[str, list[tuple[int, float]]] = {
        "start_recording": [(880, 0.10), (1047, 0.12)],
        "recording_episode": [(740, 0.10)],
        "rerecord_episode": [(350, 0.10), (350, 0.10), (350, 0.10)],
        "quick_save": [(880, 0.08), (660, 0.12)],
        "saved_episode": [(988, 0.14)],
        "reset_environment": [(523, 0.10), (659, 0.10)],
        "stop_recording": [(659, 0.10), (523, 0.10), (392, 0.18)],
    }
    return table.get(event or "", [(740, 0.10)])


def _build_tone_wav_bytes(event: str | None, sample_rate: int = 22050) -> bytes:
    key = event or "_default"
    if key in _TONE_WAV_CACHE:
        return _TONE_WAV_CACHE[key]

    pattern = _voice_tone_pattern(event)
    amplitude = 0.32
    gap_s = 0.045
    pcm = bytearray()

    for idx, (freq_hz, duration_s) in enumerate(pattern):
        n = max(1, int(duration_s * sample_rate))
        if freq_hz <= 0:
            for _ in range(n):
                pcm.extend((0).to_bytes(2, byteorder="little", signed=True))
        else:
            for i in range(n):
                val = int(32767 * amplitude * math.sin(2.0 * math.pi * float(freq_hz) * (i / sample_rate)))
                pcm.extend(int(val).to_bytes(2, byteorder="little", signed=True))
        if idx + 1 < len(pattern):
            gap_n = int(gap_s * sample_rate)
            for _ in range(gap_n):
                pcm.extend((0).to_bytes(2, byteorder="little", signed=True))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(pcm))

    data = buf.getvalue()
    _TONE_WAV_CACHE[key] = data
    return data


def _play_wav_bytes(
    wav_data: bytes,
    *,
    blocking: bool,
) -> bool:
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


def _play_tone_event(event: str | None, *, blocking: bool) -> bool:
    try:
        wav_data = _build_tone_wav_bytes(event)
    except Exception as exc:
        _warn_voice_once("tone-build-failed", f"[WARN] Failed to build tone pattern ({exc}).")
        return False
    return _play_wav_bytes(wav_data, blocking=blocking)


def _speak_with_piper_blocking(
    text: str,
    *,
    lang: str,
    piper_binary: str | None,
    piper_model: str | None,
    piper_speaker: int | None,
) -> bool:
    # Serialize neural TTS playback to avoid overlapping voices across close events.
    with _VOICE_PLAY_LOCK:
        bin_path = _resolve_piper_binary(piper_binary)
        if bin_path is None:
            _warn_voice_once(
                "missing-piper-bin",
                "[WARN] Piper binary not found. Install `piper` or set --voice-piper-binary.",
            )
            return False

        model_path = _resolve_piper_model_for_lang(lang, piper_model)
        if model_path is None:
            hint = (
                "[WARN] Piper model not found for current language. "
                "Set --voice-piper-model=/abs/path/to/model.onnx."
            )
            _warn_voice_once(f"missing-piper-model-{lang}", hint)
            return False

        fd, out_path_str = tempfile.mkstemp(prefix="songling_piper_", suffix=".wav")
        os.close(fd)
        out_path = Path(out_path_str)
        try:
            cmd = [bin_path, "--model", model_path, "--output_file", str(out_path)]
            if piper_speaker is not None:
                cmd.extend(["--speaker", str(int(piper_speaker))])
            piper_env = _piper_env_with_runtime_libs(bin_path)
            subprocess.run(
                cmd,
                input=text,
                text=True,
                check=True,
                env=piper_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wav_data = out_path.read_bytes()
            return _play_wav_bytes(wav_data, blocking=True)
        except Exception as exc:
            _warn_voice_once("piper-synth-failed", f"[WARN] Piper synthesis failed ({exc}).")
            return False
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass


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


def _ring_terminal_bell(count: int, *, blocking: bool, interval_s: float = 0.12) -> None:
    def _worker() -> None:
        repeats = max(1, int(count))
        for idx in range(repeats):
            print("\a", end="", flush=True)
            if idx + 1 < repeats:
                time.sleep(interval_s)

    if blocking:
        _worker()
        return

    try:
        threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        _worker()


def _espeak_speed_from_rate(rate: int) -> int:
    clamped = _clamp_voice_rate(rate)
    # espeak uses words-per-minute; this keeps defaults slower for intelligibility.
    return max(90, min(260, 170 + int(clamped * 0.8)))


def _ordered_voice_engines(requested_engine: str) -> list[str]:
    if requested_engine == "off":
        return ["off"]

    if requested_engine == "auto":
        base = ["piper", "spd-say", "espeak-ng", "espeak", "tone", "log-say", "bell"]
    else:
        base = [requested_engine, "piper", "spd-say", "espeak-ng", "espeak", "tone", "log-say", "bell"]

    if platform.system() != "Linux":
        base = [requested_engine, "tone", "log-say", "bell"] if requested_engine != "auto" else ["tone", "log-say", "bell"]

    deduped: list[str] = []
    seen: set[str] = set()
    for engine in base:
        if engine not in seen:
            deduped.append(engine)
            seen.add(engine)
    return deduped


def _run_tts_command(cmd: list[str], blocking: bool) -> tuple[bool, str | None]:
    try:
        if blocking:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Detect immediate spawn/runtime failure while keeping non-blocking behavior.
            time.sleep(0.04)
            rc = proc.poll()
            if rc is not None and rc != 0:
                return False, f"exit code {rc}"
        return True, None
    except Exception as exc:
        return False, str(exc)


def _speak_with_engine(
    text: str,
    *,
    lang: str,
    voice_rate: int,
    voice_engine: str,
    blocking: bool,
    event: str | None,
    piper_binary: str | None,
    piper_model: str | None,
    piper_speaker: int | None,
) -> bool:
    if voice_engine == "off":
        return True

    if voice_engine == "piper":
        if _speak_with_piper(
            text,
            lang=lang,
            piper_binary=piper_binary,
            piper_model=piper_model,
            piper_speaker=piper_speaker,
            blocking=blocking,
        ):
            return True
        _warn_voice_once("piper-play-failed", "[WARN] Piper backend failed, trying fallback engines.")
        return False

    if not _engine_supports_language(
        voice_engine,
        lang,
        piper_binary=piper_binary,
        piper_model=piper_model,
    ):
        _warn_voice_once(
            f"no-lang-{voice_engine}-{lang}",
            f"[WARN] Voice backend {voice_engine} does not provide language={lang}, skipping.",
        )
        return False

    if voice_engine == "tone":
        if _play_tone_event(event, blocking=blocking):
            return True
        _warn_voice_once("tone-play-failed", "[WARN] Tone backend unavailable, falling back.")
        return False

    if voice_engine == "bell":
        _ring_terminal_bell(_voice_bell_count(event), blocking=blocking)
        return True

    if voice_engine == "log-say":
        try:
            log_say(text, play_sounds=True, blocking=blocking)
            return True
        except Exception as exc:
            _warn_voice_once(
                "log-say-failed",
                f"[WARN] log_say failed ({exc}), trying other voice backends.",
            )
            return False

    if voice_engine == "spd-say":
        if shutil.which("spd-say") is None:
            _warn_voice_once("missing-spd-say", "[WARN] spd-say is not installed, skipping this voice backend.")
            return False

        lang_codes = ("zh-CN", "zh", "cmn") if lang == "zh" else ("en-US", "en")
        last_error: str | None = None
        for lang_code in lang_codes:
            cmd = ["spd-say", "-l", lang_code, "-r", str(_clamp_voice_rate(voice_rate))]
            if blocking:
                cmd.append("--wait")
            cmd.append(text)
            ok, err = _run_tts_command(cmd, blocking=blocking)
            if ok:
                return True
            last_error = err

        _warn_voice_once(
            f"spd-say-failed-{lang}",
            f"[WARN] spd-say failed for language={lang} ({last_error}), trying other voice backends.",
        )
        return False

    if voice_engine in {"espeak-ng", "espeak"}:
        if shutil.which(voice_engine) is None:
            _warn_voice_once(
                f"missing-{voice_engine}",
                f"[WARN] {voice_engine} is not installed, skipping this voice backend.",
            )
            return False

        speed = str(_espeak_speed_from_rate(voice_rate))
        voices = ("zh", "cmn") if lang == "zh" else ("en-us", "en")
        last_error: str | None = None
        for voice in voices:
            cmd = [voice_engine, "-v", voice, "-s", speed, text]
            ok, err = _run_tts_command(cmd, blocking=blocking)
            if ok:
                return True
            last_error = err

        _warn_voice_once(
            f"{voice_engine}-failed-{lang}",
            f"[WARN] {voice_engine} failed for language={lang} ({last_error}), trying other voice backends.",
        )
        return False

    _warn_voice_once(f"unknown-engine-{voice_engine}", f"[WARN] Unknown voice engine {voice_engine!r}, skipping.")
    return False


def _try_speak_text(
    text: str,
    *,
    lang: str,
    voice_rate: int,
    voice_engine: str,
    blocking: bool,
    event: str | None,
    piper_binary: str | None,
    piper_model: str | None,
    piper_speaker: int | None,
) -> str | None:
    for engine in _ordered_voice_engines(requested_engine=voice_engine):
        if _speak_with_engine(
            text,
            lang=lang,
            voice_rate=voice_rate,
            voice_engine=engine,
            blocking=blocking,
            event=event,
            piper_binary=piper_binary,
            piper_model=piper_model,
            piper_speaker=piper_speaker,
        ):
            return engine
    return None


def _safe_log_say(
    text: str,
    play_sounds: bool,
    blocking: bool = False,
    *,
    lang: str = "en",
    voice_rate: int = 0,
    voice_engine: str = "auto",
    piper_binary: str | None = None,
    piper_model: str | None = None,
    piper_speaker: int | None = None,
    fallback_text: str | None = None,
    event: str | None = None,
    speak_hint_after_primary: bool = False,
) -> None:
    print(f"[VOICE] {text}")
    if not play_sounds:
        return

    # Avoid garbled playback: if no backend can truly speak Chinese, skip zh text and go straight to hint fallback.
    if lang == "zh" and fallback_text and not _has_any_tts_for_language(
        "zh",
        voice_engine,
        piper_binary=piper_binary,
        piper_model=piper_model,
    ):
        _warn_voice_once(
            "no-zh-tts",
            "[WARN] No Chinese TTS voice detected. Falling back to concise English voice hints.",
        )
        print(f"[VOICE:FALLBACK] {fallback_text}")
        if _try_speak_text(
            fallback_text,
            lang="en",
            voice_rate=voice_rate,
            voice_engine=voice_engine,
            blocking=blocking,
            event=event,
            piper_binary=piper_binary,
            piper_model=piper_model,
            piper_speaker=piper_speaker,
        ):
            return

    primary_engine = _try_speak_text(
        text,
        lang=lang,
        voice_rate=voice_rate,
        voice_engine=voice_engine,
        blocking=blocking,
        event=event,
        piper_binary=piper_binary,
        piper_model=piper_model,
        piper_speaker=piper_speaker,
    )
    if primary_engine is not None:
        # For zh prompts, also speak a short English hint to ensure operator can always distinguish events.
        if lang == "zh" and fallback_text and speak_hint_after_primary:
            _try_speak_text(
                fallback_text,
                lang="en",
                voice_rate=voice_rate,
                voice_engine=voice_engine,
                blocking=False,
                event=event,
                piper_binary=piper_binary,
                piper_model=piper_model,
                piper_speaker=piper_speaker,
            )
        return

    # If Chinese TTS is unavailable or unclear, auto-fallback to short English prompt.
    if lang == "zh" and fallback_text:
        print(f"[VOICE:FALLBACK] {fallback_text}")
        if _try_speak_text(
            fallback_text,
            lang="en",
            voice_rate=voice_rate,
            voice_engine=voice_engine,
            blocking=blocking,
            event=event,
            piper_binary=piper_binary,
            piper_model=piper_model,
            piper_speaker=piper_speaker,
        ):
            return

    _warn_voice_once("voice-all-failed", "[WARN] All voice backends failed, using terminal bell fallback.")
    _ring_terminal_bell(_voice_bell_count(event), blocking=blocking)


def _voice_text(event: str, lang: str, **kwargs: Any) -> str:
    raw_episode_idx = kwargs.get("episode_idx", 0)
    try:
        spoken_episode_idx = int(raw_episode_idx) + 1
    except Exception:
        spoken_episode_idx = raw_episode_idx

    if lang == "zh":
        table = {
            "start_recording": "开始录制。",
            "recording_episode": f"第 {spoken_episode_idx} 段，开始录制。",
            "rerecord_episode": "重录当前片段。",
            "quick_save": "快速保存当前片段。",
            "saved_episode": f"第 {spoken_episode_idx} 段，保存完成。",
            "reset_environment": "请重置场景。",
            "stop_recording": "录制结束。",
        }
    else:
        table = {
            "start_recording": "Start recording.",
            "recording_episode": f"Recording episode {spoken_episode_idx}.",
            "rerecord_episode": "Re-record current episode.",
            "quick_save": "Quick save current segment.",
            "saved_episode": f"Episode {spoken_episode_idx} saved.",
            "reset_environment": "Reset the environment.",
            "stop_recording": "Recording finished.",
        }
    return table.get(event, event)


def _voice_hint_text(event: str, **kwargs: Any) -> str:
    raw_episode_idx = kwargs.get("episode_idx", 0)
    try:
        spoken_episode_idx = int(raw_episode_idx) + 1
    except Exception:
        spoken_episode_idx = raw_episode_idx

    table = {
        "start_recording": "Start recording.",
        "recording_episode": f"Episode {spoken_episode_idx}, start.",
        "rerecord_episode": "Re-record.",
        "quick_save": "Quick save.",
        "saved_episode": f"Episode {spoken_episode_idx}, saved.",
        "reset_environment": "Reset environment.",
        "stop_recording": "Recording finished.",
    }
    return table.get(event, str(event))


def _say_event(event: str, opts: DatasetOptions, blocking: bool = False, **kwargs: Any) -> None:
    text = _voice_text(event, opts.voice_lang, **kwargs)
    fallback_text = _voice_hint_text(event, **kwargs) if opts.voice_lang == "zh" else None
    _safe_log_say(
        text,
        opts.play_sounds,
        blocking=blocking,
        lang=opts.voice_lang,
        voice_rate=opts.voice_rate,
        voice_engine=opts.voice_engine,
        piper_binary=opts.voice_piper_binary,
        piper_model=opts.voice_piper_model,
        piper_speaker=opts.voice_piper_speaker,
        fallback_text=fallback_text,
        event=event,
        speak_hint_after_primary=False,
    )


def _parse_index_or_path(value: Any) -> int | Path:
    if isinstance(value, int):
        return value
    if isinstance(value, Path):
        return value

    text = _strip_wrapping_quotes(str(value)).strip()
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
        raise ValueError(
            "Songling raw record currently supports OpenCV cameras only. "
            f"Got camera type={cam_type!r}."
        )
    if "index_or_path" not in raw:
        raise ValueError("Camera config misses `index_or_path`.")

    width_raw = raw.get("width", 640)
    height_raw = raw.get("height", 480)
    fps_raw = raw.get("fps", fallback_fps)

    source = _normalize_opencv_source(_parse_index_or_path(raw["index_or_path"]))

    return OpenCVCameraConfig(
        index_or_path=source,
        width=int(640 if width_raw is None else width_raw),
        height=int(480 if height_raw is None else height_raw),
        fps=int(fallback_fps if fps_raw is None else fps_raw),
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
        index_or_path = cam_cfg.index_or_path
        if isinstance(index_or_path, Path):
            if not index_or_path.exists():
                missing.append((cam_key, index_or_path))
                continue
            # OpenCV may fail silently; preflight open gives clearer diagnostics.
            try:
                fd = os.open(index_or_path, os.O_RDONLY | os.O_NONBLOCK)
                os.close(fd)
            except OSError as exc:
                inaccessible.append((cam_key, index_or_path, str(exc)))

    if not missing and not inaccessible:
        return

    available = sorted(glob.glob("/dev/v4l/by-id/*"))
    available_text = "\n".join(f"  - {p}" for p in available) if available else "  (none)"
    issues: list[str] = []
    if missing:
        missing_text = "\n".join(f"  - {cam_key}: {path}" for cam_key, path in missing)
        issues.append("Missing camera device path(s):\n" + missing_text)
    if inaccessible:
        inaccessible_text = "\n".join(f"  - {cam_key}: {path} ({err})" for cam_key, path, err in inaccessible)
        issues.append(
            "Camera device path exists but cannot be opened:\n"
            + inaccessible_text
            + "\nHint: run from a host terminal with V4L2 access, or ensure your runtime/container is allowed to use /dev/video*."
        )

    raise FileNotFoundError(
        "\n".join(issues)
        + "\nCurrent /dev/v4l/by-id entries:\n"
        + available_text
        + "\nPlease replug the missing camera or override camera paths on CLI."
    )


def _decode_signal(data: bytes, decode: CANDecodeOptions) -> float | None:
    end = decode.byte_offset + decode.byte_length
    if len(data) < end:
        return None
    raw = int.from_bytes(
        bytes(data[decode.byte_offset:end]),
        byteorder=decode.endian,
        signed=decode.signed,
    )
    return (float(raw) * decode.scale) + decode.bias


def _decode_songling_or_fallback(arbitration_id: int, data: bytes, decode: CANDecodeOptions) -> float | None:
    if is_known_state_id(arbitration_id):
        decoded = decode_known_state(arbitration_id, data)
        if decoded is not None:
            return decoded
    return _decode_signal(data, decode)


def _open_raw_can_bus(interface: str, bitrate: int, use_fd: bool, data_bitrate: int | None):
    import can

    kwargs: dict[str, Any] = {"channel": interface, "interface": "socketcan", "bitrate": bitrate}
    if use_fd:
        kwargs["fd"] = True
        if data_bitrate is not None:
            kwargs["data_bitrate"] = data_bitrate
    return can.interface.Bus(**kwargs)


def _poll_can_state(bus, state: CANBusState, decode: CANDecodeOptions, max_msgs: int) -> None:
    got = 0
    while got < max_msgs:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            break

        got += 1
        now = time.time()
        state.total_msgs += 1
        state.recent_timestamps.append(now)
        state.seen_ids.add(msg.arbitration_id)
        state.latest_payload[msg.arbitration_id] = bytes(msg.data)

        decoded = _decode_songling_or_fallback(msg.arbitration_id, msg.data, decode)
        if decoded is not None:
            state.latest[msg.arbitration_id] = decoded
            state.latest_ts[msg.arbitration_id] = now


def _rx_hz(ts_q: deque[float]) -> float:
    if not ts_q:
        return 0.0
    now = time.time()
    while ts_q and (now - ts_q[0]) > 1.0:
        ts_q.popleft()
    return float(len(ts_q))


def _values_from_ids(ids: list[int], state: CANBusState, last_values: np.ndarray) -> np.ndarray:
    out = np.empty((len(ids),), dtype=np.float32)
    for i, can_id in enumerate(ids):
        if can_id in state.latest:
            value = float(state.latest[can_id])
            last_values[i] = value
        else:
            value = float(last_values[i])
        out[i] = value
    return out


def _decode_piper_observation_vector(
    configured_frame_ids: list[int],
    state: CANBusState,
    last_values: np.ndarray,
) -> np.ndarray:
    out = np.array(last_values, copy=True)
    allowed = set(configured_frame_ids)

    for arbitration_id in (ARM_JOINT_FEEDBACK_12, ARM_JOINT_FEEDBACK_34, ARM_JOINT_FEEDBACK_56):
        if arbitration_id not in allowed:
            continue
        payload = state.latest_payload.get(arbitration_id)
        if payload is None:
            continue
        try:
            decoded = decode_piper_joint_feedback(arbitration_id, payload)
        except Exception:
            continue
        for idx, joint_name in enumerate(DEFAULT_JOINT_NAMES[:6]):
            if joint_name in decoded:
                out[idx] = float(decoded[joint_name]) * 1e-3

    if ARM_GRIPPER_FEEDBACK in allowed:
        payload = state.latest_payload.get(ARM_GRIPPER_FEEDBACK)
        if payload is not None:
            try:
                decoded_gripper = decode_piper_gripper_feedback(payload)
                out[6] = float(decoded_gripper.position) * 1e-3
            except Exception:
                pass

    last_values[:] = out
    return out


def _decode_piper_action_vector(
    configured_frame_ids: list[int],
    state: CANBusState,
    last_values: np.ndarray,
    *,
    fallback_observation: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    out = np.array(last_values, copy=True)
    allowed = set(configured_frame_ids)
    saw_control_frame = False
    index_map = {name: idx for idx, name in enumerate(DEFAULT_JOINT_NAMES)}

    control_pairs = {
        ARM_JOINT_CTRL_12: ("joint_1", "joint_2"),
        ARM_JOINT_CTRL_34: ("joint_3", "joint_4"),
        ARM_JOINT_CTRL_56: ("joint_5", "joint_6"),
    }
    for arbitration_id, joint_names in control_pairs.items():
        if arbitration_id not in allowed:
            continue
        payload = state.latest_payload.get(arbitration_id)
        if payload is None or len(payload) < 8:
            continue
        first = int.from_bytes(payload[0:4], byteorder="big", signed=True)
        second = int.from_bytes(payload[4:8], byteorder="big", signed=True)
        out[index_map[joint_names[0]]] = float(first) * 1e-3
        out[index_map[joint_names[1]]] = float(second) * 1e-3
        saw_control_frame = True

    if ARM_GRIPPER_CTRL in allowed:
        payload = state.latest_payload.get(ARM_GRIPPER_CTRL)
        if payload is not None and len(payload) >= 8:
            gripper_pos = int.from_bytes(payload[0:4], byteorder="big", signed=True)
            out[index_map["gripper"]] = float(gripper_pos) * 1e-3
            saw_control_frame = True

    if not saw_control_frame and fallback_observation is not None:
        out = np.asarray(fallback_observation, dtype=np.float32).copy()

    last_values[:] = out
    return out, saw_control_frame


def _read_camera_frames(
    cameras: dict[str, Any],
    last_frames: dict[str, np.ndarray],
    camera_max_age_ms: int,
    camera_retry_timeout_ms: int,
) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for cam_key, cam in cameras.items():
        frame = None
        try:
            frame = cam.read_latest(max_age_ms=camera_max_age_ms)
        except Exception:
            try:
                frame = cam.async_read(timeout_ms=max(camera_retry_timeout_ms, 1))
            except Exception:
                frame = last_frames.get(cam_key)
        if frame is None:
            raise RuntimeError(
                f"Camera '{cam_key}' returned no frame. Check cable/power/device path, then retry."
            )
        last_frames[cam_key] = frame
        frames[cam_key] = frame
    return frames


def _frame_signature(frame: np.ndarray) -> int:
    # Downsample before hashing to keep this inexpensive in the control loop.
    sample = frame[::16, ::16]
    return zlib.crc32(sample.tobytes())


def _read_camera_frames_with_health(
    cameras: dict[str, Any],
    last_frames: dict[str, np.ndarray],
    camera_max_age_ms: int,
    camera_retry_timeout_ms: int,
    stale_counts: dict[str, int],
    identical_counts: dict[str, int],
    last_signatures: dict[str, int],
    last_reconnect_ts: dict[str, float],
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
                frame = last_frames.get(cam_key)
                used_cached = frame is not None

        if frame is None:
            raise RuntimeError(
                f"Camera '{cam_key}' returned no frame. Check cable/power/device path, then retry."
            )

        if used_cached:
            stale_counts[cam_key] = stale_counts.get(cam_key, 0) + 1
        else:
            stale_counts[cam_key] = 0

        sig = _frame_signature(frame)
        if last_signatures.get(cam_key) == sig:
            identical_counts[cam_key] = identical_counts.get(cam_key, 0) + 1
        else:
            identical_counts[cam_key] = 0
        last_signatures[cam_key] = sig

        stale_trigger = camera_reconnect_stale_count > 0 and stale_counts[cam_key] >= camera_reconnect_stale_count
        freeze_trigger = (
            camera_freeze_identical_count > 0 and identical_counts[cam_key] >= camera_freeze_identical_count
        )
        should_reconnect = stale_trigger or freeze_trigger

        if should_reconnect:
            now = time.time()
            cooldown_ok = (now - last_reconnect_ts.get(cam_key, 0.0)) >= max(camera_reconnect_cooldown_s, 0.0)
            if cooldown_ok:
                if stale_trigger:
                    reconnect_reason = f"stale_count={stale_counts[cam_key]}"
                else:
                    reconnect_reason = f"identical_count={identical_counts[cam_key]}"
                print(f"[WARN] Camera '{cam_key}' seems stalled ({reconnect_reason}), reconnecting...")
                try:
                    cam.disconnect()
                except Exception:
                    pass
                try:
                    cam.connect(warmup=True)
                    fresh = cam.async_read(timeout_ms=max(camera_retry_timeout_ms, 1))
                    frame = fresh
                    reconnected = True
                    stale_counts[cam_key] = 0
                    identical_counts[cam_key] = 0
                    last_signatures[cam_key] = _frame_signature(frame)
                    print(f"[INFO] Camera '{cam_key}' reconnected.")
                except Exception as reconnect_exc:
                    print(f"[WARN] Camera '{cam_key}' reconnect failed: {reconnect_exc}")
                finally:
                    last_reconnect_ts[cam_key] = time.time()

            if camera_fail_on_freeze and (stale_trigger or freeze_trigger) and not reconnected:
                raise RuntimeError(
                    f"Camera '{cam_key}' stalled ({reconnect_reason or 'stale/freeze trigger'}), "
                    "and reconnect did not recover."
                )

        last_frames[cam_key] = frame
        frames[cam_key] = frame
        health[cam_key] = {
            "used_cached": float(1.0 if used_cached else 0.0),
            "stale_count": float(stale_counts.get(cam_key, 0)),
            "identical_count": float(identical_counts.get(cam_key, 0)),
            "reconnected": float(1.0 if reconnected else 0.0),
        }

    return frames, health


def _build_features(
    camera_configs: dict[str, OpenCVCameraConfig],
    joint_names: list[str],
    use_video: bool,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    obs_names: list[str] = []
    action_names: list[str] = []
    for side in ("left", "right"):
        for joint_name in joint_names:
            name = f"{side}_{joint_name}.pos"
            obs_names.append(name)
            action_names.append(name)

    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(obs_names),),
            "names": obs_names,
        },
    }
    for cam_name, cam_cfg in camera_configs.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video" if use_video else "image",
            "shape": (int(cam_cfg.height), int(cam_cfg.width), 3),
            "names": ["height", "width", "channels"],
        }

    return features, obs_names, action_names


def _create_or_resume_dataset(
    opts: DatasetOptions,
    features: dict[str, dict[str, Any]],
    num_cameras: int,
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
        robot_type="songling_aloha_raw_can",
        use_videos=opts.video,
        image_writer_processes=opts.num_image_writer_processes,
        image_writer_threads=opts.num_image_writer_threads_per_camera * num_cameras,
        batch_encoding_size=opts.video_encoding_batch_size,
        vcodec=opts.vcodec,
        streaming_encoding=opts.streaming_encoding,
        encoder_queue_maxsize=opts.encoder_queue_maxsize,
        encoder_threads=opts.encoder_threads,
    )


def _load_songling_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = draccus.load(dict[str, Any], f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} is not a valid mapping.")
    return cfg


def _resolve_dataset_options(raw_cfg: dict[str, Any], args: argparse.Namespace) -> DatasetOptions:
    raw_dataset = raw_cfg.get("dataset") if isinstance(raw_cfg.get("dataset"), dict) else {}
    top_level_fps = raw_cfg.get("fps")

    repo_id = args.dataset_repo_id if args.dataset_repo_id is not None else raw_dataset.get("repo_id")
    if not repo_id:
        raise ValueError(
            "Missing dataset repo id. Set --dataset.repo_id=... or add dataset.repo_id in config YAML."
        )

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
    play_sounds = _coerce_bool(
        args.play_sounds if args.play_sounds is not None else raw_cfg.get("play_sounds"),
        False,
    )
    voice_lang = str(
        args.voice_lang if args.voice_lang is not None else raw_cfg.get("voice_lang", "en")
    ).strip().lower()
    if voice_lang not in {"en", "zh"}:
        raise ValueError(f"Unsupported voice language: {voice_lang}. Use 'en' or 'zh'.")
    voice_rate_raw = args.voice_rate if args.voice_rate is not None else raw_cfg.get("voice_rate")
    if voice_rate_raw is None:
        voice_rate = -35 if voice_lang == "zh" else -10
    else:
        voice_rate = int(voice_rate_raw)
    voice_engine_raw = args.voice_engine if args.voice_engine is not None else raw_cfg.get("voice_engine", "auto")
    voice_engine = _coerce_voice_engine(voice_engine_raw)
    voice_piper_model_raw = (
        args.voice_piper_model if args.voice_piper_model is not None else raw_cfg.get("voice_piper_model")
    )
    voice_piper_model = str(voice_piper_model_raw).strip() if voice_piper_model_raw not in {None, ""} else None
    voice_piper_binary_raw = (
        args.voice_piper_binary if args.voice_piper_binary is not None else raw_cfg.get("voice_piper_binary")
    )
    voice_piper_binary = str(voice_piper_binary_raw).strip() if voice_piper_binary_raw not in {None, ""} else None
    voice_piper_speaker_raw = (
        args.voice_piper_speaker if args.voice_piper_speaker is not None else raw_cfg.get("voice_piper_speaker")
    )
    voice_piper_speaker = int(voice_piper_speaker_raw) if voice_piper_speaker_raw is not None else None

    root_value = args.dataset_root if args.dataset_root is not None else raw_dataset.get("root")
    if not root_value:
        root_value = str(DEFAULT_DATASET_ROOT)
        print(f"[INFO] dataset.root not set; defaulting to: {root_value}")
    root = _ensure_local_root_is_writable(
        root_value=root_value,
        resume=resume,
        auto_increment_root=auto_increment_root,
    )

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

    video = _coerce_bool(args.dataset_video if args.dataset_video is not None else raw_dataset.get("video"), True)
    push_to_hub = _coerce_bool(
        args.dataset_push_to_hub if args.dataset_push_to_hub is not None else raw_dataset.get("push_to_hub"),
        False,
    )
    private = _coerce_bool(
        args.dataset_private if args.dataset_private is not None else raw_dataset.get("private"),
        False,
    )
    tags = _parse_tags(args.dataset_tags if args.dataset_tags is not None else raw_dataset.get("tags"))

    num_image_writer_processes = _coerce_int(
        args.dataset_num_image_writer_processes
        if args.dataset_num_image_writer_processes is not None
        else raw_dataset.get("num_image_writer_processes"),
        0,
    )
    num_image_writer_threads_per_camera = _coerce_int(
        args.dataset_num_image_writer_threads_per_camera
        if args.dataset_num_image_writer_threads_per_camera is not None
        else raw_dataset.get("num_image_writer_threads_per_camera"),
        4,
    )
    video_encoding_batch_size = _coerce_int(
        args.dataset_video_encoding_batch_size
        if args.dataset_video_encoding_batch_size is not None
        else raw_dataset.get("video_encoding_batch_size"),
        1,
    )
    vcodec = str(args.dataset_vcodec if args.dataset_vcodec is not None else raw_dataset.get("vcodec", "libsvtav1"))
    streaming_encoding = _coerce_bool(
        args.dataset_streaming_encoding
        if args.dataset_streaming_encoding is not None
        else raw_dataset.get("streaming_encoding"),
        False,
    )
    encoder_queue_maxsize = _coerce_int(
        args.dataset_encoder_queue_maxsize
        if args.dataset_encoder_queue_maxsize is not None
        else raw_dataset.get("encoder_queue_maxsize"),
        30,
    )

    if args.dataset_encoder_threads is not None:
        encoder_threads = args.dataset_encoder_threads
    else:
        raw_encoder_threads = raw_dataset.get("encoder_threads")
        encoder_threads = int(raw_encoder_threads) if raw_encoder_threads is not None else None

    return DatasetOptions(
        repo_id=str(repo_id),
        single_task=str(single_task),
        root=root,
        fps=fps,
        episode_time_s=episode_time_s,
        reset_time_s=reset_time_s,
        num_episodes=num_episodes,
        video=video,
        push_to_hub=push_to_hub,
        private=private,
        tags=tags,
        num_image_writer_processes=num_image_writer_processes,
        num_image_writer_threads_per_camera=num_image_writer_threads_per_camera,
        video_encoding_batch_size=video_encoding_batch_size,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_maxsize,
        encoder_threads=encoder_threads,
        resume=resume,
        auto_increment_root=auto_increment_root,
        play_sounds=play_sounds,
        voice_lang=voice_lang,
        voice_rate=voice_rate,
        voice_engine=voice_engine,
        voice_piper_model=voice_piper_model,
        voice_piper_binary=voice_piper_binary,
        voice_piper_speaker=voice_piper_speaker,
    )


def _resolve_hardware(
    raw_cfg: dict[str, Any],
    args: argparse.Namespace,
    joint_names: list[str],
    fallback_fps: int,
) -> tuple[SideCANRuntime, SideCANRuntime, dict[str, OpenCVCameraConfig]]:
    robot = _expect_dict(raw_cfg.get("robot"), "robot")
    left_arm = _expect_dict(robot.get("left_arm_config"), "robot.left_arm_config")
    right_arm = _expect_dict(robot.get("right_arm_config"), "robot.right_arm_config")

    left_iface = args.left_interface if args.left_interface is not None else left_arm.get("port")
    right_iface = args.right_interface if args.right_interface is not None else right_arm.get("port")
    if not left_iface or not right_iface:
        raise ValueError("Both left/right CAN interface ports are required.")

    left_bitrate = _coerce_int(args.left_bitrate if args.left_bitrate is not None else left_arm.get("can_bitrate"), 1000000)
    right_bitrate = _coerce_int(
        args.right_bitrate if args.right_bitrate is not None else right_arm.get("can_bitrate"), 1000000
    )
    left_data_bitrate = (
        _coerce_int(args.left_data_bitrate, 5000000)
        if args.left_data_bitrate is not None
        else (int(left_arm["can_data_bitrate"]) if "can_data_bitrate" in left_arm else None)
    )
    right_data_bitrate = (
        _coerce_int(args.right_data_bitrate, 5000000)
        if args.right_data_bitrate is not None
        else (int(right_arm["can_data_bitrate"]) if "can_data_bitrate" in right_arm else None)
    )

    left_use_fd = _coerce_bool(
        args.left_use_fd if args.left_use_fd is not None else left_arm.get("use_can_fd"),
        False,
    )
    right_use_fd = _coerce_bool(
        args.right_use_fd if args.right_use_fd is not None else right_arm.get("use_can_fd"),
        False,
    )

    default_obs_ids = _parse_id_list(args.observation_ids)
    default_action_ids = _parse_id_list(args.action_ids)
    left_obs_ids = _parse_id_list(args.left_observation_ids) if args.left_observation_ids else list(default_obs_ids)
    right_obs_ids = _parse_id_list(args.right_observation_ids) if args.right_observation_ids else list(default_obs_ids)
    left_action_ids = _parse_id_list(args.left_action_ids) if args.left_action_ids else list(default_action_ids)
    right_action_ids = _parse_id_list(args.right_action_ids) if args.right_action_ids else list(default_action_ids)

    for ids, label in (
        (left_obs_ids, "left observation ids"),
        (right_obs_ids, "right observation ids"),
        (left_action_ids, "left action ids"),
        (right_action_ids, "right action ids"),
    ):
        if len(ids) == 0:
            raise ValueError(f"{label} must contain at least one CAN frame id.")

    left_cams = _expect_dict(left_arm.get("cameras"), "robot.left_arm_config.cameras")
    right_cams = _expect_dict(right_arm.get("cameras"), "robot.right_arm_config.cameras")
    left_high_raw = dict(_expect_dict(left_cams.get("high"), "robot.left_arm_config.cameras.high"))
    left_elbow_raw = dict(_expect_dict(left_cams.get("elbow"), "robot.left_arm_config.cameras.elbow"))
    right_elbow_raw = dict(_expect_dict(right_cams.get("elbow"), "robot.right_arm_config.cameras.elbow"))

    if args.left_high is not None:
        left_high_raw["index_or_path"] = _strip_wrapping_quotes(args.left_high)
    if args.left_elbow is not None:
        left_elbow_raw["index_or_path"] = _strip_wrapping_quotes(args.left_elbow)
    if args.right_elbow is not None:
        right_elbow_raw["index_or_path"] = _strip_wrapping_quotes(args.right_elbow)

    camera_configs = {
        "left_high": _camera_config_from_dict(left_high_raw, fallback_fps=fallback_fps),
        "left_elbow": _camera_config_from_dict(left_elbow_raw, fallback_fps=fallback_fps),
        "right_elbow": _camera_config_from_dict(right_elbow_raw, fallback_fps=fallback_fps),
    }

    left_can = SideCANRuntime(
        interface=str(left_iface),
        bitrate=left_bitrate,
        data_bitrate=left_data_bitrate,
        use_fd=left_use_fd,
        observation_ids=left_obs_ids,
        action_ids=left_action_ids,
    )
    right_can = SideCANRuntime(
        interface=str(right_iface),
        bitrate=right_bitrate,
        data_bitrate=right_data_bitrate,
        use_fd=right_use_fd,
        observation_ids=right_obs_ids,
        action_ids=right_action_ids,
    )
    return left_can, right_can, camera_configs


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
            "Please set both display host and port together "
            "(--display_ip/--display_port, or provide both in YAML)."
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


def _print_seen_ids(prefix: str, seen_ids: set[int]) -> None:
    if not seen_ids:
        print(f"{prefix}: no CAN frames observed.")
        return
    sorted_ids = sorted(seen_ids)
    preview = ", ".join(f"0x{i:03X}" for i in sorted_ids[:24])
    extra = "" if len(sorted_ids) <= 24 else f", ... (+{len(sorted_ids) - 24} more)"
    print(f"{prefix}: {preview}{extra}")


def main() -> None:
    _patch_multiprocess_resource_tracker()

    parser = argparse.ArgumentParser(
        description="Record Songling integrated chain with raw CAN parsing into LeRobot dataset format."
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        default=Path("examples/songling_aloha/teleop.yaml"),
        help="Path to Songling YAML config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and exit.")

    parser.add_argument(
        "--joint-names",
        default=",".join(DEFAULT_JOINT_NAMES),
        help="Comma-separated joint names per side, e.g. joint_1,...,gripper.",
    )
    parser.add_argument(
        "--observation-ids",
        default=DEFAULT_OBSERVATION_IDS,
        help="Default CAN frame ids used to decode Piper-style observation joints (comma-separated, hex or decimal).",
    )
    parser.add_argument(
        "--action-ids",
        default=DEFAULT_ACTION_IDS,
        help="Default CAN frame ids used to decode Piper-style action targets (comma-separated, hex or decimal).",
    )
    parser.add_argument("--left-observation-ids", default=None, help="Override left side observation frame ids.")
    parser.add_argument("--right-observation-ids", default=None, help="Override right side observation frame ids.")
    parser.add_argument("--left-action-ids", default=None, help="Override left side action frame ids.")
    parser.add_argument("--right-action-ids", default=None, help="Override right side action frame ids.")

    parser.add_argument("--decode-byte-offset", type=int, default=0, help="CAN payload byte offset for decoding.")
    parser.add_argument(
        "--decode-byte-length",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Number of bytes to decode into one signal value.",
    )
    parser.add_argument(
        "--decode-signed",
        type=_parse_cli_bool,
        default=True,
        help="Whether decoded integer is signed (true/false).",
    )
    parser.add_argument(
        "--decode-endian",
        choices=["little", "big"],
        default="little",
        help="Endianness for integer decoding.",
    )
    parser.add_argument(
        "--decode-scale",
        type=float,
        default=1e-4,
        help="Scale applied after integer decode: value = raw * scale + bias.",
    )
    parser.add_argument("--decode-bias", type=float, default=0.0, help="Bias applied after scaling.")

    parser.add_argument("--max-can-poll-msgs", type=int, default=128, help="Max CAN frames consumed per loop per side.")
    parser.add_argument("--camera-max-age-ms", type=int, default=500, help="Max age for camera read_latest().")
    parser.add_argument(
        "--camera-retry-timeout-ms",
        type=int,
        default=60,
        help="Retry timeout for camera async_read when read_latest is stale.",
    )
    parser.add_argument(
        "--camera-reconnect-stale-count",
        type=int,
        default=12,
        help="Reconnect camera after this many consecutive stale/cached frames. Set <=0 to disable.",
    )
    parser.add_argument(
        "--camera-freeze-identical-count",
        type=int,
        default=120,
        help="Reconnect camera after this many consecutive identical frames. Set <=0 to disable.",
    )
    parser.add_argument(
        "--camera-reconnect-cooldown-s",
        type=float,
        default=2.0,
        help="Minimum seconds between reconnect attempts for the same camera.",
    )
    parser.add_argument(
        "--camera-fail-on-freeze",
        type=_parse_cli_bool,
        default=False,
        help="If true, abort recording when camera reconnect cannot recover from freeze.",
    )
    parser.add_argument(
        "--display_data",
        "--display-data",
        dest="display_data",
        type=_parse_cli_bool,
        default=None,
        help="Enable live Rerun visualization while recording.",
    )
    parser.add_argument(
        "--display_ip",
        "--display-ip",
        dest="display_ip",
        default=None,
        help="Rerun server IP. If omitted, local viewer spawn is used.",
    )
    parser.add_argument(
        "--display_port",
        "--display-port",
        dest="display_port",
        type=int,
        default=None,
        help="Rerun server port (used with --display_ip).",
    )
    parser.add_argument(
        "--display_compressed_images",
        "--display-compressed-images",
        dest="display_compressed_images",
        type=_parse_cli_bool,
        default=None,
        help="Compress images before logging to Rerun.",
    )

    parser.add_argument("--left-interface", "--robot.left_arm_config.port", dest="left_interface", default=None)
    parser.add_argument("--right-interface", "--robot.right_arm_config.port", dest="right_interface", default=None)
    parser.add_argument(
        "--left-bitrate", "--robot.left_arm_config.can_bitrate", dest="left_bitrate", type=int, default=None
    )
    parser.add_argument(
        "--right-bitrate", "--robot.right_arm_config.can_bitrate", dest="right_bitrate", type=int, default=None
    )
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
    parser.add_argument(
        "--left-use-fd",
        "--robot.left_arm_config.use_can_fd",
        dest="left_use_fd",
        type=_parse_cli_bool,
        default=None,
    )
    parser.add_argument(
        "--right-use-fd",
        "--robot.right_arm_config.use_can_fd",
        dest="right_use_fd",
        type=_parse_cli_bool,
        default=None,
    )

    parser.add_argument(
        "--left-high",
        "--robot.left_arm_config.cameras.high.index_or_path",
        dest="left_high",
        default=None,
        help="Override camera device for left_high.",
    )
    parser.add_argument(
        "--left-elbow",
        "--robot.left_arm_config.cameras.elbow.index_or_path",
        dest="left_elbow",
        default=None,
        help="Override camera device for left_elbow.",
    )
    parser.add_argument(
        "--right-elbow",
        "--robot.right_arm_config.cameras.elbow.index_or_path",
        dest="right_elbow",
        default=None,
        help="Override camera device for right_elbow.",
    )

    parser.add_argument("--dataset.repo_id", "--dataset-repo-id", dest="dataset_repo_id", default=None)
    parser.add_argument("--dataset.single_task", "--dataset-single-task", dest="dataset_single_task", default=None)
    parser.add_argument(
        "--play_sounds",
        "--play-sounds",
        dest="play_sounds",
        type=_parse_cli_bool,
        default=None,
        help="Enable voice prompts for recording/reset/save events.",
    )
    parser.add_argument(
        "--voice-lang",
        "--voice_lang",
        dest="voice_lang",
        choices=["en", "zh"],
        default=None,
        help="Language for voice prompts and event text.",
    )
    parser.add_argument(
        "--voice-rate",
        "--voice_rate",
        dest="voice_rate",
        type=int,
        default=None,
        help="Speech rate for voice prompts (-100..100). Lower is slower and clearer.",
    )
    parser.add_argument(
        "--voice-engine",
        "--voice_engine",
        dest="voice_engine",
        choices=list(VOICE_ENGINE_CHOICES),
        default=None,
        help="Voice backend: auto/piper/spd-say/espeak-ng/espeak/tone/log-say/bell/off.",
    )
    parser.add_argument(
        "--voice-piper-model",
        "--voice_piper_model",
        dest="voice_piper_model",
        default=None,
        help="Path to Piper .onnx voice model (required when using --voice-engine=piper if auto-discovery fails).",
    )
    parser.add_argument(
        "--voice-piper-binary",
        "--voice_piper_binary",
        dest="voice_piper_binary",
        default=None,
        help="Piper executable path or command name (default: piper in PATH).",
    )
    parser.add_argument(
        "--voice-piper-speaker",
        "--voice_piper_speaker",
        dest="voice_piper_speaker",
        type=int,
        default=None,
        help="Optional Piper speaker id for multi-speaker models.",
    )
    parser.add_argument(
        "--dataset.root",
        "--dataset-root",
        dest="dataset_root",
        default=None,
        help=f"Dataset local root path. Defaults to {DEFAULT_DATASET_ROOT} when omitted.",
    )
    parser.add_argument("--dataset.fps", "--dataset-fps", dest="dataset_fps", type=int, default=None)
    parser.add_argument(
        "--dataset.episode_time_s",
        "--dataset-episode-time-s",
        dest="dataset_episode_time_s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--dataset.reset_time_s",
        "--dataset-reset-time-s",
        dest="dataset_reset_time_s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--dataset.num_episodes",
        "--dataset-num-episodes",
        dest="dataset_num_episodes",
        type=int,
        default=None,
    )
    parser.add_argument("--dataset.video", "--dataset-video", dest="dataset_video", type=_parse_cli_bool, default=None)
    parser.add_argument(
        "--dataset.push_to_hub",
        "--dataset-push-to-hub",
        dest="dataset_push_to_hub",
        type=_parse_cli_bool,
        default=None,
    )
    parser.add_argument(
        "--dataset.private",
        "--dataset-private",
        dest="dataset_private",
        type=_parse_cli_bool,
        default=None,
    )
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
    parser.add_argument(
        "--resume",
        type=_parse_cli_bool,
        default=None,
        help="Append to existing dataset root (true/false).",
    )
    parser.add_argument(
        "--dataset.auto_increment_root",
        "--dataset-auto-increment-root",
        dest="dataset_auto_increment_root",
        type=_parse_cli_bool,
        default=None,
        help="When root exists and --resume=false, auto-create next numbered run directory.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        quoted = " ".join(shlex.quote(item) for item in unknown)
        print(f"[WARN] Ignoring unsupported overrides: {quoted}")

    raw_cfg = _load_songling_yaml(args.config_path)
    dataset_opts = _resolve_dataset_options(raw_cfg, args)
    display_opts = _resolve_display_options(raw_cfg, args)

    joint_names = _parse_csv(args.joint_names)
    if not joint_names:
        raise ValueError("No joint names resolved. Please set --joint-names.")

    left_can, right_can, camera_configs = _resolve_hardware(
        raw_cfg=raw_cfg,
        args=args,
        joint_names=joint_names,
        fallback_fps=dataset_opts.fps,
    )

    decode_opts = CANDecodeOptions(
        byte_offset=args.decode_byte_offset,
        byte_length=args.decode_byte_length,
        signed=args.decode_signed,
        endian=args.decode_endian,
        scale=args.decode_scale,
        bias=args.decode_bias,
    )
    features, _, _ = _build_features(
        camera_configs=camera_configs,
        joint_names=joint_names,
        use_video=dataset_opts.video,
    )

    print("=" * 50)
    print("Songling Raw CAN Dataset Record")
    print("=" * 50)
    print(f"Config: {args.config_path}")
    print(f"Dataset repo_id: {dataset_opts.repo_id}")
    print(f"Dataset root: {dataset_opts.root}")
    print(f"FPS: {dataset_opts.fps}")
    print(f"Episodes: {dataset_opts.num_episodes} (episode_time_s={dataset_opts.episode_time_s})")
    print(f"Reset time: {dataset_opts.reset_time_s}s")
    print(
        f"Voice prompts: {'on' if dataset_opts.play_sounds else 'off'} "
        f"(lang={dataset_opts.voice_lang}, rate={dataset_opts.voice_rate}, engine={dataset_opts.voice_engine})"
    )
    if dataset_opts.voice_engine == "piper":
        print(
            "Piper config: "
            f"model={dataset_opts.voice_piper_model or 'auto-discovery'}, "
            f"binary={dataset_opts.voice_piper_binary or 'piper'}, "
            f"speaker={dataset_opts.voice_piper_speaker if dataset_opts.voice_piper_speaker is not None else 'default'}"
        )
    print(f"Cameras: {', '.join(CAMERA_KEYS)}")
    print(
        f"Left CAN: {left_can.interface} ({'FD' if left_can.use_fd else 'CAN2.0'}, bitrate={left_can.bitrate})"
    )
    print(
        f"Right CAN: {right_can.interface} ({'FD' if right_can.use_fd else 'CAN2.0'}, bitrate={right_can.bitrate})"
    )
    print(
        "Display: "
        + (
            f"on ({display_opts.display_ip}:{display_opts.display_port})"
            if (display_opts.display_data and display_opts.display_ip and display_opts.display_port)
            else ("on (local viewer)" if display_opts.display_data else "off")
        )
    )
    print(
        "Camera recovery: "
        f"stale_count>={args.camera_reconnect_stale_count}, "
        f"identical_count>={args.camera_freeze_identical_count}, "
        f"cooldown={args.camera_reconnect_cooldown_s}s, "
        f"fail_on_freeze={args.camera_fail_on_freeze}"
    )
    print()

    if args.dry_run:
        print("Resolved settings successfully (--dry-run).")
        return

    try:
        import can  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("python-can is required. Install with `pip install python-can`.") from exc

    cameras: dict[str, Any] = {}
    left_bus = None
    right_bus = None
    dataset: LeRobotDataset | None = None
    interrupted = False
    saved_episodes = 0
    frames_total = 0
    rr = None
    listener = None
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    left_state = CANBusState()
    right_state = CANBusState()

    left_obs_last = np.zeros((len(joint_names),), dtype=np.float32)
    right_obs_last = np.zeros((len(joint_names),), dtype=np.float32)
    left_action_last = np.zeros((len(joint_names),), dtype=np.float32)
    right_action_last = np.zeros((len(joint_names),), dtype=np.float32)
    left_warned_action_fallback = False
    right_warned_action_fallback = False
    episode_action_source_summaries: list[dict[str, Any]] = []
    current_episode_summary: dict[str, Any] | None = None
    last_camera_frames: dict[str, np.ndarray] = {}
    camera_stale_counts: dict[str, int] = {}
    camera_identical_counts: dict[str, int] = {}
    camera_last_signatures: dict[str, int] = {}
    camera_last_reconnect_ts: dict[str, float] = {}

    try:
        if display_opts.display_data:
            try:
                import rerun as _rr  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "rerun-sdk is required when --display_data=true. Install with `pip install rerun-sdk`."
                ) from exc
            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(
                session_name="songling_raw_can_record",
                ip=display_opts.display_ip,
                port=display_opts.display_port,
            )
            rr = _rr

        _validate_camera_devices(camera_configs)
        cameras = make_cameras_from_configs(camera_configs)
        for cam_name, cam in cameras.items():
            cam.connect(warmup=True)
            print(f"[INFO] Connected camera: {cam_name}")

        left_bus = _open_raw_can_bus(
            interface=left_can.interface,
            bitrate=left_can.bitrate,
            use_fd=left_can.use_fd,
            data_bitrate=left_can.data_bitrate,
        )
        right_bus = _open_raw_can_bus(
            interface=right_can.interface,
            bitrate=right_can.bitrate,
            use_fd=right_can.use_fd,
            data_bitrate=right_can.data_bitrate,
        )

        dataset = _create_or_resume_dataset(
            opts=dataset_opts,
            features=features,
            num_cameras=len(cameras),
        )

        listener, events = init_keyboard_listener()
        if listener is not None:
            print(
                "[INFO] Keyboard controls: "
                "Right Arrow=quick save current episode, "
                "Left Arrow=discard and re-record current episode, "
                "Esc=stop recording."
            )

        _say_event("start_recording", dataset_opts)
        with VideoEncodingManager(dataset):
            while saved_episodes < dataset_opts.num_episodes and not events["stop_recording"]:
                episode_idx = dataset.num_episodes
                current_episode_summary = {
                    "episode_index": int(episode_idx),
                    "frames": 0,
                    "left_control_frames": 0,
                    "right_control_frames": 0,
                }
                _say_event("recording_episode", dataset_opts, episode_idx=episode_idx)
                print(f"\n[INFO] Recording episode {episode_idx} ...")
                ep_start = time.perf_counter()
                ep_frames = 0
                last_log = time.time()
                quick_save_requested = False

                while (time.perf_counter() - ep_start) < dataset_opts.episode_time_s:
                    if events["exit_early"]:
                        quick_save_requested = not events["rerecord_episode"] and not events["stop_recording"]
                        events["exit_early"] = False
                        if quick_save_requested:
                            print("[INFO] Quick-save requested (Right Arrow).")
                        break

                    loop_start = time.perf_counter()

                    _poll_can_state(left_bus, left_state, decode_opts, max_msgs=args.max_can_poll_msgs)
                    _poll_can_state(right_bus, right_state, decode_opts, max_msgs=args.max_can_poll_msgs)

                    camera_frames, camera_health = _read_camera_frames_with_health(
                        cameras=cameras,
                        last_frames=last_camera_frames,
                        camera_max_age_ms=args.camera_max_age_ms,
                        camera_retry_timeout_ms=args.camera_retry_timeout_ms,
                        stale_counts=camera_stale_counts,
                        identical_counts=camera_identical_counts,
                        last_signatures=camera_last_signatures,
                        last_reconnect_ts=camera_last_reconnect_ts,
                        camera_reconnect_stale_count=args.camera_reconnect_stale_count,
                        camera_freeze_identical_count=args.camera_freeze_identical_count,
                        camera_reconnect_cooldown_s=args.camera_reconnect_cooldown_s,
                        camera_fail_on_freeze=args.camera_fail_on_freeze,
                    )

                    left_obs = _decode_piper_observation_vector(left_can.observation_ids, left_state, left_obs_last)
                    right_obs = _decode_piper_observation_vector(
                        right_can.observation_ids, right_state, right_obs_last
                    )
                    left_action, left_has_control = _decode_piper_action_vector(
                        left_can.action_ids,
                        left_state,
                        left_action_last,
                        fallback_observation=left_obs,
                    )
                    right_action, right_has_control = _decode_piper_action_vector(
                        right_can.action_ids,
                        right_state,
                        right_action_last,
                        fallback_observation=right_obs,
                    )
                    if not left_has_control and not left_warned_action_fallback:
                        print(
                            "[WARN] Left side control frames were not observed. "
                            "Using observation feedback as dataset action."
                        )
                        left_warned_action_fallback = True
                    if not right_has_control and not right_warned_action_fallback:
                        print(
                            "[WARN] Right side control frames were not observed. "
                            "Using observation feedback as dataset action."
                        )
                        right_warned_action_fallback = True
                    left_hz = _rx_hz(left_state.recent_timestamps)
                    right_hz = _rx_hz(right_state.recent_timestamps)

                    frame: dict[str, Any] = {
                        "observation.state": np.concatenate((left_obs, right_obs), axis=0).astype(
                            np.float32, copy=False
                        ),
                        "action": np.concatenate((left_action, right_action), axis=0).astype(np.float32, copy=False),
                        "task": dataset_opts.single_task,
                    }
                    for cam_key, img in camera_frames.items():
                        frame[f"observation.images.{cam_key}"] = img

                    dataset.add_frame(frame)
                    ep_frames += 1
                    frames_total += 1
                    if current_episode_summary is not None:
                        current_episode_summary["frames"] = int(current_episode_summary["frames"]) + 1
                        current_episode_summary["left_control_frames"] = int(
                            current_episode_summary["left_control_frames"]
                        ) + int(left_has_control)
                        current_episode_summary["right_control_frames"] = int(
                            current_episode_summary["right_control_frames"]
                        ) + int(right_has_control)

                    if display_opts.display_data and rr is not None:
                        rr.set_time("frame", sequence=frames_total)
                        for cam_key, img in camera_frames.items():
                            entity = rr.Image(img).compress() if display_opts.display_compressed_images else rr.Image(img)
                            rr.log(f"observation.images.{cam_key}", entity=entity)
                            rr.log(f"cameras.{cam_key}.stale_count", rr.Scalars(camera_health[cam_key]["stale_count"]))
                            rr.log(
                                f"cameras.{cam_key}.identical_count",
                                rr.Scalars(camera_health[cam_key]["identical_count"]),
                            )
                            rr.log(f"cameras.{cam_key}.used_cached", rr.Scalars(camera_health[cam_key]["used_cached"]))
                            rr.log(f"cameras.{cam_key}.reconnected", rr.Scalars(camera_health[cam_key]["reconnected"]))
                        for idx, joint_name in enumerate(joint_names):
                            rr.log(f"observation.state.left_{joint_name}.pos", rr.Scalars(float(left_obs[idx])))
                            rr.log(f"observation.state.right_{joint_name}.pos", rr.Scalars(float(right_obs[idx])))
                            rr.log(f"action.left_{joint_name}.pos", rr.Scalars(float(left_action[idx])))
                            rr.log(f"action.right_{joint_name}.pos", rr.Scalars(float(right_action[idx])))
                        rr.log("can.left.rx_hz", rr.Scalars(float(left_hz)))
                        rr.log("can.right.rx_hz", rr.Scalars(float(right_hz)))
                        rr.log("can.left.total_msgs", rr.Scalars(float(left_state.total_msgs)))
                        rr.log("can.right.total_msgs", rr.Scalars(float(right_state.total_msgs)))

                    now = time.time()
                    if now - last_log >= 1.0:
                        print(
                            f"[INFO] ep={episode_idx} frame={ep_frames} "
                            f"left_rx_hz={left_hz:.1f} right_rx_hz={right_hz:.1f}"
                        )
                        last_log = now

                    dt = time.perf_counter() - loop_start
                    precise_sleep(max((1.0 / dataset_opts.fps) - dt, 0.0))

                if events["rerecord_episode"]:
                    _say_event("rerecord_episode", dataset_opts)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
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
                    episode_action_source_summaries.append(_finalize_action_source_summary(current_episode_summary))
                    current_episode_summary = None
                _say_event("saved_episode", dataset_opts, episode_idx=episode_idx)
                print(f"[INFO] Saved episode {episode_idx} ({ep_frames} frames).")

                if not events["stop_recording"] and saved_episodes < dataset_opts.num_episodes and dataset_opts.reset_time_s > 0:
                    _say_event("reset_environment", dataset_opts)
                    print(f"[INFO] Reset window: {dataset_opts.reset_time_s:.1f}s (press Right Arrow to skip)")
                    reset_end = time.perf_counter() + dataset_opts.reset_time_s
                    while time.perf_counter() < reset_end:
                        if events["exit_early"]:
                            events["exit_early"] = False
                            break
                        _poll_can_state(left_bus, left_state, decode_opts, max_msgs=args.max_can_poll_msgs)
                        _poll_can_state(right_bus, right_state, decode_opts, max_msgs=args.max_can_poll_msgs)
                        precise_sleep(0.01)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Received Ctrl+C, stopping recording loop.")
        if dataset is not None and dataset.episode_buffer is not None and dataset.episode_buffer.get("size", 0) > 0:
            dataset.save_episode()
            saved_episodes += 1
            if current_episode_summary is not None:
                episode_action_source_summaries.append(_finalize_action_source_summary(current_episode_summary))
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
                left_can=left_can,
                right_can=right_can,
                episode_summaries=episode_action_source_summaries,
                left_seen_ids=left_state.seen_ids,
                right_seen_ids=right_state.seen_ids,
            )
            print(f"[INFO] Wrote action-source summary: {summary_path}")
            if dataset_opts.push_to_hub:
                dataset.push_to_hub(tags=dataset_opts.tags, private=dataset_opts.private)
                print("[INFO] Dataset pushed to Hugging Face Hub.")

        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        if left_bus is not None:
            try:
                left_bus.shutdown()
            except Exception:
                pass
        if right_bus is not None:
            try:
                right_bus.shutdown()
            except Exception:
                pass
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        print(f"Episodes saved: {saved_episodes}")
        print(f"Frames written: {frames_total}")
        print(f"Dataset root: {dataset_opts.root}")
        _print_seen_ids("Left seen IDs", left_state.seen_ids)
        _print_seen_ids("Right seen IDs", right_state.seen_ids)
        if interrupted:
            print("[INFO] Recording stopped by user.")


if __name__ == "__main__":
    main()
