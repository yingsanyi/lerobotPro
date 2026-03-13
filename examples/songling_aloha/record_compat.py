#!/usr/bin/env python

"""Songling dataset recording launcher compatible with `lerobot-record` CLI overrides.

This script forwards all unknown arguments to `lerobot-record` unchanged, while
injecting a default `--config_path` pointing to the Songling unified YAML when
the user did not provide one explicitly.

Examples:
    python examples/songling_aloha/record_compat.py \
      --dataset.repo_id=your_hf_username/songling_aloha_demo \
      --dataset.single_task="Bimanual teleoperation with Songling ALOHA profile"

    python examples/songling_aloha/record_compat.py \
      --config_path examples/songling_aloha/teleop.yaml \
      --dataset.repo_id=your_hf_username/songling_aloha_demo
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("examples/songling_aloha/teleop.yaml")
RECORD_TOP_LEVEL_KEYS = {
    "robot",
    "teleop",
    "policy",
    "dataset",
    "display_data",
    "display_ip",
    "display_port",
    "display_compressed_images",
    "play_sounds",
    "resume",
}
DATASET_KEYS = {
    "repo_id",
    "single_task",
    "root",
    "fps",
    "episode_time_s",
    "reset_time_s",
    "num_episodes",
    "video",
    "push_to_hub",
    "private",
    "tags",
    "num_image_writer_processes",
    "num_image_writer_threads_per_camera",
    "video_encoding_batch_size",
    "vcodec",
    "streaming_encoding",
    "encoder_queue_maxsize",
    "encoder_threads",
    "rename_map",
}


def _has_dataset_overrides(args: list[str]) -> bool:
    return any(arg.startswith("--dataset.") for arg in args)


def _get_last_flag_value(args: list[str], names: tuple[str, ...]) -> str | None:
    value: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        matched_name = next((name for name in names if arg == name or arg.startswith(f"{name}=")), None)
        if matched_name is None:
            i += 1
            continue
        if arg == matched_name:
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                raise ValueError(f"Missing value after {matched_name}.")
            continue
        value = arg.split("=", 1)[1]
        i += 1
    return value


def _parse_cli_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}. Use true/false.")


def _has_hf_auth() -> bool:
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return True
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        try:
            return bool(token_file.read_text(encoding="utf-8").strip())
        except Exception:
            return False
    return False


def _ensure_local_root_is_writable(
    root_value: str,
    resume: bool,
) -> Path:
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
            "Please choose a writable local path, e.g. "
            "--dataset.root=/home/aiteam/wyl/lerobot0.5.0/lerobot/outputs/songling_datasets/run_001"
        ) from None

    if root_path.exists() and not resume:
        raise FileExistsError(
            f"Dataset root already exists: {root_path}\n"
            "Use a new path, or set --resume=true to append to an existing dataset."
        )
    return root_path


def _extract_last_flag_value_and_strip(
    args: list[str],
    names: tuple[str, ...],
) -> tuple[str | None, list[str]]:
    value: str | None = None
    stripped: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        matched_name = next((name for name in names if arg == name or arg.startswith(f"{name}=")), None)
        if matched_name is None:
            stripped.append(arg)
            i += 1
            continue

        if arg == matched_name:
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                # Keep argparse-like behavior: a flag without value is invalid for this option.
                raise ValueError(f"Missing value after {matched_name}.")
            continue

        value = arg.split("=", 1)[1]
        i += 1
    return value, stripped


def _load_yaml_as_dict(config_path: Path) -> dict[str, Any]:
    try:
        import draccus  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "draccus is required to parse Songling YAML defaults. "
            "Please run in your LeRobot environment."
        ) from e

    with config_path.open("r", encoding="utf-8") as f:
        cfg = draccus.load(dict[str, Any], f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} is not a valid mapping.")
    return cfg


def _sanitize_to_record_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in RECORD_TOP_LEVEL_KEYS:
        if key in raw_cfg:
            out[key] = raw_cfg[key]

    raw_dataset = raw_cfg.get("dataset")
    dataset = raw_dataset if isinstance(raw_dataset, dict) else {}
    dataset = {k: v for k, v in dataset.items() if k in DATASET_KEYS}

    # Songling unified yaml may store top-level fps (used by teleoperate/visualize).
    # For lerobot-record, dataset.fps is the canonical field.
    top_level_fps = raw_cfg.get("fps")
    if "fps" not in dataset and isinstance(top_level_fps, int):
        dataset["fps"] = top_level_fps

    out["dataset"] = dataset
    return out


def _detect_integrated_songling_openarm_cfg(raw_cfg: dict[str, Any]) -> bool:
    robot = raw_cfg.get("robot")
    teleop = raw_cfg.get("teleop")
    if not isinstance(robot, dict) or not isinstance(teleop, dict):
        return False

    robot_type = robot.get("type")
    teleop_type = teleop.get("type")
    if robot_type != "bi_openarm_follower" or teleop_type != "bi_openarm_leader":
        return False

    robot_left = robot.get("left_arm_config")
    robot_right = robot.get("right_arm_config")
    teleop_left = teleop.get("left_arm_config")
    teleop_right = teleop.get("right_arm_config")
    if not all(isinstance(v, dict) for v in (robot_left, robot_right, teleop_left, teleop_right)):
        return False

    same_left = teleop_left.get("port") == robot_left.get("port")
    same_right = teleop_right.get("port") == robot_right.get("port")
    return bool(same_left and same_right)


def _write_temp_yaml(config: dict[str, Any]) -> Path:
    try:
        import draccus  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "draccus is required to write temporary Songling record config."
        ) from e

    fd, path_str = tempfile.mkstemp(prefix="songling_record_", suffix=".yaml")
    os.close(fd)
    path = Path(path_str)
    with path.open("w", encoding="utf-8") as f:
        draccus.dump(config, f, indent=2)
    return path


def _validate_dataset_defaults(
    effective_cfg: dict[str, Any],
    passthrough: list[str],
) -> None:
    if _has_dataset_overrides(passthrough):
        return
    dataset = effective_cfg.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(
            "No --dataset.* overrides were provided and YAML has no dataset section.\n"
            "Please pass required dataset fields (e.g. --dataset.repo_id and --dataset.single_task), "
            "or add a dataset section into the YAML."
        )
    missing = [k for k in ("repo_id", "single_task") if not dataset.get(k)]
    if missing:
        raise ValueError(
            "No --dataset.* overrides were provided and YAML misses required dataset field(s): "
            f"{', '.join(missing)}."
        )


def _forward_to_songling_raw_record(
    source_config: Path,
    passthrough: list[str],
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "examples/songling_aloha/record_raw_can_dataset.py",
        f"--config_path={source_config}",
        *passthrough,
    ]
    print(
        "Detected Songling integrated chain config; forwarding to raw CAN record path "
        "(3 cameras + raw CAN parser, no Damiao handshake)."
    )
    print("Forwarding command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-compatible Songling launcher for lerobot-record with YAML fallback defaults."
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Default config path used when no --config_path override is provided in passthrough args.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the forwarded lerobot-record command without executing it.",
    )
    parser.add_argument(
        "--allow-integrated-openarm-mode",
        action="store_true",
        help="Bypass Songling integrated-chain safety check and forward to lerobot-record anyway.",
    )
    args, passthrough = parser.parse_known_args()

    config_override, passthrough = _extract_last_flag_value_and_strip(
        passthrough, ("--config_path", "--config-path")
    )
    source_config = Path(config_override) if config_override is not None else args.config_path
    if not source_config.exists():
        raise FileNotFoundError(f"Config file not found: {source_config}")

    raw_cfg = _load_yaml_as_dict(source_config)
    if _detect_integrated_songling_openarm_cfg(raw_cfg) and not args.allow_integrated_openarm_mode:
        _forward_to_songling_raw_record(
            source_config=source_config,
            passthrough=passthrough,
            dry_run=args.dry_run,
        )
        return

    record_cfg = _sanitize_to_record_config(raw_cfg)
    _validate_dataset_defaults(record_cfg, passthrough)

    # Enforce explicit local save path so data location is always intentional.
    root_override, passthrough = _extract_last_flag_value_and_strip(passthrough, ("--dataset.root",))
    if root_override is None or not root_override.strip():
        raise ValueError(
            "Please explicitly set local output path with --dataset.root=... "
            "(e.g. --dataset.root=/data/songling_aloha_runs/exp001)."
        )
    resume_override = _get_last_flag_value(passthrough, ("--resume",))
    if resume_override is None:
        resume = bool(record_cfg.get("resume", False))
    else:
        resume = _parse_cli_bool(resume_override)
    root_path = _ensure_local_root_is_writable(root_value=root_override, resume=resume)
    passthrough.append(f"--dataset.root={root_path}")

    # If HF auth is not configured, force local-only recording.
    push_to_hub_override, passthrough = _extract_last_flag_value_and_strip(
        passthrough, ("--dataset.push_to_hub",)
    )
    if push_to_hub_override is None:
        requested_push = bool(record_cfg.get("dataset", {}).get("push_to_hub", False))
        push_override_set = False
    else:
        requested_push = _parse_cli_bool(push_to_hub_override)
        push_override_set = True

    if requested_push and not _has_hf_auth():
        print("[INFO] Hugging Face auth not detected. Forcing --dataset.push_to_hub=false for local-only save.")
        passthrough.append("--dataset.push_to_hub=false")
    elif push_override_set:
        passthrough.append(f"--dataset.push_to_hub={'true' if requested_push else 'false'}")

    temp_config = _write_temp_yaml(record_cfg)
    passthrough = [f"--config_path={temp_config}", *passthrough]

    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_record", *passthrough]
    print("Forwarding command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        try:
            temp_config.unlink(missing_ok=True)
        except Exception:
            pass
        return

    try:
        raise SystemExit(subprocess.call(cmd))
    finally:
        try:
            temp_config.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
