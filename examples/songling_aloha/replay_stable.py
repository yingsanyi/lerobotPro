#!/usr/bin/env python3

"""Stable launcher for Songling ALOHA dataset replay.

This wrapper keeps the real replay logic in `replay_episodes.py` and only
adds:

- safer defaults for live robot replay
- a dataset default that points at `outputs/songling_aloha/local_debug_run`
- automatic fallback to `conda run -n <env> python` when the current Python
  does not have the replay dependencies installed
- passthrough of extra arguments to the underlying replay script
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = Path(__file__).resolve().with_name("replay_episodes.py")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "examples" / "songling_aloha" / "teleop.yaml"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "songling_aloha" / "local_debug_run"
DEFAULT_DATASET_REPO_ID = "local/songling_debug_run"
DEFAULT_CONDA_ENV = "lerobot_v050"
REQUIRED_MODULES = ("numpy", "torch", "draccus", "lerobot")


def _existing_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _guess_dataset_repo_id(dataset_root: Path) -> str:
    summary_path = dataset_root / "meta" / "songling_recording_summary.json"
    info_path = dataset_root / "meta" / "info.json"

    for candidate in (summary_path, info_path):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        repo_id = payload.get("repo_id")
        if isinstance(repo_id, str) and repo_id.strip():
            return repo_id.strip()

    return DEFAULT_DATASET_REPO_ID


def _missing_modules() -> list[str]:
    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def _resolve_python_runner(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.python is not None:
        return [args.python], f"explicit interpreter: {args.python}"

    missing = _missing_modules()
    if not missing:
        return [sys.executable], f"current interpreter: {sys.executable}"

    conda = shutil.which("conda")
    if conda and args.conda_env:
        return (
            [conda, "run", "--no-capture-output", "-n", args.conda_env, "python"],
            f"conda env '{args.conda_env}' (current interpreter missing: {', '.join(missing)})",
        )

    modules = ", ".join(missing)
    raise SystemExit(
        "Cannot start replay because the current Python is missing required modules "
        f"({modules}), and no usable conda fallback was found.\n"
        "Try one of the following:\n"
        f"  1. conda activate {DEFAULT_CONDA_ENV}\n"
        f"  2. python {REPLAY_SCRIPT} --dataset.root {DEFAULT_DATASET_ROOT}\n"
        f"  3. python {Path(__file__).resolve()} --conda-env <your_env_name>"
    )


def _build_replay_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    dataset_root = _existing_path(args.dataset_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    dataset_repo_id = args.dataset_repo_id or _guess_dataset_repo_id(dataset_root)

    command = [
        str(REPLAY_SCRIPT),
        "--config-path",
        str(_existing_path(str(args.config_path))),
        "--dataset.root",
        str(dataset_root),
        "--dataset.repo_id",
        dataset_repo_id,
        "--dataset.episode",
        str(args.episode),
        "--start-frame",
        str(args.start_frame),
        "--no-play-sounds",
        "--session-name",
        args.session_name,
    ]

    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])

    if args.display:
        command.append("--display-data=true")
    else:
        command.append("--display-data=false")

    if args.fps is not None:
        command.extend(["--fps", str(args.fps)])
    elif args.mode == "live":
        command.extend(["--fps", str(args.live_fps)])

    if args.dry_run:
        command.append("--dry-run")

    if args.mode == "live":
        command.extend(
            [
                "--connect-live",
                "--leader-activity-check-s",
                str(args.leader_activity_check_s),
                "--pre-replay-enable-and-zero=true",
                "--post-replay-return-to-zero=true",
                "--post-replay-hold-s",
                str(args.post_hold_seconds),
                "--speed-percent",
                str(args.speed_percent),
                "--command-repeat",
                str(args.command_repeat),
                "--max-relative-target",
                str(args.max_relative_target),
                "--live-max-joint-step",
                str(args.max_joint_step),
                "--live-max-gripper-step",
                str(args.max_gripper_step),
                "--live-read-cameras=false",
                "--live-include-recorded-media=false",
            ]
        )
        if args.allow_active_leader:
            command.append("--allow-active-leader=true")

    command.extend(passthrough)
    return command


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Stable Songling ALOHA replay launcher. "
            "Defaults to outputs/songling_aloha/local_debug_run."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
        help="Replay mode. 'offline' is safe default; 'live' sends commands to the robot.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Songling unified config path.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Local dataset root. Relative paths are resolved from repo root.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Dataset repo id. If omitted, read from dataset metadata when available.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame within the episode.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame count limit.")
    parser.add_argument("--fps", type=int, default=None, help="Override playback FPS for both modes.")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable Rerun display. Disabled by default for lighter and more stable replay.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved replay settings and exit.")
    parser.add_argument(
        "--python",
        default=None,
        help="Explicit Python interpreter for running replay_episodes.py.",
    )
    parser.add_argument(
        "--conda-env",
        default=DEFAULT_CONDA_ENV,
        help="Fallback conda env name when current Python misses replay dependencies.",
    )
    parser.add_argument(
        "--session-name",
        default="songling_aloha_replay_stable",
        help="Rerun session name passed to the underlying replay script.",
    )

    live_group = parser.add_argument_group("live mode defaults")
    live_group.add_argument(
        "--live-fps",
        type=int,
        default=10,
        help="Default live replay FPS when --fps is not provided.",
    )
    live_group.add_argument(
        "--leader-activity-check-s",
        type=float,
        default=3.0,
        help="How long to observe the bus before live replay for leader traffic.",
    )
    live_group.add_argument(
        "--post-hold-seconds",
        type=float,
        default=5.0,
        help="Hold duration after live replay. Use -1 to hold until Ctrl+C, 0 to disable.",
    )
    live_group.add_argument(
        "--speed-percent",
        type=int,
        default=10,
        help="Conservative robot speed percent for live replay.",
    )
    live_group.add_argument(
        "--command-repeat",
        type=int,
        default=2,
        help="Repeat each live command this many times for better bus robustness.",
    )
    live_group.add_argument(
        "--max-relative-target",
        type=float,
        default=1.5,
        help="Conservative per-command relative target for live replay.",
    )
    live_group.add_argument(
        "--max-joint-step",
        type=float,
        default=0.8,
        help="Per-command joint step limit in degrees for live replay.",
    )
    live_group.add_argument(
        "--max-gripper-step",
        type=float,
        default=2.0,
        help="Per-command gripper step limit in mm for live replay.",
    )
    live_group.add_argument(
        "--allow-active-leader",
        action="store_true",
        help="Force live replay even if shared-bus leader traffic is detected. Not recommended.",
    )

    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> int:
    args, passthrough = _parse_args()
    runner, runner_reason = _resolve_python_runner(args)
    replay_command = _build_replay_command(args, passthrough)
    full_command = runner + replay_command

    print("=" * 72)
    print("Songling ALOHA Stable Replay Launcher")
    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Dataset root: {_existing_path(args.dataset_root)}")
    print(f"Config path: {_existing_path(str(args.config_path))}")
    print(f"Python runner: {runner_reason}")
    if args.mode == "live":
        effective_fps = args.fps if args.fps is not None else args.live_fps
        print(
            "Live safety profile: "
            f"fps={effective_fps}, speed_percent={args.speed_percent}, "
            f"command_repeat={args.command_repeat}, max_relative_target={args.max_relative_target}, "
            f"joint_step={args.max_joint_step}, gripper_step={args.max_gripper_step}, "
            f"post_hold_seconds={args.post_hold_seconds}"
        )
    print("Command:")
    print("  " + shlex.join(full_command))
    sys.stdout.flush()

    completed = subprocess.run(full_command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
