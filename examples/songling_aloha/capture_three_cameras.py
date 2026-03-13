#!/usr/bin/env python

"""Capture one still image from three cameras.

Example:
    # 1) Songling recommended path: fallback to YAML for camera ports
    python examples/songling_aloha/capture_three_cameras.py \
      --config-path examples/songling_aloha/teleop.yaml \
      --output-dir /tmp/songling_snapshots

    # 2) Explicit device overrides (no YAML required)
    python examples/songling_aloha/capture_three_cameras.py \
      --left-high /dev/video0 \
      --left-elbow /dev/video2 \
      --right-elbow /dev/video4 \
      --output-dir /tmp/songling_snapshots

    # 3) LeRobot-style overrides: CLI wins over YAML defaults
    # Note: this expects your environment to have LeRobot installed (e.g. `pip install -e '.[openarms]'`).
    python examples/songling_aloha/capture_three_cameras.py \
      --config-path examples/songling_aloha/teleop.yaml \
      --robot.left_arm_config.cameras.high.index_or_path=/dev/video4 \
      --robot.left_arm_config.cameras.elbow.index_or_path=/dev/video0 \
      --robot.right_arm_config.cameras.elbow.index_or_path=/dev/video2
"""

import argparse
import re
import time
from pathlib import Path
from typing import Any

import cv2


def parse_device(device: str | int) -> str | int:
    """Parse a device arg as int index when possible, otherwise keep as path string."""
    if isinstance(device, int):
        return device
    try:
        return int(str(device))
    except ValueError:
        return device


def device_slug(device: str | int) -> str:
    """Build a filesystem-safe slug from a camera device path/index."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(device)).strip("_").lower()


def capture_one(camera_name: str, device: str | int, output_dir: Path, warmup_frames: int) -> Path:
    cap = cv2.VideoCapture(parse_device(device))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera '{camera_name}' at {device}")

    for _ in range(warmup_frames):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from camera '{camera_name}' at {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{camera_name}__{device_slug(device)}__{timestamp}.jpg"
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Failed to write image for camera '{camera_name}' to {output_path}")

    return output_path


def _load_draccus():
    try:
        import draccus  # type: ignore

    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "draccus is required to load camera devices from YAML fallback.\n"
            "Either:\n"
            "  1) Run inside the LeRobot environment (recommended), or\n"
            "  2) Pass all 3 camera devices explicitly with --left-high/--left-elbow/--right-elbow."
        ) from e
    return draccus


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _extract_cli_camera_overrides(unknown_args: list[str]) -> dict[str, str]:
    """Extract lerobot-style dotted camera overrides from unknown CLI args."""
    dotted_key_map = {
        "robot.left_arm_config.cameras.high.index_or_path": "left_high",
        "robot.left_arm_config.cameras.elbow.index_or_path": "left_elbow",
        "robot.right_arm_config.cameras.elbow.index_or_path": "right_elbow",
    }
    out: dict[str, str] = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if not arg.startswith("--"):
            i += 1
            continue

        key: str | None = None
        value: str | None = None
        if "=" in arg:
            raw_key, raw_value = arg[2:].split("=", 1)
            key = raw_key
            value = raw_value
        elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
            key = arg[2:]
            value = unknown_args[i + 1]
            i += 1

        if key in dotted_key_map and value is not None:
            out[dotted_key_map[key]] = _strip_wrapping_quotes(value)
        i += 1
    return out


def _expect_dict(value, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping at `{path}` in YAML config, got: {type(value)}")
    return value


def _devices_from_songling_yaml(config_path: Path) -> dict[str, str | int]:
    draccus = _load_draccus()
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = draccus.load(dict[str, Any], f)

    root = _expect_dict(raw_cfg, "root")
    robot = _expect_dict(root.get("robot"), "robot")
    left_arm = _expect_dict(robot.get("left_arm_config"), "robot.left_arm_config")
    right_arm = _expect_dict(robot.get("right_arm_config"), "robot.right_arm_config")
    left_cams = _expect_dict(left_arm.get("cameras"), "robot.left_arm_config.cameras")
    right_cams = _expect_dict(right_arm.get("cameras"), "robot.right_arm_config.cameras")
    left_high_cfg = _expect_dict(left_cams.get("high"), "robot.left_arm_config.cameras.high")
    left_elbow_cfg = _expect_dict(left_cams.get("elbow"), "robot.left_arm_config.cameras.elbow")
    right_elbow_cfg = _expect_dict(right_cams.get("elbow"), "robot.right_arm_config.cameras.elbow")

    missing = []
    if "index_or_path" not in left_high_cfg:
        missing.append("robot.left_arm_config.cameras.high.index_or_path")
    if "index_or_path" not in left_elbow_cfg:
        missing.append("robot.left_arm_config.cameras.elbow.index_or_path")
    if "index_or_path" not in right_elbow_cfg:
        missing.append("robot.right_arm_config.cameras.elbow.index_or_path")
    if missing:
        raise ValueError(f"Missing required camera field(s) in YAML: {', '.join(missing)}")

    return {
        "left_high": left_high_cfg["index_or_path"],
        "left_elbow": left_elbow_cfg["index_or_path"],
        "right_elbow": right_elbow_cfg["index_or_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one still frame from three Songling ALOHA cameras.")
    parser.add_argument(
        "--config-path",
        "--config_path",
        type=Path,
        default=Path("examples/songling_aloha/teleop.yaml"),
        help="Songling unified YAML. Used as fallback when a camera device is not specified explicitly.",
    )
    parser.add_argument("--left-high", default=None, help="Path/index override for left_high (top) camera.")
    parser.add_argument("--left-elbow", default=None, help="Path/index override for left_elbow camera.")
    parser.add_argument("--right-elbow", default=None, help="Path/index override for right_elbow camera.")
    parser.add_argument("--output-dir", default="/tmp/songling_snapshots", help="Directory to save images.")
    parser.add_argument(
        "--warmup-frames", type=int, default=15, help="Number of warmup frames before capture."
    )
    args, unknown_args = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    devices: dict[str, str | int | None] = {
        "left_high": args.left_high,
        "left_elbow": args.left_elbow,
        "right_elbow": args.right_elbow,
    }

    # Support lerobot-style dotted CLI overrides:
    # --robot.left_arm_config.cameras.high.index_or_path=...
    # --robot.left_arm_config.cameras.elbow.index_or_path=...
    # --robot.right_arm_config.cameras.elbow.index_or_path=...
    cli_devices = _extract_cli_camera_overrides(unknown_args)
    for k, v in cli_devices.items():
        if devices.get(k) is None:
            devices[k] = v

    if any(v is None for v in devices.values()):
        if args.config_path is None or not args.config_path.exists():
            raise ValueError(
                "Some camera devices were not provided, but --config-path is missing or does not exist.\n"
                "Either pass all 3 devices explicitly, or set --config-path to a valid Songling YAML."
            )

        fallback_devices = _devices_from_songling_yaml(args.config_path)
        for k, v in fallback_devices.items():
            if devices.get(k) is None:
                devices[k] = v

    missing = [k for k, v in devices.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing camera device(s): {', '.join(missing)}. "
            "Provide them via --left-high/--left-elbow/--right-elbow or in the YAML config."
        )

    for camera_name in ("left_high", "left_elbow", "right_elbow"):
        device = devices[camera_name]
        output_path = capture_one(camera_name, device, output_dir, args.warmup_frames)
        print(f"{camera_name}: saved {output_path}")


if __name__ == "__main__":
    main()
