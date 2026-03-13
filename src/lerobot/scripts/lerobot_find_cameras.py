#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper to find the camera devices available in your system.

Example:

```shell
lerobot-find-cameras opencv
```
"""

# NOTE(Steven): RealSense can also be identified/opened as OpenCV cameras. If you know the camera is a RealSense, use the `lerobot-find-cameras realsense` flag to avoid confusion.
# NOTE(Steven): macOS cameras sometimes report different FPS at init time, not an issue here as we don't specify FPS when opening the cameras, but the information displayed might not be truthful.

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()


def _devnode_token(cam_id: str) -> str:
    # Prefer a human-readable basename like "video0".
    try:
        if cam_id.startswith("/"):
            return Path(cam_id).name
    except Exception:
        pass
    return _slug(cam_id)


def _byid_token(by_id_path: str) -> str:
    """Build a short but stable token from a /dev/v4l/by-id symlink path."""
    name = Path(by_id_path).name
    # Common udev pattern:
    # usb-<vendor>_<model>_<SERIAL>-video-index0
    m = re.search(r"_([A-Za-z0-9]+)-video-index(\d+)$", name)
    if m:
        serial = m.group(1)
        idx = m.group(2)
        return f"{serial}_idx{idx}"
    m = re.search(r"-video-index(\d+)$", name)
    if m:
        return f"videoidx{m.group(1)}"
    token = _slug(name)
    return token[:48] if len(token) > 48 else token


def _find_v4l_symlinks(devnode: str) -> dict[str, list[str]]:
    """Best-effort mapping from /dev/videoX -> /dev/v4l/by-id and /dev/v4l/by-path symlinks."""
    devnode_path = Path(devnode)
    out: dict[str, list[str]] = {"by_id": [], "by_path": []}
    for base, key in [(Path("/dev/v4l/by-id"), "by_id"), (Path("/dev/v4l/by-path"), "by_path")]:
        if not base.exists():
            continue
        for p in base.iterdir():
            try:
                if p.is_symlink() and p.resolve() == devnode_path:
                    out[key].append(str(p))
            except Exception:
                continue
    out["by_id"].sort()
    out["by_path"].sort()
    return out


def _yaml_scalar_for_index_or_path(value: str) -> str:
    raw = value.strip()
    if re.fullmatch(r"-?\d+", raw):
        return raw
    return raw


def _yaml_scalar(value: object) -> str:
    """Serialize simple scalar values for in-place YAML updates."""
    if value is None:
        raise ValueError("Cannot serialize None as YAML scalar.")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


_CV2_BACKEND_NAMES: dict[int, str] = {
    0: "ANY",
    200: "V4L2",
    700: "DSHOW",
    800: "PVAPI",
    1000: "ANDROID",
    1200: "AVFOUNDATION",
    1400: "MSMF",
}


def _backend_comment(value: str) -> str:
    try:
        backend_i = int(str(value))
    except Exception:
        return ""
    backend_name = _CV2_BACKEND_NAMES.get(backend_i)
    return f"  # {backend_name}" if backend_name else ""


def _replace_songling_camera_config_in_text(
    text: str,
    *,
    left_high: str,
    left_elbow: str,
    right_elbow: str,
    opencv_width: int | None = None,
    opencv_height: int | None = None,
    opencv_fps: int | None = None,
    opencv_fourcc: str | None = None,
    opencv_backend: int | None = None,
) -> tuple[str, dict[tuple[str, str], set[str]]]:
    targets: dict[tuple[str, str], dict[str, str]] = {
        ("left", "high"): {"index_or_path": _yaml_scalar_for_index_or_path(left_high)},
        ("left", "elbow"): {"index_or_path": _yaml_scalar_for_index_or_path(left_elbow)},
        ("right", "elbow"): {"index_or_path": _yaml_scalar_for_index_or_path(right_elbow)},
    }
    # Optionally sync common OpenCV capture params into the profile config.
    for key in targets:
        if opencv_width is not None:
            targets[key]["width"] = _yaml_scalar(opencv_width)
        if opencv_height is not None:
            targets[key]["height"] = _yaml_scalar(opencv_height)
        if opencv_fps is not None:
            targets[key]["fps"] = _yaml_scalar(opencv_fps)
        if opencv_fourcc is not None:
            fourcc = opencv_fourcc.strip()
            if not fourcc:
                raise ValueError("--opencv-fourcc cannot be empty when provided.")
            targets[key]["fourcc"] = _yaml_scalar(fourcc)
        if opencv_backend is not None:
            targets[key]["backend"] = _yaml_scalar(opencv_backend)

    written_fields: dict[tuple[str, str], set[str]] = {k: set() for k in targets}

    lines = text.splitlines(keepends=True)
    out_lines: list[str] = []
    in_top: str | None = None
    in_arm: str | None = None
    in_cameras = False
    in_camera: str | None = None

    def _flush_missing_fields() -> None:
        if in_arm is None or in_camera is None:
            return
        camera_key = (in_arm, in_camera)
        if camera_key not in targets:
            return
        desired = targets[camera_key]
        for field_name in ("index_or_path", "width", "height", "fps", "fourcc", "backend"):
            if field_name not in desired or field_name in written_fields[camera_key]:
                continue
            value = desired[field_name]
            if field_name == "backend":
                out_lines.append(f"        {field_name}: {value}{_backend_comment(value)}\n")
            else:
                out_lines.append(f"        {field_name}: {value}\n")
            written_fields[camera_key].add(field_name)

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if indent == 0 and stripped.endswith(":"):
            _flush_missing_fields()
            section = stripped[:-1]
            in_top = section if section in {"teleop", "robot"} else None
            in_arm = None
            in_cameras = False
            in_camera = None
            out_lines.append(line)
            continue

        if in_top != "robot":
            out_lines.append(line)
            continue

        if indent == 2 and stripped.endswith(":"):
            _flush_missing_fields()
            if stripped == "left_arm_config:":
                in_arm = "left"
            elif stripped == "right_arm_config:":
                in_arm = "right"
            else:
                in_arm = None
            in_cameras = False
            in_camera = None
            out_lines.append(line)
            continue

        if in_arm is None:
            out_lines.append(line)
            continue

        if indent == 4 and stripped.endswith(":"):
            _flush_missing_fields()
            if stripped == "cameras:":
                in_cameras = True
                in_camera = None
            else:
                in_cameras = False
                in_camera = None
            out_lines.append(line)
            continue

        if not in_cameras:
            out_lines.append(line)
            continue

        if indent == 6 and stripped.endswith(":"):
            _flush_missing_fields()
            in_camera = stripped[:-1]
            out_lines.append(line)
            continue

        if in_camera is None:
            out_lines.append(line)
            continue

        camera_key = (in_arm, in_camera)
        if camera_key in targets and indent == 8 and ":" in stripped:
            field_name = stripped.split(":", 1)[0].strip()
            if field_name in targets[camera_key]:
                value = targets[camera_key][field_name]
                prefix = line[: len(line) - len(line.lstrip(" "))]
                nl = "\n" if line.endswith("\n") else ""
                original_no_nl = line[:-1] if nl else line
                comment = ""
                if "#" in original_no_nl:
                    hash_idx = original_no_nl.find("#")
                    # Include preceding spaces to preserve formatting.
                    start = hash_idx
                    while start > 0 and original_no_nl[start - 1] == " ":
                        start -= 1
                    comment = original_no_nl[start:]
                if field_name == "backend":
                    comment = _backend_comment(value)
                out_lines.append(f"{prefix}{field_name}: {value}{comment}{nl}")
                written_fields[camera_key].add(field_name)
                continue

        out_lines.append(line)

    _flush_missing_fields()
    return "".join(out_lines), written_fields


def update_songling_config_camera_ports(
    config_path: Path,
    *,
    left_high: str,
    left_elbow: str,
    right_elbow: str,
    opencv_width: int | None = None,
    opencv_height: int | None = None,
    opencv_fps: int | None = None,
    opencv_fourcc: str | None = None,
    opencv_backend: int | None = None,
) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    original = config_path.read_text(encoding="utf-8")
    updated, written_fields = _replace_songling_camera_config_in_text(
        original,
        left_high=left_high,
        left_elbow=left_elbow,
        right_elbow=right_elbow,
        opencv_width=opencv_width,
        opencv_height=opencv_height,
        opencv_fps=opencv_fps,
        opencv_fourcc=opencv_fourcc,
        opencv_backend=opencv_backend,
    )

    expected_fields = {
        ("left", "high"): {"index_or_path"},
        ("left", "elbow"): {"index_or_path"},
        ("right", "elbow"): {"index_or_path"},
    }
    for key in expected_fields:
        if opencv_width is not None:
            expected_fields[key].add("width")
        if opencv_height is not None:
            expected_fields[key].add("height")
        if opencv_fps is not None:
            expected_fields[key].add("fps")
        if opencv_fourcc is not None:
            expected_fields[key].add("fourcc")
        if opencv_backend is not None:
            expected_fields[key].add("backend")

    if written_fields != expected_fields:
        raise RuntimeError(
            f"Failed to update camera mapping uniquely in {config_path}. "
            f"written_fields={written_fields}, expected_fields={expected_fields}"
        )

    config_path.write_text(updated, encoding="utf-8")
    print(
        "Updated Songling camera mapping in config yaml: "
        f"left_high={left_high}, left_elbow={left_elbow}, right_elbow={right_elbow}"
    )


def update_songling_teleop_camera_ports(
    config_path: Path,
    *,
    left_high: str,
    left_elbow: str,
    right_elbow: str,
    opencv_width: int | None = None,
    opencv_height: int | None = None,
    opencv_fps: int | None = None,
    opencv_fourcc: str | None = None,
    opencv_backend: int | None = None,
) -> None:
    # Deprecated alias kept for backward compatibility.
    update_songling_config_camera_ports(
        config_path=config_path,
        left_high=left_high,
        left_elbow=left_elbow,
        right_elbow=right_elbow,
        opencv_width=opencv_width,
        opencv_height=opencv_height,
        opencv_fps=opencv_fps,
        opencv_fourcc=opencv_fourcc,
        opencv_backend=opencv_backend,
    )


def update_songling_record_camera_ports(
    config_path: Path,
    *,
    left_high: str,
    left_elbow: str,
    right_elbow: str,
    opencv_width: int | None = None,
    opencv_height: int | None = None,
    opencv_fps: int | None = None,
    opencv_fourcc: str | None = None,
    opencv_backend: int | None = None,
) -> None:
    # Deprecated alias kept for backward compatibility.
    update_songling_config_camera_ports(
        config_path=config_path,
        left_high=left_high,
        left_elbow=left_elbow,
        right_elbow=right_elbow,
        opencv_width=opencv_width,
        opencv_height=opencv_height,
        opencv_fps=opencv_fps,
        opencv_fourcc=opencv_fourcc,
        opencv_backend=opencv_backend,
    )


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid manifest format (expected list): {manifest_path}")
    return data


def _suggested_port_from_item(item: dict[str, Any], label: str) -> str:
    port = item.get("suggested_index_or_path")
    if isinstance(port, str) and port:
        return port
    raise ValueError(f"{label} missing suggested_index_or_path")


def _extract_video_number(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"/dev/video(\d+)$", value)
    if match is None:
        return None
    return int(match.group(1))


def _port_from_live_opencv(index: int) -> str | None:
    try:
        live = find_all_opencv_cameras()
    except Exception:
        return None
    if not live:
        return None

    # 1) If index is a live list ordinal (e.g. Camera #2), use it directly.
    if 0 <= index < len(live):
        item = live[index]
        port = (item.get("by_id") or [item.get("id")])[0]
        if isinstance(port, str) and port:
            logger.warning(
                "Manifest index %s is out of range for the provided manifest. "
                "Falling back to current live OpenCV camera ordinal #%s -> %s.",
                index,
                index,
                port,
            )
            return port

    # 2) If index looks like /dev/videoN, match N from current cameras.
    for item in live:
        video_no = _extract_video_number(item.get("id"))
        if video_no == index:
            port = (item.get("by_id") or [item.get("id")])[0]
            if isinstance(port, str) and port:
                logger.warning(
                    "Manifest index %s not found by ordinal. "
                    "Interpreting it as /dev/video%s from live cameras -> %s.",
                    index,
                    index,
                    port,
                )
                return port
    return None


def _manifest_suggested_port(manifest: list[dict[str, Any]], index: int) -> str:
    # 1) Prefer explicit "index" field written in manifest.
    for item in manifest:
        if isinstance(item, dict) and item.get("index") == index:
            return _suggested_port_from_item(item, f"manifest entry index={index}")

    # 2) Accept /dev/videoN style numbers by matching camera id/suggested path.
    for item in manifest:
        if not isinstance(item, dict):
            continue
        id_video_no = _extract_video_number(item.get("id"))
        suggested_video_no = _extract_video_number(item.get("suggested_index_or_path"))
        if id_video_no == index or suggested_video_no == index:
            return _suggested_port_from_item(item, f"manifest entry matching /dev/video{index}")

    # 3) Fall back to manifest list ordinal.
    if 0 <= index < len(manifest):
        item = manifest[index]
        if isinstance(item, dict):
            return _suggested_port_from_item(item, f"manifest[{index}]")
        raise ValueError(f"manifest[{index}] is not a mapping")

    # 4) Manifest may be stale. Try current live OpenCV cameras.
    live_port = _port_from_live_opencv(index)
    if live_port is not None:
        return live_port

    raise ValueError(
        f"Index {index} not found in manifest (size={len(manifest)}), "
        "and no matching live OpenCV camera could be resolved. "
        "Re-run `lerobot-find-cameras opencv --mode snapshot` to refresh manifest."
    )


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """
    Finds all available OpenCV cameras plugged into the system.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            cam_id = cam_info.get("id")
            if isinstance(cam_id, str) and cam_id.startswith("/dev/video"):
                cam_info.update(_find_v4l_symlinks(cam_id))
            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(opencv_cameras)} OpenCV cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense" or "opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: list[dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter == "opencv":
        all_cameras_info.extend(find_all_opencv_cameras())
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No cameras (OpenCV or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                elif key in ("by_id", "by_path") and isinstance(value, list):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    filename_prefix: str,
    images_dir: Path,
    camera_type: str,
    image_ext: str = "png",
):
    """
    Saves a single image to disk using Pillow. Handles color conversion if necessary.
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")
        images_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{filename_prefix}.{image_ext}"
        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image for camera {filename_prefix} (type {camera_type}): {e}")


def create_camera_instance(
    cam_meta: dict[str, Any],
    *,
    opencv_width: int | None = None,
    opencv_height: int | None = None,
    opencv_fps: int | None = None,
    opencv_fourcc: str | None = None,
    opencv_backend: int | None = None,
) -> dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    try:
        if cam_type == "OpenCV":
            index_or_path = Path(cam_id) if isinstance(cam_id, str) and cam_id.startswith("/") else cam_id
            # Try a "preferred" config (bandwidth-friendly) first; fall back to default config on failure.
            preferred_cfg = OpenCVCameraConfig(
                index_or_path=index_or_path,
                color_mode=ColorMode.RGB,
                width=opencv_width,
                height=opencv_height,
                fps=opencv_fps,
                fourcc=opencv_fourcc,
                backend=opencv_backend if opencv_backend is not None else 0,
            )
            try:
                instance = OpenCVCamera(preferred_cfg)
            except Exception:
                default_cfg = OpenCVCameraConfig(index_or_path=index_or_path, color_mode=ColorMode.RGB)
                instance = OpenCVCamera(default_cfg)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=True)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(
    cam_dict: dict[str, Any],
    output_dir: Path,
    snapshot_prefix: str,
    *,
    max_age_ms: int = 1000,
    image_ext: str = "png",
) -> Path | None:
    """Capture and save a single snapshot from a camera."""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))

    try:
        # Prefer read_latest to avoid blocking forever in case of stale cameras.
        if hasattr(cam, "read_latest"):
            image_data = cam.read_latest(max_age_ms=max_age_ms)
        else:
            image_data = cam.read()
        save_image(image_data, snapshot_prefix, output_dir, cam_type_str, image_ext=image_ext)
        return output_dir / f"{snapshot_prefix}.{image_ext}"
    except TimeoutError:
        logger.warning(f"Timeout reading from {cam_type_str} camera for snapshot '{snapshot_prefix}'.")
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera for snapshot '{snapshot_prefix}': {e}")
    return None


def cleanup_cameras(cameras_to_use: list[dict[str, Any]]):
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
    mode: str = "snapshot",
    image_ext: str = "png",
    max_age_ms: int = 1000,
    opencv_width: int | None = None,
    opencv_height: int | None = None,
    opencv_fps: int | None = None,
    opencv_fourcc: str | None = None,
    opencv_backend: int | None = None,
    save_manifest: bool = True,
    snapshot_name_style: str = "short",
):
    """
    Connects to detected cameras (optionally filtered by type) and saves images from each.
    Uses default stream profiles for width, height, and FPS.

    Args:
        output_dir: Directory to save images.
        record_time_s: Duration in seconds to record images.
        camera_type: Optional string to filter cameras ("realsense" or "opencv").
                            If None, uses all detected cameras.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir} (mode={mode})")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    if mode == "list":
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(
            cam_meta,
            opencv_width=opencv_width,
            opencv_height=opencv_height,
            opencv_fps=opencv_fps,
            opencv_fourcc=opencv_fourcc,
            opencv_backend=opencv_backend,
        )
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting image save.")
        return

    manifest: list[dict[str, Any]] = []
    try:
        if mode == "snapshot":
            for i, cam_dict in enumerate(cameras_to_use):
                meta = cam_dict["meta"]
                cam_type = str(meta.get("type", "unknown")).lower()
                cam_id = str(meta.get("id", "unknown"))
                by_id = meta.get("by_id") or []
                if snapshot_name_style == "long":
                    by_id_tag = _slug(Path(by_id[0]).name) if by_id else ""
                    prefix = f"{cam_type}_{i:02d}__{_slug(cam_id)}"
                    if by_id_tag:
                        prefix += f"__byid_{by_id_tag}"
                else:
                    dev_tok = _devnode_token(cam_id)
                    byid_tok = _byid_token(by_id[0]) if by_id else ""
                    prefix = f"{cam_type}_{i:02d}"
                    if dev_tok:
                        prefix += f"__{dev_tok}"
                    if byid_tok:
                        prefix += f"__{byid_tok}"

                snap_path = process_camera_image(
                    cam_dict,
                    output_dir,
                    prefix,
                    max_age_ms=max_age_ms,
                    image_ext=image_ext,
                )
                manifest.append(
                    {
                        "index": i,
                        "type": meta.get("type"),
                        "id": meta.get("id"),
                        "by_id": meta.get("by_id", []),
                        "by_path": meta.get("by_path", []),
                        "default_stream_profile": meta.get("default_stream_profile", {}),
                        "backend_api": meta.get("backend_api"),
                        "snapshot": str(snap_path) if snap_path else None,
                        "suggested_index_or_path": (meta.get("by_id") or [meta.get("id")])[0],
                    }
                )

        elif mode == "record":
            logger.info(f"Starting image capture for {record_time_s} seconds from {len(cameras_to_use)} cameras.")
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < record_time_s:
                for i, cam_dict in enumerate(cameras_to_use):
                    meta = cam_dict["meta"]
                    cam_type = str(meta.get("type", "unknown")).lower()
                    cam_id = str(meta.get("id", "unknown"))
                    if snapshot_name_style == "long":
                        prefix = f"{cam_type}_{i:02d}__{_slug(cam_id)}__latest"
                    else:
                        prefix = f"{cam_type}_{i:02d}__{_devnode_token(cam_id)}__latest"
                    _ = process_camera_image(
                        cam_dict,
                        output_dir,
                        prefix,
                        max_age_ms=max_age_ms,
                        image_ext=image_ext,
                    )
                time.sleep(0.1)
        else:
            raise ValueError(f"Unknown mode={mode!r}. Expected list|snapshot|record.")
    except KeyboardInterrupt:
        logger.info("Capture interrupted by user.")
    finally:
        cleanup_cameras(cameras_to_use)

    if save_manifest and manifest:
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nSaved manifest: {manifest_path}")
        print("Use the saved snapshots to map physical cameras to ports, then use `suggested_index_or_path` values.")
    print(f"Image capture finished. Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--mode",
        choices=["list", "snapshot", "record", "configure-songling"],
        default="snapshot",
        help="Mode: list cameras only, save one snapshot per camera, record for a duration, "
        "or only update the Songling config yaml camera mapping.",
    )
    parser.add_argument(
        "--image-ext",
        choices=["png", "jpg"],
        default="png",
        help="Snapshot image format. Default: png",
    )
    parser.add_argument(
        "--max-age-ms",
        type=int,
        default=1000,
        help="Max frame age for read_latest() when saving snapshots. Default: 1000ms",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=6.0,
        help="Time duration to attempt capturing frames. Default: 6 seconds.",
    )
    parser.add_argument("--opencv-width", type=int, default=None, help="Optional OpenCV width override.")
    parser.add_argument("--opencv-height", type=int, default=None, help="Optional OpenCV height override.")
    parser.add_argument("--opencv-fps", type=int, default=None, help="Optional OpenCV fps override.")
    parser.add_argument("--opencv-fourcc", default=None, help="Optional OpenCV FOURCC (e.g. MJPG).")
    parser.add_argument(
        "--opencv-backend",
        type=int,
        default=None,
        help="Optional OpenCV backend (e.g. 200 for V4L2).",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable writing manifest.json alongside snapshots.",
    )
    parser.add_argument(
        "--snapshot-name-style",
        choices=["short", "long"],
        default="short",
        help="Filename style for saved snapshots. 'short' is readable; 'long' includes full slugified IDs.",
    )
    parser.add_argument(
        "--write-songling-config",
        type=Path,
        default=None,
        help="Optional path to the Songling config yaml to auto-update camera fields.",
    )
    parser.add_argument(
        "--write-songling-teleop",
        type=Path,
        default=None,
        help="Deprecated alias of --write-songling-config.",
    )
    parser.add_argument(
        "--songling-left-high",
        default=None,
        help="Port/path to write into robot.left_arm_config.cameras.high.index_or_path.",
    )
    parser.add_argument(
        "--songling-left-elbow",
        default=None,
        help="Port/path to write into robot.left_arm_config.cameras.elbow.index_or_path.",
    )
    parser.add_argument(
        "--songling-right-elbow",
        default=None,
        help="Port/path to write into robot.right_arm_config.cameras.elbow.index_or_path.",
    )
    parser.add_argument(
        "--songling-manifest",
        type=Path,
        default=None,
        help="Optional manifest.json produced by `lerobot-find-cameras --mode snapshot` to resolve ports by index.",
    )
    parser.add_argument(
        "--songling-left-high-index",
        type=int,
        default=None,
        help="Camera selector for left_high. Accepts manifest ordinal (0,1,2,...) "
        "or /dev/videoN number (e.g. 5 for /dev/video5).",
    )
    parser.add_argument(
        "--songling-left-elbow-index",
        type=int,
        default=None,
        help="Camera selector for left_elbow. Accepts manifest ordinal or /dev/videoN number.",
    )
    parser.add_argument(
        "--songling-right-elbow-index",
        type=int,
        default=None,
        help="Camera selector for right_elbow. Accepts manifest ordinal or /dev/videoN number.",
    )
    args = parser.parse_args()

    write_songling_config = args.write_songling_config
    if args.write_songling_teleop is not None:
        if write_songling_config is None:
            write_songling_config = args.write_songling_teleop
        elif write_songling_config != args.write_songling_teleop:
            raise ValueError(
                "--write-songling-config and --write-songling-teleop were both provided with different paths."
            )

    if write_songling_config is not None:
        if args.songling_left_high and args.songling_left_elbow and args.songling_right_elbow:
            left_high = args.songling_left_high
            left_elbow = args.songling_left_elbow
            right_elbow = args.songling_right_elbow
        elif (
            args.songling_manifest is not None
            and args.songling_left_high_index is not None
            and args.songling_left_elbow_index is not None
            and args.songling_right_elbow_index is not None
        ):
            manifest = _load_manifest(args.songling_manifest)
            left_high = _manifest_suggested_port(manifest, args.songling_left_high_index)
            left_elbow = _manifest_suggested_port(manifest, args.songling_left_elbow_index)
            right_elbow = _manifest_suggested_port(manifest, args.songling_right_elbow_index)
        else:
            raise ValueError(
                "--write-songling-config requires either:\n"
                "  (A) --songling-left-high/--songling-left-elbow/--songling-right-elbow, or\n"
                "  (B) --songling-manifest plus --songling-*-index for all three cameras."
            )

        update_songling_config_camera_ports(
            write_songling_config,
            left_high=left_high,
            left_elbow=left_elbow,
            right_elbow=right_elbow,
            opencv_width=args.opencv_width,
            opencv_height=args.opencv_height,
            opencv_fps=args.opencv_fps,
            opencv_fourcc=args.opencv_fourcc,
            opencv_backend=args.opencv_backend,
        )
        if args.mode == "configure-songling":
            return
    elif args.mode == "configure-songling":
        raise ValueError("--mode configure-songling requires --write-songling-config.")

    args_dict = vars(args)
    args_dict["save_manifest"] = not args.no_manifest
    args_dict.pop("no_manifest", None)
    args_dict.pop("write_songling_config", None)
    args_dict.pop("write_songling_teleop", None)
    args_dict.pop("songling_left_high", None)
    args_dict.pop("songling_left_elbow", None)
    args_dict.pop("songling_right_elbow", None)
    args_dict.pop("songling_manifest", None)
    args_dict.pop("songling_left_high_index", None)
    args_dict.pop("songling_left_elbow_index", None)
    args_dict.pop("songling_right_elbow_index", None)
    save_images_from_all_cameras(**args_dict)


if __name__ == "__main__":
    main()
