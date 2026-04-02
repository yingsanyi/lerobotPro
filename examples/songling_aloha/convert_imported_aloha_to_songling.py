#!/usr/bin/env python

"""Convert an imported ALOHA-style dataset into a Songling-compatible LeRobot v3 dataset.

This script is intended for local datasets that were copied from elsewhere and still use:

- LeRobot v2.1 metadata/layout
- `observations.state` instead of `observation.state`
- ALOHA camera keys such as `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- arm joint values stored in radians
- gripper values stored in meters or normalized units

It writes a new dataset root and leaves the source dataset untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    cast_stats_to_numpy,
    create_empty_dataset_info,
    flatten_dict,
    write_episodes,
    write_info,
    write_json,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import get_video_duration_in_s
from lerobot.utils.utils import init_logging


logger = logging.getLogger(__name__)

SOURCE_STATE_KEY = "observations.state"
TARGET_STATE_KEY = "observation.state"
ACTION_KEY = "action"

DEFAULT_CAMERA_MAP = {
    "observation.images.cam_high": "observation.images.left_high",
    "observation.images.cam_left_wrist": "observation.images.left_elbow",
    "observation.images.cam_right_wrist": "observation.images.right_elbow",
}

TARGET_VECTOR_NAMES = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_gripper.pos",
]

GRIPPER_INDICES = (6, 13)
ARM_INDICES = tuple(idx for idx in range(len(TARGET_VECTOR_NAMES)) if idx not in GRIPPER_INDICES)
IMPORT_SUMMARY_FILENAME = "songling_import_summary.json"


@dataclass
class _StepStatsAccumulator:
    joint_deltas: list[np.ndarray] = field(default_factory=list)
    gripper_deltas: list[np.ndarray] = field(default_factory=list)

    def add(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(TARGET_VECTOR_NAMES) or len(array) < 2:
            return
        delta = np.abs(np.diff(array, axis=0))
        self.joint_deltas.append(delta[:, ARM_INDICES].reshape(-1))
        self.gripper_deltas.append(delta[:, GRIPPER_INDICES].reshape(-1))


@dataclass
class _ImportAnalysisAccumulator:
    action_step_stats: _StepStatsAccumulator = field(default_factory=_StepStatsAccumulator)
    state_step_stats: _StepStatsAccumulator = field(default_factory=_StepStatsAccumulator)
    action_vs_state_abs_sum: float = 0.0
    action_vs_state_count: int = 0
    action_vs_next_state_abs_sum: float = 0.0
    action_vs_next_state_count: int = 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object per line in {path}, got: {type(payload)!r}")
            items.append(payload)
    return items


def _ensure_local_hf_cache(root: Path) -> None:
    home_root = root / ".hf_home"
    datasets_root = home_root / "datasets"
    downloads_root = datasets_root / "downloads"
    datasets_root.mkdir(parents=True, exist_ok=True)
    downloads_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(home_root / "hub")

    try:
        import datasets

        datasets.config.HF_CACHE_HOME = str(home_root)
        datasets.config.HF_DATASETS_CACHE = str(datasets_root)
        datasets.config.DOWNLOADED_DATASETS_PATH = str(downloads_root)
    except Exception:
        pass


def _resolve_gripper_scale(unit: str) -> float:
    if unit == "m":
        return 1000.0
    if unit == "mm":
        return 1.0
    if unit == "normalized_70":
        return 70.0
    if unit == "normalized_100":
        return 100.0
    raise ValueError(f"Unsupported gripper unit: {unit}")


def _transform_vector_array(values: np.ndarray, *, arm_unit: str, gripper_scale: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).copy()
    if array.shape[-1] != len(TARGET_VECTOR_NAMES):
        raise ValueError(
            f"Expected last dimension {len(TARGET_VECTOR_NAMES)} for state/action vectors, got {array.shape}."
        )

    if arm_unit == "rad":
        array[..., ARM_INDICES] = np.rad2deg(array[..., ARM_INDICES])
    elif arm_unit != "deg":
        raise ValueError(f"Unsupported arm unit: {arm_unit}")

    array[..., GRIPPER_INDICES] *= float(gripper_scale)
    return array.astype(np.float32, copy=False)


def _transform_vector_stats(
    stats: dict[str, Any],
    *,
    arm_unit: str,
    gripper_scale: float,
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for stat_name, value in stats.items():
        if stat_name == "count":
            converted[stat_name] = value
            continue
        transformed = _transform_vector_array(np.asarray(value, dtype=np.float32), arm_unit=arm_unit, gripper_scale=gripper_scale)
        converted[stat_name] = transformed.tolist()
    return converted


def _step_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _finalize_step_stats(accumulator: _StepStatsAccumulator) -> dict[str, float]:
    joint = (
        np.concatenate(accumulator.joint_deltas, axis=0).astype(np.float32, copy=False)
        if accumulator.joint_deltas
        else np.asarray([], dtype=np.float32)
    )
    gripper = (
        np.concatenate(accumulator.gripper_deltas, axis=0).astype(np.float32, copy=False)
        if accumulator.gripper_deltas
        else np.asarray([], dtype=np.float32)
    )
    return {
        "joint_p95_deg": _step_percentile(joint, 95.0),
        "joint_p99_deg": _step_percentile(joint, 99.0),
        "joint_max_deg": float(np.max(joint)) if joint.size else 0.0,
        "gripper_p95_mm": _step_percentile(gripper, 95.0),
        "gripper_p99_mm": _step_percentile(gripper, 99.0),
        "gripper_max_mm": float(np.max(gripper)) if gripper.size else 0.0,
    }


def _mean_abs(sum_value: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return float(sum_value / count)


def _finalize_import_analysis(accumulator: _ImportAnalysisAccumulator) -> dict[str, Any]:
    mae_vs_state = _mean_abs(accumulator.action_vs_state_abs_sum, accumulator.action_vs_state_count)
    mae_vs_next_state = _mean_abs(accumulator.action_vs_next_state_abs_sum, accumulator.action_vs_next_state_count)

    semantics_kind = "independent_control"
    preferred_control_source = ACTION_KEY
    replay_reason = "Replay uses recorded action targets directly."
    if (
        accumulator.action_vs_next_state_count > 0
        and mae_vs_next_state <= max(1e-6, 0.1 * max(mae_vs_state, 1e-6))
        and mae_vs_next_state < 0.1
    ):
        semantics_kind = "matches_next_observation_state"
        preferred_control_source = TARGET_STATE_KEY
        replay_reason = (
            "Imported action closely matches the next observation.state frame, so live replay should follow "
            "observation.state for a trajectory that better matches the recorded follower path."
        )
    elif mae_vs_state < mae_vs_next_state:
        semantics_kind = "matches_same_step_observation_state"

    return {
        "action_semantics": {
            "kind": semantics_kind,
            "mean_abs_error_vs_observation_state": mae_vs_state,
            "mean_abs_error_vs_next_observation_state": mae_vs_next_state,
        },
        "replay": {
            "preferred_control_source": preferred_control_source,
            "reason": replay_reason,
        },
        "step_stats": {
            ACTION_KEY: _finalize_step_stats(accumulator.action_step_stats),
            TARGET_STATE_KEY: _finalize_step_stats(accumulator.state_step_stats),
        },
    }


def _camera_shape_hwc(feature: dict[str, Any]) -> tuple[int, int, int]:
    shape = list(feature.get("shape") or ())
    names = list(feature.get("names") or ())
    if len(shape) != 3:
        raise ValueError(f"Expected 3D camera shape, got {shape}.")
    if names == ["channels", "height", "width"]:
        return int(shape[1]), int(shape[2]), int(shape[0])
    if names == ["height", "width", "channels"] or not names:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(f"Unsupported camera shape metadata names: {names}")


def _build_target_features(
    *,
    source_info: dict[str, Any],
    fps: int,
    camera_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(TARGET_VECTOR_NAMES),),
            "names": TARGET_VECTOR_NAMES,
        },
        TARGET_STATE_KEY: {
            "dtype": "float32",
            "shape": (len(TARGET_VECTOR_NAMES),),
            "names": TARGET_VECTOR_NAMES,
        },
    }

    source_features = source_info.get("features", {})
    for source_key, target_key in camera_map.items():
        source_feature = source_features.get(source_key)
        if not isinstance(source_feature, dict):
            raise KeyError(f"Source dataset is missing camera feature {source_key!r}.")
        height, width, channels = _camera_shape_hwc(source_feature)
        target_feature = {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        }
        source_video_info = source_feature.get("info")
        if isinstance(source_video_info, dict):
            target_video_info = dict(source_video_info)
            target_video_info["video.height"] = height
            target_video_info["video.width"] = width
            target_video_info["video.channels"] = channels
            target_video_info["video.fps"] = int(fps)
            target_feature["info"] = target_video_info
        features[target_key] = target_feature

    features["timestamp"] = {"dtype": "float32", "shape": (1,), "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    features["index"] = {"dtype": "int64", "shape": (1,), "names": None}
    features["task_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def _convert_tasks(source_root: Path, target_root: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    legacy_tasks = sorted(_load_jsonl(source_root / "meta" / "tasks.jsonl"), key=lambda item: int(item["task_index"]))
    task_index_to_text = {int(item["task_index"]): str(item["task"]) for item in legacy_tasks}
    tasks_df = pd.DataFrame({"task_index": list(task_index_to_text.keys())}, index=pd.Index(task_index_to_text.values(), name="task"))
    write_tasks(tasks_df, target_root)
    return tasks_df, task_index_to_text


def _convert_data_files(
    *,
    source_root: Path,
    target_root: Path,
    arm_unit: str,
    gripper_scale: float,
) -> tuple[list[dict[str, int]], int, dict[str, Any]]:
    source_paths = sorted((source_root / "data").glob("*/*.parquet"))
    if not source_paths:
        raise FileNotFoundError(f"No parquet episode files found under {(source_root / 'data')}.")

    episode_data_rows: list[dict[str, int]] = []
    dataset_cursor = 0
    analysis = _ImportAnalysisAccumulator()

    for file_counter, source_path in enumerate(source_paths):
        chunk_index = file_counter // DEFAULT_CHUNK_SIZE
        file_index = file_counter % DEFAULT_CHUNK_SIZE

        df = pd.read_parquet(source_path)
        if SOURCE_STATE_KEY not in df.columns:
            raise KeyError(f"Missing required source column {SOURCE_STATE_KEY!r} in {source_path}.")
        if ACTION_KEY not in df.columns:
            raise KeyError(f"Missing required source column {ACTION_KEY!r} in {source_path}.")

        converted_columns: dict[str, Any] = {}
        action = _transform_vector_array(
            np.asarray(df[ACTION_KEY].tolist(), dtype=np.float32),
            arm_unit=arm_unit,
            gripper_scale=gripper_scale,
        )
        state = _transform_vector_array(
            np.asarray(df[SOURCE_STATE_KEY].tolist(), dtype=np.float32),
            arm_unit=arm_unit,
            gripper_scale=gripper_scale,
        )
        analysis.action_step_stats.add(action)
        analysis.state_step_stats.add(state)
        analysis.action_vs_state_abs_sum += float(np.sum(np.abs(action - state), dtype=np.float64))
        analysis.action_vs_state_count += int(action.size)
        if len(action) > 1:
            analysis.action_vs_next_state_abs_sum += float(np.sum(np.abs(action[:-1] - state[1:]), dtype=np.float64))
            analysis.action_vs_next_state_count += int(action[:-1].size)
        converted_columns[ACTION_KEY] = [row.tolist() for row in action]
        converted_columns[TARGET_STATE_KEY] = [row.tolist() for row in state]

        for column in df.columns:
            if column in {ACTION_KEY, SOURCE_STATE_KEY}:
                continue
            converted_columns[column] = df[column]

        out_df = pd.DataFrame(converted_columns)
        if "timestamp" in out_df.columns:
            out_df["timestamp"] = out_df["timestamp"].astype(np.float32)
        for int_column in ("frame_index", "episode_index", "index", "task_index"):
            if int_column in out_df.columns:
                out_df[int_column] = out_df[int_column].astype(np.int64)

        episode_length = int(len(out_df))
        if episode_length <= 0:
            raise ValueError(f"Episode parquet has no rows: {source_path}")

        output_path = target_root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(output_path, index=False)

        episode_index = int(out_df["episode_index"].iloc[0]) if "episode_index" in out_df.columns else file_counter
        episode_data_rows.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": dataset_cursor,
                "dataset_to_index": dataset_cursor + episode_length,
            }
        )
        dataset_cursor += episode_length

    return episode_data_rows, dataset_cursor, _finalize_import_analysis(analysis)


def _convert_videos(
    *,
    source_root: Path,
    target_root: Path,
    camera_map: dict[str, str],
    total_episodes: int,
) -> list[dict[str, Any]]:
    video_rows: list[dict[str, Any]] = [{"episode_index": ep_idx} for ep_idx in range(total_episodes)]

    for source_key, target_key in camera_map.items():
        source_paths = sorted((source_root / "videos").glob(f"*/{source_key}/episode_*.mp4"))
        if len(source_paths) != total_episodes:
            raise ValueError(
                f"Camera {source_key!r} has {len(source_paths)} video episodes, expected {total_episodes}."
            )

        for file_counter, source_path in enumerate(source_paths):
            chunk_index = file_counter // DEFAULT_CHUNK_SIZE
            file_index = file_counter % DEFAULT_CHUNK_SIZE
            output_path = target_root / "videos" / target_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)

            duration_s = float(get_video_duration_in_s(source_path))
            video_rows[file_counter].update(
                {
                    f"videos/{target_key}/chunk_index": chunk_index,
                    f"videos/{target_key}/file_index": file_index,
                    f"videos/{target_key}/from_timestamp": 0.0,
                    f"videos/{target_key}/to_timestamp": duration_s,
                }
            )

    return video_rows


def _convert_episode_stats(
    *,
    source_root: Path,
    arm_unit: str,
    gripper_scale: float,
    camera_map: dict[str, str],
) -> list[dict[str, Any]]:
    legacy_rows = sorted(_load_jsonl(source_root / "meta" / "episodes_stats.jsonl"), key=lambda item: int(item["episode_index"]))
    converted_rows: list[dict[str, Any]] = []

    for row in legacy_rows:
        episode_index = int(row["episode_index"])
        stats = row.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid stats payload for episode {episode_index}.")

        converted_stats: dict[str, Any] = {}
        for key, value in stats.items():
            if key == SOURCE_STATE_KEY:
                converted_stats[TARGET_STATE_KEY] = _transform_vector_stats(
                    value,
                    arm_unit=arm_unit,
                    gripper_scale=gripper_scale,
                )
            elif key == ACTION_KEY:
                converted_stats[ACTION_KEY] = _transform_vector_stats(
                    value,
                    arm_unit=arm_unit,
                    gripper_scale=gripper_scale,
                )
            elif key in camera_map:
                converted_stats[camera_map[key]] = value
            else:
                converted_stats[key] = value

        converted_rows.append({"episode_index": episode_index, "stats": converted_stats})

    return converted_rows


def _write_episode_metadata(
    *,
    target_root: Path,
    legacy_episodes: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    if not (len(legacy_episodes) == len(data_rows) == len(video_rows) == len(episode_stats)):
        raise ValueError(
            "Episode metadata counts do not match: "
            f"episodes={len(legacy_episodes)} data={len(data_rows)} videos={len(video_rows)} stats={len(episode_stats)}"
        )

    episode_records: list[dict[str, Any]] = []
    aggregate_input: list[dict[str, dict[str, np.ndarray]]] = []

    for legacy_ep, data_row, video_row, stats_row in zip(legacy_episodes, data_rows, video_rows, episode_stats, strict=True):
        episode_index = int(legacy_ep["episode_index"])
        ids = {episode_index, int(data_row["episode_index"]), int(video_row["episode_index"]), int(stats_row["episode_index"])}
        if len(ids) != 1:
            raise ValueError(f"Episode index mismatch while merging metadata: {sorted(ids)}")

        stats_dict = stats_row["stats"]
        aggregate_input.append(cast_stats_to_numpy(stats_dict))

        record = {
            "episode_index": episode_index,
            "tasks": legacy_ep.get("tasks", []),
            "length": int(legacy_ep["length"]),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            **data_row,
            **video_row,
            **flatten_dict({"stats": stats_dict}),
        }
        episode_records.append(record)

    episodes_ds = Dataset.from_list(episode_records)
    write_episodes(episodes_ds, target_root)
    return aggregate_stats(aggregate_input)


def _write_import_summary(
    *,
    target_root: Path,
    source_root: Path,
    fps: int,
    arm_unit: str,
    gripper_unit: str,
    gripper_scale: float,
    camera_map: dict[str, str],
    import_analysis: dict[str, Any],
) -> None:
    summary = {
        "source_root": str(source_root),
        "conversion": {
            "codebase_version": CODEBASE_VERSION,
            "arm_unit_in": arm_unit,
            "arm_unit_out": "deg",
            "gripper_unit_in": gripper_unit,
            "gripper_unit_out": "mm",
            "gripper_scale_applied": gripper_scale,
            "fps": int(fps),
            "camera_map": camera_map,
        },
        "analysis": import_analysis,
    }
    write_json(summary, target_root / "meta" / IMPORT_SUMMARY_FILENAME)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an imported ALOHA-style dataset into Songling format.")
    parser.add_argument("--input-root", type=Path, required=True, help="Source dataset root containing meta/, data/, videos/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Target dataset root to create.")
    parser.add_argument("--fps", type=int, default=None, help="Override output FPS. Defaults to source dataset FPS.")
    parser.add_argument(
        "--robot-type",
        default="bi_songling_follower",
        help="robot_type to write into the converted dataset metadata.",
    )
    parser.add_argument(
        "--arm-unit",
        choices=("rad", "deg"),
        default="rad",
        help="Unit used by the source arm joints.",
    )
    parser.add_argument(
        "--gripper-unit",
        choices=("m", "mm", "normalized_70", "normalized_100"),
        default="m",
        help="Unit used by the source gripper values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-root if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = args.input_root.expanduser().resolve()
    target_root = args.output_root.expanduser().resolve()
    _ensure_local_hf_cache(target_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {source_root}")
    if target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {target_root}. Use --overwrite to replace it.")
        shutil.rmtree(target_root)

    source_info = _load_json(source_root / "meta" / "info.json")
    source_version = str(source_info.get("codebase_version", ""))
    if source_version != "v2.1":
        raise ValueError(f"Expected a v2.1 source dataset, got {source_version!r}.")

    fps = int(args.fps if args.fps is not None else source_info.get("fps", 30))
    gripper_scale = _resolve_gripper_scale(args.gripper_unit)
    camera_map = dict(DEFAULT_CAMERA_MAP)

    logger.info("Converting dataset from %s to %s", source_root, target_root)
    logger.info(
        "Numeric conversion: arm %s -> deg, gripper %s -> mm (scale=%.3f), fps=%d",
        args.arm_unit,
        args.gripper_unit,
        gripper_scale,
        fps,
    )

    target_root.mkdir(parents=True, exist_ok=True)

    tasks_df, _ = _convert_tasks(source_root, target_root)
    legacy_episodes = sorted(_load_jsonl(source_root / "meta" / "episodes.jsonl"), key=lambda item: int(item["episode_index"]))
    data_rows, total_frames, import_analysis = _convert_data_files(
        source_root=source_root,
        target_root=target_root,
        arm_unit=args.arm_unit,
        gripper_scale=gripper_scale,
    )
    logger.info(
        "Imported action semantics: kind=%s mae_vs_state=%.6f mae_vs_next_state=%.6f preferred_replay=%s",
        import_analysis["action_semantics"]["kind"],
        float(import_analysis["action_semantics"]["mean_abs_error_vs_observation_state"]),
        float(import_analysis["action_semantics"]["mean_abs_error_vs_next_observation_state"]),
        import_analysis["replay"]["preferred_control_source"],
    )
    logger.info(
        "Imported step stats: action joint_p95=%.3f deg gripper_p95=%.3f mm | state joint_p95=%.3f deg gripper_p95=%.3f mm",
        float(import_analysis["step_stats"][ACTION_KEY]["joint_p95_deg"]),
        float(import_analysis["step_stats"][ACTION_KEY]["gripper_p95_mm"]),
        float(import_analysis["step_stats"][TARGET_STATE_KEY]["joint_p95_deg"]),
        float(import_analysis["step_stats"][TARGET_STATE_KEY]["gripper_p95_mm"]),
    )
    total_episodes = len(legacy_episodes)
    if len(data_rows) != total_episodes:
        raise ValueError(f"Data files count {len(data_rows)} does not match legacy episodes {total_episodes}.")

    video_rows = _convert_videos(
        source_root=source_root,
        target_root=target_root,
        camera_map=camera_map,
        total_episodes=total_episodes,
    )
    episode_stats = _convert_episode_stats(
        source_root=source_root,
        arm_unit=args.arm_unit,
        gripper_scale=gripper_scale,
        camera_map=camera_map,
    )
    aggregated_stats = _write_episode_metadata(
        target_root=target_root,
        legacy_episodes=legacy_episodes,
        data_rows=data_rows,
        video_rows=video_rows,
        episode_stats=episode_stats,
    )
    write_stats(aggregated_stats, target_root)

    features = _build_target_features(source_info=source_info, fps=fps, camera_map=camera_map)
    info = create_empty_dataset_info(
        codebase_version=CODEBASE_VERSION,
        fps=fps,
        features=features,
        use_videos=True,
        robot_type=str(args.robot_type),
    )
    info["total_episodes"] = total_episodes
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = int(len(tasks_df))
    write_info(info, target_root)

    _write_import_summary(
        target_root=target_root,
        source_root=source_root,
        fps=fps,
        arm_unit=args.arm_unit,
        gripper_unit=args.gripper_unit,
        gripper_scale=gripper_scale,
        camera_map=camera_map,
        import_analysis=import_analysis,
    )

    logger.info("Conversion finished: %s", target_root)

    # Basic smoke-check: ensure the converted dataset can be opened with the current codebase.
    dataset = LeRobotDataset(repo_id=f"local/{target_root.name}", root=target_root, download_videos=False)
    logger.info(
        "Smoke check passed: episodes=%d frames=%d fps=%d robot_type=%s",
        dataset.num_episodes,
        dataset.num_frames,
        dataset.fps,
        dataset.meta.robot_type,
    )


if __name__ == "__main__":
    init_logging()
    main()
