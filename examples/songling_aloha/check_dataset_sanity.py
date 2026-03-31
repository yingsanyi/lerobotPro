#!/usr/bin/env python

"""Sanity-check a newly recorded Songling ALOHA dataset.

This checker validates the dataset against two layers of assumptions:

1. Songling / Piper-like CAN protocol storage convention in this repo:
   - arm joints are stored in degrees
   - gripper is stored in mm
2. Mobile ALOHA-style policy convention after projection:
   - arm joints should be interpreted in radians
   - gripper should be normalized into [0, 1]

It also inspects the sidecar action-source summary produced by
`record_raw_can_dataset.py` so we can tell whether dataset actions came from
real control frames or from observation fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_JOINTS_PER_SIDE = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
EXPECTED_FEATURE_NAMES = [
    f"{side}_{joint_name}.pos" for side in ("left", "right") for joint_name in EXPECTED_JOINTS_PER_SIDE
]
ALOHA_JOINT_LABELS = {
    "joint_1": "waist",
    "joint_2": "shoulder",
    "joint_3": "elbow",
    "joint_4": "forearm_roll",
    "joint_5": "wrist_angle",
    "joint_6": "wrist_rotate",
    "gripper": "gripper",
}
CAPTURE_SUMMARY_FILENAME = "songling_recording_summary.json"
GRIPPER_RANGE_CANDIDATES_MM = (70.0, 100.0)
SANITY_PROFILE_CHOICES = ("auto", "songling_70", "songling_100")
SANITY_PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "songling_70": {
        "gripper_max_mm": 70.0,
        "joint_margin_deg": 3.0,
        "gripper_margin_mm": 5.0,
    },
    "songling_100": {
        "gripper_max_mm": 100.0,
        "joint_margin_deg": 5.0,
        "gripper_margin_mm": 10.0,
    },
}
SONGLING_JOINT_LIMITS_DEG = {
    "joint_1": (-150.0, 150.0),
    "joint_2": (0.0, 180.0),
    "joint_3": (-170.0, 0.0),
    "joint_4": (-100.0, 100.0),
    "joint_5": (-70.0, 70.0),
    "joint_6": (-120.0, 120.0),
}


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


def _is_writable_dir_or_parent(path: Path) -> bool:
    candidate = path.expanduser()
    probe = candidate
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe.exists() and os.access(probe, os.W_OK)


def _ensure_hf_datasets_cache(root: Path) -> None:
    home_fallback = root / ".hf_home"
    home_fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home_fallback)

    env_value = os.getenv("HF_DATASETS_CACHE")
    if env_value is not None and _is_writable_dir_or_parent(Path(env_value)):
        return

    fallback = home_fallback / "datasets"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(fallback)


def _parse_episode_csv(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _resolve_repo_id(repo_id: str | None, root: Path) -> str:
    if repo_id is not None and repo_id.strip():
        return repo_id.strip()
    return f"local/{root.name}"


def _feature_names(dataset: Any, key: str) -> list[str]:
    feature = dataset.features.get(key)
    if feature is None:
        raise ValueError(f"Dataset is missing required feature '{key}'.")
    names = feature.get("names")
    if not isinstance(names, list):
        raise ValueError(f"Dataset feature '{key}' does not expose named dimensions.")
    return [str(name) for name in names]


def _feature_matrix(dataset: Any, key: str) -> np.ndarray:
    hf_dataset = dataset.hf_dataset.with_format(None)
    return np.asarray(hf_dataset[key], dtype=np.float32)


def _feature_vector_names_ok(names: list[str]) -> bool:
    return names == EXPECTED_FEATURE_NAMES


def _extract_joint_name(feature_name: str) -> tuple[str, str]:
    base_name = feature_name.removesuffix(".pos")
    if base_name.startswith("left_"):
        return "left", base_name.removeprefix("left_")
    if base_name.startswith("right_"):
        return "right", base_name.removeprefix("right_")
    raise ValueError(f"Unexpected feature name format: {feature_name}")


def _project_to_mobile_aloha(values: np.ndarray, names: list[str], gripper_max_mm: float) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for idx, name in enumerate(names):
        _, joint_name = _extract_joint_name(name)
        if joint_name == "gripper":
            out[:, idx] = out[:, idx] / gripper_max_mm
        else:
            out[:, idx] = np.deg2rad(out[:, idx])
    return out


def _limit_tuple_for_joint(joint_name: str, gripper_max_mm: float) -> tuple[float, float]:
    if joint_name == "gripper":
        return (0.0, float(gripper_max_mm))
    return tuple(float(v) for v in SONGLING_JOINT_LIMITS_DEG[joint_name])


def _load_capture_summary(root: Path) -> dict[str, Any]:
    summary_path = root / "meta" / CAPTURE_SUMMARY_FILENAME
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_episode_indices(dataset: Any) -> np.ndarray:
    hf_dataset = dataset.hf_dataset.with_format(None)
    return np.asarray(hf_dataset["episode_index"], dtype=np.int64)


def _summarize_feature_ranges(
    *,
    values: np.ndarray,
    names: list[str],
    gripper_max_mm: float,
    joint_margin_deg: float,
    gripper_margin_mm: float,
    errors: list[str],
) -> list[str]:
    lines: list[str] = []
    for idx, name in enumerate(names):
        _, joint_name = _extract_joint_name(name)
        lower, upper = _limit_tuple_for_joint(joint_name, gripper_max_mm)
        margin = gripper_margin_mm if joint_name == "gripper" else joint_margin_deg
        value_min = float(np.min(values[:, idx]))
        value_max = float(np.max(values[:, idx]))
        aloha_label = ALOHA_JOINT_LABELS[joint_name]
        lines.append(
            f"- {name} ({aloha_label}): [{value_min:.3f}, {value_max:.3f}] "
            f"expected [{lower:.3f}, {upper:.3f}]"
        )
        if value_min < (lower - margin) or value_max > (upper + margin):
            errors.append(
                f"{name} exceeded protocol range: observed [{value_min:.3f}, {value_max:.3f}], "
                f"expected [{lower:.3f}, {upper:.3f}] with margin {margin:.3f}."
            )
    return lines


def _summarize_projected_ranges(
    *,
    projected_values: np.ndarray,
    names: list[str],
    gripper_norm_margin: float,
    warnings: list[str],
) -> list[str]:
    lines: list[str] = []
    for idx, name in enumerate(names):
        _, joint_name = _extract_joint_name(name)
        value_min = float(np.min(projected_values[:, idx]))
        value_max = float(np.max(projected_values[:, idx]))
        if joint_name == "gripper":
            lines.append(f"- {name}: normalized gripper range [{value_min:.3f}, {value_max:.3f}]")
            if value_min < -gripper_norm_margin or value_max > 1.0 + gripper_norm_margin:
                warnings.append(
                    f"{name} projects outside mobile-aloha-style normalized gripper range [0, 1]: "
                    f"observed [{value_min:.3f}, {value_max:.3f}]."
                )
        else:
            lines.append(f"- {name}: projected rad range [{value_min:.3f}, {value_max:.3f}]")
    return lines


def _resolve_sanity_profile(
    *,
    requested_profile: str,
    gripper_max_mm: float | None,
    joint_margin_deg: float | None,
    gripper_margin_mm: float | None,
    state_values: np.ndarray,
    action_values: np.ndarray,
    names: list[str],
) -> tuple[str, float, float, float]:
    gripper_indices = [idx for idx, name in enumerate(names) if _extract_joint_name(name)[1] == "gripper"]
    if not gripper_indices:
        raise ValueError("No gripper feature found in dataset vectors.")

    observed_gripper_max = float(
        max(
            np.max(state_values[:, gripper_indices]),
            np.max(action_values[:, gripper_indices]),
        )
    )

    if requested_profile != "auto":
        defaults = SANITY_PROFILE_DEFAULTS[requested_profile]
        resolved_gripper = float(gripper_max_mm if gripper_max_mm is not None else defaults["gripper_max_mm"])
        resolved_joint_margin = float(joint_margin_deg if joint_margin_deg is not None else defaults["joint_margin_deg"])
        resolved_gripper_margin = float(
            gripper_margin_mm if gripper_margin_mm is not None else defaults["gripper_margin_mm"]
        )
        return requested_profile, resolved_gripper, resolved_joint_margin, resolved_gripper_margin

    auto_gripper = float(gripper_max_mm) if gripper_max_mm is not None else float(max(GRIPPER_RANGE_CANDIDATES_MM))
    if gripper_max_mm is None:
        for candidate in sorted(GRIPPER_RANGE_CANDIDATES_MM):
            # Allow a small slack to absorb encoder/offset noise near max opening.
            if observed_gripper_max <= float(candidate) + 2.0:
                auto_gripper = float(candidate)
                break

    auto_profile = "songling_100" if auto_gripper >= 85.0 else "songling_70"
    defaults = SANITY_PROFILE_DEFAULTS[auto_profile]
    resolved_joint_margin = float(joint_margin_deg if joint_margin_deg is not None else defaults["joint_margin_deg"])
    resolved_gripper_margin = float(gripper_margin_mm if gripper_margin_mm is not None else defaults["gripper_margin_mm"])
    return f"auto->{auto_profile}", float(auto_gripper), resolved_joint_margin, resolved_gripper_margin


def _filter_summary_episodes(summary: dict[str, Any], selected_episodes: set[int] | None) -> list[dict[str, Any]]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        return []

    filtered: list[dict[str, Any]] = []
    for entry in episodes:
        if not isinstance(entry, dict):
            continue
        try:
            episode_index = int(entry["episode_index"])
        except Exception:
            continue
        if selected_episodes is not None and episode_index not in selected_episodes:
            continue
        filtered.append(entry)
    return sorted(filtered, key=lambda item: int(item["episode_index"]))


def _summarize_action_source(
    *,
    action_values: np.ndarray,
    state_values: np.ndarray,
    episode_indices: np.ndarray,
    summary: dict[str, Any],
    selected_episodes: set[int] | None,
    min_control_ratio: float,
    heuristic_atol: float,
    warnings: list[str],
    errors: list[str],
) -> list[str]:
    def _loop_ranges_preview(value: Any, *, max_items: int = 6) -> tuple[int, list[str]]:
        if not isinstance(value, list):
            return 0, []
        total = 0
        preview: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                start = int(item.get("start_loop", 0))
                end = int(item.get("end_loop", start))
            except Exception:
                continue
            if end < start:
                start, end = end, start
            total += (end - start) + 1
            if len(preview) < max_items:
                token = str(start) if start == end else f"{start}-{end}"
                reason = str(item.get("reason", "unknown"))
                preview.append(f"{token}({reason})")
        return total, preview

    lines: list[str] = []
    episode_entries = _filter_summary_episodes(summary, selected_episodes)
    if episode_entries:
        lines.append(f"- source: {CAPTURE_SUMMARY_FILENAME}")
        capture_policy = summary.get("capture_policy")
        if isinstance(capture_policy, dict):
            lines.append(
                "- policy: "
                f"missing_label_policy={capture_policy.get('missing_label_policy', 'unknown')}, "
                f"command_freshness_s={capture_policy.get('command_freshness_s', 'unknown')}"
            )
        for entry in episode_entries:
            episode_index = int(entry["episode_index"])
            frames = int(entry.get("frames", 0))
            left_control = int(entry.get("left_control_frames", 0))
            right_control = int(entry.get("right_control_frames", 0))
            left_ratio = float(entry.get("left_control_ratio", 0.0))
            right_ratio = float(entry.get("right_control_ratio", 0.0))
            dropped_missing = int(entry.get("dropped_missing_control_frames", 0))
            dropped_loops_total, dropped_preview = _loop_ranges_preview(entry.get("dropped_loop_ranges"))
            lines.append(
                f"- episode {episode_index}: "
                f"left_control={left_control}/{frames} ({left_ratio:.1%}), "
                f"right_control={right_control}/{frames} ({right_ratio:.1%}), "
                f"dropped_missing_control={dropped_missing}"
            )
            if dropped_preview:
                preview_text = ", ".join(dropped_preview)
                lines.append(
                    f"- episode {episode_index} dropped_loop_ranges: total={dropped_loops_total}, "
                    f"ranges={len(entry.get('dropped_loop_ranges', []))}, preview={preview_text}"
                )
            if left_ratio < min_control_ratio:
                errors.append(
                    f"Episode {episode_index} left side control ratio is {left_ratio:.1%}, "
                    f"below threshold {min_control_ratio:.1%}."
                )
            if right_ratio < min_control_ratio:
                errors.append(
                    f"Episode {episode_index} right side control ratio is {right_ratio:.1%}, "
                    f"below threshold {min_control_ratio:.1%}."
                )
        return lines

    lines.append("- source: heuristic fallback (summary sidecar missing)")
    for side, col_slice in (("left", slice(0, 7)), ("right", slice(7, 14))):
        close_mask = np.all(
            np.isclose(
                action_values[:, col_slice],
                state_values[:, col_slice],
                atol=heuristic_atol,
                rtol=0.0,
            ),
            axis=1,
        )
        ratio = float(np.mean(close_mask)) if len(close_mask) > 0 else 0.0
        lines.append(f"- {side}: action==state heuristic ratio {ratio:.1%}")
        if ratio >= min_control_ratio:
            warnings.append(
                f"{side} action is almost always equal to observation.state ({ratio:.1%}) "
                "without a recording summary; this often means observation fallback was used."
            )
    unique_episodes = sorted({int(v) for v in episode_indices.tolist()})
    lines.append(
        f"- heuristic covers episodes {unique_episodes}; rerun recording with the updated recorder to get exact ratios."
    )
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check a Songling ALOHA dataset.")
    parser.add_argument("--dataset.root", "--dataset-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset.repo_id", "--dataset-repo-id", dest="dataset_repo_id", default=None)
    parser.add_argument(
        "--profile",
        choices=SANITY_PROFILE_CHOICES,
        default="auto",
        help=(
            "Validation profile preset. "
            "'auto' infers 70mm/100mm gripper profile from data and applies matching default margins."
        ),
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Optional comma-separated episode indices to inspect. Defaults to all episodes.",
    )
    parser.add_argument(
        "--gripper-max-mm",
        type=float,
        default=None,
        help="Override expected max gripper stroke in mm. If unset, resolved from --profile.",
    )
    parser.add_argument(
        "--joint-margin-deg",
        type=float,
        default=None,
        help="Override allowed arm-joint range slack in degrees before flagging an error.",
    )
    parser.add_argument(
        "--gripper-margin-mm",
        type=float,
        default=None,
        help="Override allowed gripper range slack in mm before flagging an error.",
    )
    parser.add_argument(
        "--gripper-norm-margin",
        type=float,
        default=0.05,
        help="Allowed slack around projected mobile-aloha normalized gripper range [0, 1].",
    )
    parser.add_argument(
        "--min-control-ratio",
        type=float,
        default=0.9,
        help="Minimum per-side control-frame ratio required when action-source summary is available.",
    )
    parser.add_argument(
        "--heuristic-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for the fallback action==state heuristic.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    _patch_multiprocess_resource_tracker()
    _ensure_hf_datasets_cache(dataset_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    selected_episodes = _parse_episode_csv(args.episodes)
    repo_id = _resolve_repo_id(args.dataset_repo_id, dataset_root)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=selected_episodes,
        download_videos=False,
    )

    state_names = _feature_names(dataset, "observation.state")
    action_names = _feature_names(dataset, "action")
    if not _feature_vector_names_ok(state_names):
        raise ValueError(
            "Unexpected observation.state names.\n"
            f"Expected: {EXPECTED_FEATURE_NAMES}\n"
            f"Got:      {state_names}"
        )
    if not _feature_vector_names_ok(action_names):
        raise ValueError(
            "Unexpected action names.\n"
            f"Expected: {EXPECTED_FEATURE_NAMES}\n"
            f"Got:      {action_names}"
        )

    state_values = _feature_matrix(dataset, "observation.state")
    action_values = _feature_matrix(dataset, "action")
    if state_values.ndim != 2 or state_values.shape[1] != len(EXPECTED_FEATURE_NAMES):
        raise ValueError(f"Expected observation.state shape [N, 14], got {state_values.shape}.")
    if action_values.ndim != 2 or action_values.shape[1] != len(EXPECTED_FEATURE_NAMES):
        raise ValueError(f"Expected action shape [N, 14], got {action_values.shape}.")

    (
        resolved_profile,
        resolved_gripper_max_mm,
        resolved_joint_margin_deg,
        resolved_gripper_margin_mm,
    ) = _resolve_sanity_profile(
        requested_profile=str(args.profile),
        gripper_max_mm=args.gripper_max_mm,
        joint_margin_deg=args.joint_margin_deg,
        gripper_margin_mm=args.gripper_margin_mm,
        state_values=state_values,
        action_values=action_values,
        names=state_names,
    )

    episode_indices = _get_episode_indices(dataset)
    summary = _load_capture_summary(dataset_root)
    projected_state = _project_to_mobile_aloha(state_values, state_names, resolved_gripper_max_mm)
    projected_action = _project_to_mobile_aloha(action_values, action_names, resolved_gripper_max_mm)

    errors: list[str] = []
    warnings: list[str] = []

    protocol_state_lines = _summarize_feature_ranges(
        values=state_values,
        names=state_names,
        gripper_max_mm=resolved_gripper_max_mm,
        joint_margin_deg=resolved_joint_margin_deg,
        gripper_margin_mm=resolved_gripper_margin_mm,
        errors=errors,
    )
    protocol_action_lines = _summarize_feature_ranges(
        values=action_values,
        names=action_names,
        gripper_max_mm=resolved_gripper_max_mm,
        joint_margin_deg=resolved_joint_margin_deg,
        gripper_margin_mm=resolved_gripper_margin_mm,
        errors=errors,
    )
    aloha_state_lines = _summarize_projected_ranges(
        projected_values=projected_state,
        names=state_names,
        gripper_norm_margin=args.gripper_norm_margin,
        warnings=warnings,
    )
    aloha_action_lines = _summarize_projected_ranges(
        projected_values=projected_action,
        names=action_names,
        gripper_norm_margin=args.gripper_norm_margin,
        warnings=warnings,
    )
    action_source_lines = _summarize_action_source(
        action_values=action_values,
        state_values=state_values,
        episode_indices=episode_indices,
        summary=summary,
        selected_episodes=None if selected_episodes is None else set(selected_episodes),
        min_control_ratio=args.min_control_ratio,
        heuristic_atol=args.heuristic_atol,
        warnings=warnings,
        errors=errors,
    )

    print("Dataset")
    print(f"- repo_id: {repo_id}")
    print(f"- root: {dataset_root}")
    print(f"- episodes: {dataset.num_episodes}")
    print(f"- frames: {dataset.num_frames}")
    print(f"- sanity_profile: {resolved_profile}")
    print(
        f"- gripper_max_mm: {resolved_gripper_max_mm:.1f} "
        f"(common Piper candidates: {', '.join(str(int(v)) for v in GRIPPER_RANGE_CANDIDATES_MM)})"
    )
    print(f"- joint_margin_deg: {resolved_joint_margin_deg:.1f}")
    print(f"- gripper_margin_mm: {resolved_gripper_margin_mm:.1f}")

    print("\nProtocol-Native Ranges: observation.state")
    for line in protocol_state_lines:
        print(line)

    print("\nProtocol-Native Ranges: action")
    for line in protocol_action_lines:
        print(line)

    print("\nMobile-ALOHA Projection: observation.state")
    for line in aloha_state_lines:
        print(line)

    print("\nMobile-ALOHA Projection: action")
    for line in aloha_action_lines:
        print(line)

    print("\nAction Source")
    for line in action_source_lines:
        print(line)

    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nResult: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nResult: PASS")


if __name__ == "__main__":
    main()
