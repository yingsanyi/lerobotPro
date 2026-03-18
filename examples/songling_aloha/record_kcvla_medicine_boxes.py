#!/usr/bin/env python

"""Record a SongLing KC-VLA dataset with configurable instruction and keyword metadata.

This wrapper keeps the standard SongLing recording flow, then annotates the
newly recorded episodes with:

- English instruction (`task` / `dataset.single_task`)
- Chinese `keyword_text`
- Chinese `counterfactual_keyword_text`

It is parameterized instead of task-hardcoded. You can either pass complete
strings or build keyword sets from repeated `--keyword-item-zh` flags.

Examples:
    python examples/songling_aloha/record_kcvla_medicine_boxes.py \
      --dataset.repo_id your_name/songling_kcvla_medicine_boxes \
      --instruction "Place the painkiller in the left box and place the vitamin in the right box." \
      --keyword-item-zh 止痛药 \
      --keyword-item-zh 维生素 \
      --counterfactual-mode manual \
      --counterfactual-keyword-item-zh 维生素 \
      --counterfactual-keyword-item-zh 止痛药 \
      --dataset.num_episodes 20 \
      --dataset.episode_time_s 45 \
      --display_data=true

    python examples/songling_aloha/record_kcvla_medicine_boxes.py \
      --dataset.repo_id your_name/songling_kcvla_demo \
      --instruction "Place the screwdriver in the left tray and place the pliers in the right tray." \
      --keyword-text "螺丝刀, 钳子" \
      --counterfactual-mode swap
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lerobot.datasets.utils import (
    DATA_DIR,
    get_hf_features_from_features,
    load_info,
    to_parquet_with_hf_images,
    write_info,
)
from lerobot.utils.constants import COUNTERFACTUAL_KEYWORD_TEXT, KEYWORD_TEXT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "examples" / "songling_aloha" / "teleop.yaml"
DEFAULT_RECORD_COMPAT_PATH = PROJECT_ROOT / "examples" / "songling_aloha" / "record_compat.py"
DEFAULT_DATASET_ROOT_BASE = PROJECT_ROOT / "outputs" / "songling_aloha" / "kcvla_recordings"
DEFAULT_REPO_ID = "songling_local/kcvla_songling"
DEFAULT_STRING_FEATURE = {"dtype": "string", "shape": (1,), "names": None}


def _timestamped_default_root() -> Path:
    run_name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    return DEFAULT_DATASET_ROOT_BASE / run_name


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
                raise ValueError(f"Missing value after {matched_name}.")
            continue

        value = arg.split("=", 1)[1]
        i += 1
    return value, stripped


def _has_flag(args: list[str], names: tuple[str, ...]) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args for name in names)


def _clean_text(item: Any) -> str:
    if item is None:
        return ""
    return str(item).replace("\n", " ").replace("\t", " ").strip()


def _split_keyword_items(value: Any) -> list[str]:
    value = _clean_text(value)
    if not value:
        return []
    for delimiter in ("，", ";", "；", "|", "\n"):
        value = value.replace(delimiter, ",")
    return [item for item in (_clean_text(part) for part in value.split(",")) if item]


def _join_keyword_items(items: list[str], *, separator: str) -> str:
    cleaned_items = [_clean_text(item) for item in items if _clean_text(item)]
    return separator.join(cleaned_items)


def _resolve_keyword_text(*, full_text: str | None, items: list[str], separator: str, field_name: str) -> str:
    if full_text is not None and _clean_text(full_text):
        return _join_keyword_items(_split_keyword_items(full_text), separator=separator)
    if items:
        return _join_keyword_items(items, separator=separator)
    raise ValueError(f"Missing {field_name}. Provide the full text or at least one repeated item flag.")


def _build_swap_counterfactual(keyword_text: str, *, separator: str) -> str:
    items = _split_keyword_items(keyword_text)
    if len(items) < 2 or len(set(items)) < 2:
        raise ValueError(
            "counterfactual_mode=swap requires at least two distinct keywords in keyword_text or keyword-item-zh."
        )
    swapped = items[1:] + items[:1]
    return _join_keyword_items(swapped, separator=separator)


def _ensure_string_feature(info: dict, feature_name: str) -> bool:
    features = info.setdefault("features", {})
    feature = features.get(feature_name)
    if feature is None:
        features[feature_name] = dict(DEFAULT_STRING_FEATURE)
        return True

    if feature.get("dtype") != "string":
        raise ValueError(f"Feature '{feature_name}' must have dtype='string', got {feature.get('dtype')}")
    return False


def _load_episode_indices(root: Path) -> set[int]:
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.exists():
        return set()

    episode_indices: set[int] = set()
    for parquet_path in sorted(episodes_dir.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index"])
        episode_indices.update(int(value) for value in df["episode_index"].dropna().unique())
    return episode_indices


def annotate_dataset_keywords(
    root: str | Path,
    *,
    episode_indices: set[int],
    keyword_text: str,
    counterfactual_keyword_text: str,
) -> dict[str, int]:
    dataset_root = Path(root)
    parquet_paths = sorted((dataset_root / DATA_DIR).glob("*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet files found under {dataset_root / DATA_DIR}")

    info = load_info(dataset_root)
    schema_changed = False
    schema_changed |= _ensure_string_feature(info, KEYWORD_TEXT)
    schema_changed |= _ensure_string_feature(info, COUNTERFACTUAL_KEYWORD_TEXT)
    if schema_changed:
        write_info(info, dataset_root)
        info = load_info(dataset_root)

    hf_features = get_hf_features_from_features(info["features"])
    summary = {"updated_rows": 0, "updated_files": 0, "episode_count": len(episode_indices)}

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path).reset_index(drop=True)
        if "episode_index" not in df.columns:
            raise ValueError(f"Column 'episode_index' not found in {parquet_path}")

        mask = df["episode_index"].isin(episode_indices)
        if not mask.any():
            continue

        if KEYWORD_TEXT not in df.columns:
            df[KEYWORD_TEXT] = ""
        if COUNTERFACTUAL_KEYWORD_TEXT not in df.columns:
            df[COUNTERFACTUAL_KEYWORD_TEXT] = ""

        df.loc[mask, KEYWORD_TEXT] = keyword_text
        df.loc[mask, COUNTERFACTUAL_KEYWORD_TEXT] = counterfactual_keyword_text
        to_parquet_with_hf_images(df, parquet_path, features=hf_features)

        summary["updated_rows"] += int(mask.sum())
        summary["updated_files"] += 1

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", "--config_path", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dataset.repo_id",
        "--dataset-repo-id",
        dest="dataset_repo_id",
        default=DEFAULT_REPO_ID,
        help="Repo id written into the recorded LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset.root",
        "--dataset-root",
        dest="dataset_root",
        type=Path,
        default=None,
        help="Local dataset root. Defaults to a timestamped folder under outputs/songling_aloha/kcvla_recordings/.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="English task instruction written to dataset.single_task. Required unless --dataset.single_task is passed through.",
    )
    parser.add_argument(
        "--keyword-text",
        type=str,
        default=None,
        help="Complete Chinese keyword_text string, e.g. '止痛药, 维生素'.",
    )
    parser.add_argument(
        "--keyword-item-zh",
        action="append",
        default=[],
        help="Repeated Chinese keyword item. Can be passed multiple times instead of --keyword-text.",
    )
    parser.add_argument(
        "--keyword-separator",
        type=str,
        default=", ",
        help="Separator used when joining repeated keyword items into keyword_text.",
    )
    parser.add_argument(
        "--counterfactual-mode",
        choices=("manual", "swap", "disable"),
        default="manual",
        help="manual: require explicit counterfactual keyword input; swap: rotate the positive keyword set; disable: write an empty counterfactual string.",
    )
    parser.add_argument(
        "--counterfactual-keyword-text",
        type=str,
        default=None,
        help="Complete Chinese counterfactual_keyword_text string.",
    )
    parser.add_argument(
        "--counterfactual-keyword-item-zh",
        action="append",
        default=[],
        help="Repeated Chinese counterfactual keyword item. Can be passed multiple times instead of --counterfactual-keyword-text.",
    )
    parser.add_argument(
        "--counterfactual-separator",
        type=str,
        default=", ",
        help="Separator used when joining repeated counterfactual keyword items.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the forwarded SongLing record command and metadata plan without executing it.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args, passthrough = build_arg_parser().parse_known_args()

    config_override, passthrough = _extract_last_flag_value_and_strip(passthrough, ("--config_path", "--config-path"))
    repo_override, passthrough = _extract_last_flag_value_and_strip(passthrough, ("--dataset.repo_id",))
    root_override, passthrough = _extract_last_flag_value_and_strip(passthrough, ("--dataset.root",))
    task_override, passthrough = _extract_last_flag_value_and_strip(passthrough, ("--dataset.single_task",))

    config_path = Path(config_override) if config_override is not None else args.config_path
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    dataset_root = Path(root_override) if root_override is not None else (args.dataset_root or _timestamped_default_root())
    dataset_root = dataset_root.expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path.cwd() / dataset_root).resolve()

    dataset_repo_id = repo_override if repo_override is not None else args.dataset_repo_id
    instruction = task_override if task_override is not None else args.instruction
    if instruction is None or not _clean_text(instruction):
        raise ValueError("Missing English instruction. Provide --instruction or pass --dataset.single_task explicitly.")
    instruction = _clean_text(instruction)

    keyword_text = _resolve_keyword_text(
        full_text=args.keyword_text,
        items=args.keyword_item_zh,
        separator=args.keyword_separator,
        field_name="keyword_text / keyword-item-zh",
    )

    if args.counterfactual_mode == "manual":
        counterfactual_keyword_text = _resolve_keyword_text(
            full_text=args.counterfactual_keyword_text,
            items=args.counterfactual_keyword_item_zh,
            separator=args.counterfactual_separator,
            field_name="counterfactual_keyword_text / counterfactual-keyword-item-zh",
        )
    elif args.counterfactual_mode == "swap":
        counterfactual_keyword_text = _build_swap_counterfactual(
            keyword_text,
            separator=args.counterfactual_separator,
        )
    else:
        counterfactual_keyword_text = ""

    recommended_swap = ""
    try:
        recommended_swap = _build_swap_counterfactual(keyword_text, separator=args.counterfactual_separator)
    except ValueError:
        recommended_swap = ""

    before_episode_indices = _load_episode_indices(dataset_root) if dataset_root.exists() else set()

    cmd = [
        sys.executable,
        str(DEFAULT_RECORD_COMPAT_PATH),
        f"--config_path={config_path}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={instruction}",
        *passthrough,
    ]
    if not _has_flag(passthrough, ("--dataset.push_to_hub",)):
        cmd.append("--dataset.push_to_hub=false")

    print("KC-VLA SongLing recording plan:")
    print(f"  dataset_root: {dataset_root}")
    print(f"  repo_id: {dataset_repo_id}")
    print(f"  instruction: {instruction}")
    print(f"  keyword_text: {keyword_text}")
    print(f"  counterfactual_mode: {args.counterfactual_mode}")
    print(f"  counterfactual_keyword_text: {counterfactual_keyword_text}")
    if recommended_swap:
        print(f"  recommended_swap_counterfactual: {recommended_swap}")
    print("Forwarding command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    if args.dry_run:
        return

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    after_episode_indices = _load_episode_indices(dataset_root)
    new_episode_indices = after_episode_indices - before_episode_indices
    if not new_episode_indices:
        logging.warning("No new episodes were detected under %s; skipped KC-VLA keyword annotation.", dataset_root)
        return

    summary = annotate_dataset_keywords(
        dataset_root,
        episode_indices=new_episode_indices,
        keyword_text=keyword_text,
        counterfactual_keyword_text=counterfactual_keyword_text,
    )
    logging.info(
        "Annotated %d new episodes across %d parquet files (%d rows) with KC-VLA metadata.",
        summary["episode_count"],
        summary["updated_files"],
        summary["updated_rows"],
    )


if __name__ == "__main__":
    main()
