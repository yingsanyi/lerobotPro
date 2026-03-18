#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Generate `counterfactual_keyword_text` offline for KC-VLA datasets.

This script updates a local LeRobot dataset in-place by scanning `data/**/*.parquet`
and generating a new `counterfactual_keyword_text` column from the existing
`keyword_text` column.

Generation rules are intentionally conservative:
1. If a frame already has multiple distinct keywords, use a permutation of the
   same keyword set. This is the safest counterfactual because those objects are
   already present in the frame.
2. Otherwise, sample a replacement keyword from the same task pool.
3. If the task pool has no alternative, fall back to the global keyword pool.
4. If an explicit JSON replacement map is provided, it is used before the pool
   fallback.

Examples:

```shell
lerobot-generate-counterfactual-keywords \
    --root /path/to/your_dataset

lerobot-generate-counterfactual-keywords \
    --root /path/to/your_dataset \
    --mapping-path /path/to/keyword_replacements.json \
    --overwrite-existing
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_TASKS_PATH,
    get_hf_features_from_features,
    load_info,
    load_tasks,
    to_parquet_with_hf_images,
    write_info,
)
from lerobot.utils.constants import COUNTERFACTUAL_KEYWORD_TEXT, HF_LEROBOT_HOME, KEYWORD_TEXT

DEFAULT_STRING_FEATURE = {"dtype": "string", "shape": (1,), "names": None}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        return str(value).strip()
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("\t", " ").strip()


def split_keyword_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            items.extend(split_keyword_text(item))
        return items

    text = _clean_text(value)
    if not text:
        return []
    for delimiter in ("，", ";", "；", "|", "\n"):
        text = text.replace(delimiter, ",")
    return [item for item in (_clean_text(part) for part in text.split(",")) if item]


def join_keyword_text(keywords: list[str]) -> str:
    return ", ".join(_clean_text(keyword) for keyword in keywords if _clean_text(keyword))


def load_keyword_replacement_map(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}

    with open(path, encoding="utf-8") as f:
        raw_map = json.load(f)

    if not isinstance(raw_map, dict):
        raise ValueError("Replacement map must be a JSON object of keyword -> replacement list")

    normalized: dict[str, list[str]] = {}
    for source_keyword, candidates in raw_map.items():
        source = _clean_text(source_keyword)
        if not source:
            continue
        if isinstance(candidates, str):
            candidate_list = split_keyword_text(candidates)
        elif isinstance(candidates, list):
            candidate_list = []
            for candidate in candidates:
                candidate_list.extend(split_keyword_text(candidate))
        else:
            raise ValueError(
                f"Replacement map entry for '{source}' must be a string or list of strings, got {type(candidates)}"
            )
        normalized[source] = [candidate for candidate in candidate_list if candidate != source]
    return normalized


def _stable_pick(options: list[str], key: str, seed: int) -> str:
    if not options:
        raise ValueError("Cannot pick from an empty candidate list")
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(options)
    return options[index]


def _permute_keywords(keywords: list[str]) -> list[str] | None:
    if len(keywords) < 2 or len(set(keywords)) < 2:
        return None
    rotated = keywords[1:] + keywords[:1]
    if rotated != keywords:
        return rotated
    reversed_keywords = list(reversed(keywords))
    if reversed_keywords != keywords:
        return reversed_keywords
    return None


def _resolve_dataset_root(root: str | Path | None, repo_id: str | None) -> Path:
    if root is not None:
        return Path(root)
    if repo_id is not None:
        return HF_LEROBOT_HOME / repo_id
    raise ValueError("Either --root or --repo-id must be provided")


def _ensure_string_feature(info: dict[str, Any], feature_name: str) -> bool:
    features = info.setdefault("features", {})
    feature = features.get(feature_name)
    if feature is None:
        features[feature_name] = dict(DEFAULT_STRING_FEATURE)
        return True

    if feature.get("dtype") != "string":
        raise ValueError(f"Feature '{feature_name}' must have dtype='string', got {feature.get('dtype')}")
    return False


def _load_task_index_to_name(root: Path) -> dict[int, str]:
    tasks_path = root / DEFAULT_TASKS_PATH
    if not tasks_path.exists():
        return {}
    tasks = load_tasks(root)
    return {int(row.task_index): str(task_name) for task_name, row in tasks.iterrows()}


def _task_labels_for_df(df: pd.DataFrame, task_index_to_name: dict[int, str]) -> list[str | None]:
    if "task" in df.columns:
        return [_clean_text(value) or None for value in df["task"]]
    if "task_index" in df.columns:
        labels: list[str | None] = []
        for value in df["task_index"]:
            if pd.isna(value):
                labels.append(None)
                continue
            labels.append(task_index_to_name.get(int(value)))
        return labels
    return [None] * len(df)


def _build_keyword_pools(
    parquet_paths: list[Path],
    keyword_column: str,
    task_index_to_name: dict[int, str],
) -> tuple[dict[str, list[str]], list[str]]:
    task_pools: dict[str, set[str]] = {}
    global_pool: set[str] = set()

    for parquet_path in parquet_paths:
        available_columns = set(pq.read_schema(parquet_path).names)
        columns = [column for column in (keyword_column, "task", "task_index") if column in available_columns]
        if keyword_column not in columns:
            continue

        df = pd.read_parquet(parquet_path, columns=columns)
        task_labels = _task_labels_for_df(df, task_index_to_name)
        for row_index, keyword_text in enumerate(df[keyword_column]):
            keywords = split_keyword_text(keyword_text)
            if not keywords:
                continue
            global_pool.update(keywords)
            task_label = task_labels[row_index]
            if task_label is None:
                continue
            task_pools.setdefault(task_label, set()).update(keywords)

    return ({task: sorted(keywords) for task, keywords in task_pools.items()}, sorted(global_pool))


def generate_counterfactual_keyword_text(
    keyword_text: Any,
    *,
    row_key: str,
    seed: int,
    task_label: str | None,
    task_pools: dict[str, list[str]],
    global_pool: list[str],
    replacement_map: dict[str, list[str]],
    pool_scope: str = "task",
    prefer_permutation: bool = True,
) -> str:
    keywords = split_keyword_text(keyword_text)
    if not keywords:
        return ""

    if prefer_permutation:
        permuted = _permute_keywords(keywords)
        if permuted is not None:
            return join_keyword_text(permuted)

    replacements: list[str] = []
    for position, keyword in enumerate(keywords):
        candidates = [candidate for candidate in replacement_map.get(keyword, []) if candidate != keyword]
        if not candidates and pool_scope == "task" and task_label is not None:
            candidates = [candidate for candidate in task_pools.get(task_label, []) if candidate != keyword]
        if not candidates:
            candidates = [candidate for candidate in global_pool if candidate != keyword]
        if not candidates:
            return ""
        replacements.append(_stable_pick(candidates, f"{row_key}:{position}:{keyword}", seed))

    if replacements == keywords:
        return ""
    return join_keyword_text(replacements)


def process_dataset_root(
    root: str | Path,
    *,
    repo_id: str | None = None,
    keyword_column: str = KEYWORD_TEXT,
    counterfactual_column: str = COUNTERFACTUAL_KEYWORD_TEXT,
    mapping_path: str | Path | None = None,
    overwrite_existing: bool = False,
    pool_scope: str = "task",
    prefer_permutation: bool = True,
    seed: int = 0,
    dry_run: bool = False,
) -> dict[str, int]:
    dataset_root = _resolve_dataset_root(root, repo_id)
    data_dir = dataset_root / DATA_DIR
    parquet_paths = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet files found under {data_dir}")

    info = load_info(dataset_root)
    schema_changed = False
    schema_changed |= _ensure_string_feature(info, keyword_column)
    schema_changed |= _ensure_string_feature(info, counterfactual_column)
    if schema_changed and not dry_run:
        write_info(info, dataset_root)
        info = load_info(dataset_root)

    task_index_to_name = _load_task_index_to_name(dataset_root)
    replacement_map = load_keyword_replacement_map(mapping_path)
    task_pools, global_pool = _build_keyword_pools(parquet_paths, keyword_column, task_index_to_name)
    if not global_pool:
        raise ValueError(f"No valid keywords found in column '{keyword_column}'")

    hf_features = get_hf_features_from_features(info["features"])
    summary = {"total_rows": 0, "generated_rows": 0, "preserved_rows": 0, "skipped_rows": 0}

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path).reset_index(drop=True)
        if keyword_column not in df.columns:
            raise ValueError(f"Column '{keyword_column}' not found in {parquet_path}")

        task_labels = _task_labels_for_df(df, task_index_to_name)
        existing_values = (
            [_clean_text(value) for value in df[counterfactual_column]]
            if counterfactual_column in df.columns
            else [""] * len(df)
        )
        generated_values: list[str] = []

        for row_index, keyword_text in enumerate(df[keyword_column]):
            summary["total_rows"] += 1
            existing_value = existing_values[row_index]
            if existing_value and not overwrite_existing:
                generated_values.append(existing_value)
                summary["preserved_rows"] += 1
                continue

            row = df.iloc[row_index]
            row_key = (
                f"{parquet_path.name}:{row.get('episode_index', '')}:{row.get('frame_index', '')}:"
                f"{row.get('index', row_index)}"
            )
            generated_value = generate_counterfactual_keyword_text(
                keyword_text,
                row_key=row_key,
                seed=seed,
                task_label=task_labels[row_index],
                task_pools=task_pools,
                global_pool=global_pool,
                replacement_map=replacement_map,
                pool_scope=pool_scope,
                prefer_permutation=prefer_permutation,
            )
            generated_values.append(generated_value)
            if generated_value:
                summary["generated_rows"] += 1
            else:
                summary["skipped_rows"] += 1

        df[counterfactual_column] = generated_values
        if not dry_run:
            to_parquet_with_hf_images(df, parquet_path, features=hf_features)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root containing meta/ and data/")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo id resolved under HF_LEROBOT_HOME")
    parser.add_argument("--keyword-column", type=str, default=KEYWORD_TEXT)
    parser.add_argument("--counterfactual-column", type=str, default=COUNTERFACTUAL_KEYWORD_TEXT)
    parser.add_argument("--mapping-path", type=Path, default=None, help="Optional JSON keyword replacement map")
    parser.add_argument(
        "--pool-scope",
        type=str,
        choices=("task", "global"),
        default="task",
        help="Prefer task-local candidate pools before falling back to the global keyword pool",
    )
    parser.add_argument("--overwrite-existing", action="store_true", help="Regenerate non-empty values as well")
    parser.add_argument(
        "--prefer-permutation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer permuting existing multi-keyword sets before pool-based replacement",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Compute results without writing parquet files")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args()

    summary = process_dataset_root(
        root=args.root,
        repo_id=args.repo_id,
        keyword_column=args.keyword_column,
        counterfactual_column=args.counterfactual_column,
        mapping_path=args.mapping_path,
        overwrite_existing=args.overwrite_existing,
        pool_scope=args.pool_scope,
        prefer_permutation=args.prefer_permutation,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    logging.info(
        "Processed %d rows: generated=%d preserved=%d skipped=%d",
        summary["total_rows"],
        summary["generated_rows"],
        summary["preserved_rows"],
        summary["skipped_rows"],
    )


if __name__ == "__main__":
    main()
