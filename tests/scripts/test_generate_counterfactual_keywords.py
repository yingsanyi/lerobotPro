#!/usr/bin/env python

import json

import pandas as pd

from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_FEATURES, get_hf_features_from_features, to_parquet_with_hf_images, write_info, write_tasks
from lerobot.scripts.lerobot_generate_counterfactual_keywords import process_dataset_root
from lerobot.utils.constants import COUNTERFACTUAL_KEYWORD_TEXT, KEYWORD_TEXT


def _make_info(features: dict[str, dict], total_frames: int) -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": "dummy_robot",
        "total_episodes": 2,
        "total_frames": total_frames,
        "total_tasks": 2,
        "total_videos": 0,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 30,
        "splits": {},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": None,
        "features": features,
    }


def _make_dataset_root(tmp_path, *, include_counterfactual_feature: bool = False) -> tuple:
    root = tmp_path / "kcvla_dataset"
    features = {
        **DEFAULT_FEATURES,
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        KEYWORD_TEXT: {"dtype": "string", "shape": (1,), "names": None},
    }
    if include_counterfactual_feature:
        features[COUNTERFACTUAL_KEYWORD_TEXT] = {"dtype": "string", "shape": (1,), "names": None}

    write_info(_make_info(features, total_frames=4), root)
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["sort medicine", "sort snacks"], name="task"))
    write_tasks(tasks, root)

    df = pd.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "frame_index": [0, 1, 2, 3],
            "episode_index": [0, 0, 1, 1],
            "index": [0, 1, 2, 3],
            "task_index": [0, 0, 0, 1],
            "observation.state": [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            "action": [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9]],
            KEYWORD_TEXT: ["painkiller, vitamin", "painkiller", "vitamin", "cracker"],
        }
    )
    if include_counterfactual_feature:
        df[COUNTERFACTUAL_KEYWORD_TEXT] = ["manual alt", "", "", ""]

    path = root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_parquet_with_hf_images(df, path, features=get_hf_features_from_features(features))
    return root, path


def test_process_dataset_root_generates_counterfactual_keywords(tmp_path):
    root, parquet_path = _make_dataset_root(tmp_path, include_counterfactual_feature=False)

    summary = process_dataset_root(
        root=root,
        mapping_path=None,
        seed=7,
        dry_run=False,
        overwrite_existing=False,
        pool_scope="task",
    )

    df = pd.read_parquet(parquet_path)
    assert df[COUNTERFACTUAL_KEYWORD_TEXT].tolist() == [
        "vitamin, painkiller",
        "vitamin",
        "painkiller",
        "painkiller",
    ]
    assert summary == {"total_rows": 4, "generated_rows": 4, "preserved_rows": 0, "skipped_rows": 0}

    with open(root / "meta" / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    assert COUNTERFACTUAL_KEYWORD_TEXT in info["features"]


def test_process_dataset_root_preserves_existing_values_and_uses_mapping(tmp_path):
    root, parquet_path = _make_dataset_root(tmp_path, include_counterfactual_feature=True)
    mapping_path = tmp_path / "keyword_map.json"
    mapping_path.write_text('{"cracker": ["cookie"]}', encoding="utf-8")

    summary = process_dataset_root(
        root=root,
        mapping_path=mapping_path,
        seed=11,
        dry_run=False,
        overwrite_existing=False,
        pool_scope="task",
    )

    df = pd.read_parquet(parquet_path)
    assert df[COUNTERFACTUAL_KEYWORD_TEXT].tolist() == [
        "manual alt",
        "vitamin",
        "painkiller",
        "cookie",
    ]
    assert summary == {"total_rows": 4, "generated_rows": 3, "preserved_rows": 1, "skipped_rows": 0}
