#!/usr/bin/env python3
"""One-command uploader for Songling ALOHA datasets to Hugging Face Hub.

Usage:

    python examples/songling_aloha/push_dataset_to_hub.py \
      --dataset-root /abs/path/to/dataset \
      --repo-id YSanYi/songling_aloha_plate_stack_v1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

REQUIRED_RELATIVE_PATHS = (
    "data/chunk-000/file-000.parquet",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "videos/observation.images.left_elbow/chunk-000/file-000.mp4",
    "videos/observation.images.left_high/chunk-000/file-000.mp4",
    "videos/observation.images.right_elbow/chunk-000/file-000.mp4",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push a local Songling ALOHA dataset to HF Hub.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Local dataset root (required). Example: /home/aiteam/.../outputs/songling_aloha/songling_aloha_plate_stack_v1",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HF dataset repo id (required). Example: YSanYi/songling_aloha_plate_stack_v1",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional custom commit message. If omitted, a timestamped message is used.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-upload remote file verification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate local dataset and print resolved settings.",
    )
    return parser.parse_args()


def _assert_dataset_layout(dataset_root: Path) -> None:
    missing = [rel for rel in REQUIRED_RELATIVE_PATHS if not (dataset_root / rel).exists()]
    if missing:
        missing_lines = "\n".join(f"- {rel}" for rel in missing)
        raise FileNotFoundError(
            f"Dataset layout validation failed under {dataset_root}.\nMissing required files:\n{missing_lines}"
        )


def _default_commit_message(dataset_root: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Upload {dataset_root.name} ({now})"


def _verify_remote_files(api: HfApi, repo_id: str) -> None:
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    required_remote = set(REQUIRED_RELATIVE_PATHS)
    missing_remote = sorted(required_remote - remote_files)
    if missing_remote:
        missing_lines = "\n".join(f"- {rel}" for rel in missing_remote)
        raise RuntimeError(
            "Upload completed but remote verification failed.\n"
            f"Repository: {repo_id}\nMissing required remote files:\n{missing_lines}"
        )


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    _assert_dataset_layout(dataset_root)

    commit_message = args.commit_message or _default_commit_message(dataset_root)

    print("Resolved upload config")
    print(f"- dataset_root: {dataset_root}")
    print(f"- repo_id: {args.repo_id}")
    print(f"- commit_message: {commit_message}")
    print(f"- verify_remote: {not args.skip_verify}")
    print(f"- private_if_create: {bool(args.private)}")
    print("- auto_create_repo_if_missing: True")

    if args.dry_run:
        print("Dry run complete. No upload performed.")
        return

    api = HfApi()

    try:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=bool(args.private))
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Failed to create/check remote repo {args.repo_id}: {exc}") from exc

    try:
        commit_info = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(dataset_root),
            path_in_repo=".",
            commit_message=commit_message,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            "Upload failed. Check your HF token permissions and network connectivity.\n"
            f"repo_id={args.repo_id}\ndataset_root={dataset_root}\nerror={exc}"
        ) from exc

    commit_id = getattr(commit_info, "oid", None) or getattr(commit_info, "commit_id", None)
    if commit_id is None:
        commit_id = str(commit_info)
    commit_url = f"https://huggingface.co/datasets/{args.repo_id}/commit/{commit_id}"

    if not args.skip_verify:
        _verify_remote_files(api=api, repo_id=args.repo_id)

    print("Upload complete")
    print(f"- dataset: https://huggingface.co/datasets/{args.repo_id}")
    print(f"- commit: {commit_url}")


if __name__ == "__main__":
    main()
