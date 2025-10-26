"""Unit tests for :class:`deepsynth.pipelines.efficient_incremental_uploader.EfficientIncrementalUploader`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from deepsynth.pipelines.efficient_incremental_uploader import EfficientIncrementalUploader


@pytest.fixture()
def uploader(tmp_path: Path) -> EfficientIncrementalUploader:
    """Create a lightweight uploader instance without running the heavy initialiser."""

    instance = object.__new__(EfficientIncrementalUploader)
    instance.work_dir = tmp_path
    instance.samples_dir = tmp_path / "samples"
    instance.samples_dir.mkdir()
    instance.uploaded_dir = tmp_path / "uploaded"
    instance.uploaded_dir.mkdir()
    instance.upload_progress_file = tmp_path / "upload_progress.json"
    instance.upload_progress = {
        "last_uploaded_batch": -1,
        "total_uploaded_samples": 0,
        "upload_count": 0,
        "dataset_created": False,
    }
    instance.batches_per_upload = 2
    instance.hf_token = "token"
    instance.username = "user"
    instance.dataset_name = "user/dataset"
    instance.api = None  # Not needed for the tested methods
    instance.shard_manager = None  # Patched in individual tests when required
    return instance


def test_filter_duplicate_samples_updates_registry(uploader: EfficientIncrementalUploader) -> None:
    """Duplicates and previously processed samples should be filtered out."""

    registry: Dict[str, Dict[str, int]] = {"source": {"train": 3}}
    samples: List[Dict[str, object]] = [
        {"source_dataset": "source", "original_split": "train", "original_index": 4},
        {"source_dataset": "source", "original_split": "train", "original_index": 2},  # already processed
        {"source_dataset": "source", "original_split": "train", "original_index": 4},  # duplicate in chunk
        {"source_dataset": "source", "original_split": "train", "original_index": 5},
        {"source_dataset": "other", "original_split": "test", "original_index": "1"},
        {"source_dataset": None, "original_split": "train", "original_index": 10},
    ]

    filtered, updated_registry, skipped = uploader.filter_duplicate_samples(samples, registry)

    assert skipped == 2  # one duplicate in chunk + one already processed remotely
    assert len(filtered) == 4
    assert updated_registry["source"]["train"] == 5
    assert updated_registry["other"]["test"] == 1


def test_load_upload_progress_converts_legacy_fields(uploader: EfficientIncrementalUploader) -> None:
    """Older progress files should be migrated to the current schema."""

    legacy_payload = {
        "last_uploaded_batch": 3,
        "total_uploaded": 1200,
        "upload_count": 2,
    }
    uploader.upload_progress_file.write_text(json.dumps(legacy_payload))

    progress = uploader.load_upload_progress()

    assert progress["total_uploaded_samples"] == 1200
    assert progress["dataset_created"] is True
    assert progress["last_uploaded_batch"] == 3


def test_get_pending_batches_filters_processed(uploader: EfficientIncrementalUploader) -> None:
    """Only batches with an index greater than the last uploaded one should be returned."""

    (uploader.samples_dir / "batch_00001.pkl").write_bytes(b"")
    (uploader.samples_dir / "batch_00003.pkl").write_bytes(b"")
    (uploader.samples_dir / "batch_00002.pkl").write_bytes(b"")

    uploader.upload_progress["last_uploaded_batch"] = 1

    pending = uploader.get_pending_batches()

    assert [batch_id for batch_id, _ in pending] == [2, 3]
    assert all(path.name.startswith("batch_") for _, path in pending)
