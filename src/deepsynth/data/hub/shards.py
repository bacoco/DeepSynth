"""Utilities for uploading dataset shards to the HuggingFace Hub.

This module centralises the new incremental upload strategy where each
batch of samples is serialized as an independent shard stored under the
``data/`` directory of the dataset repository.  An accompanying index
(`_deepsynth/shards.json`) tracks the shards already present on the Hub and the
``(source_dataset, original_index)`` pairs they contain so that
subsequent uploads can avoid duplicates.
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict
from datasets import Image as ImageFeature
from huggingface_hub import HfApi, hf_hub_download

INDEX_PATH_IN_REPO = "_deepsynth/shards.json"
DEFAULT_SHARD_PREFIX = "batch_"


@dataclass
class UploadResult:
    """Summary of a shard upload operation."""

    shard_id: str
    uploaded_samples: int
    skipped_duplicates: int
    index_updated: bool


class HubShardManager:
    """Manage dataset shards stored on the HuggingFace Hub."""

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        api: Optional[HfApi] = None,
        shard_prefix: str = DEFAULT_SHARD_PREFIX,
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.api = api or HfApi()
        self.shard_prefix = shard_prefix

        # Ensure the dataset repository exists before attempting any upload.
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token,
            private=False,
            exist_ok=True,
        )

        self.index = self._load_index()
        self._ensure_index_structure()
        self.known_shards = {entry["id"] for entry in self.index["shards"]}
        self._existing_pairs = {
            (item["source_dataset"], item["original_index"])
            for entry in self.index["shards"]
            for item in entry.get("original_indices", [])
        }
        self._pending_index_update = False
        self._next_batch_id = self._compute_next_batch_id()

    # ------------------------------------------------------------------
    def _load_index(self) -> Dict[str, object]:
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                filename=INDEX_PATH_IN_REPO,
            )
            with open(local_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            # No index present yet.
            return {"shards": []}

    def _ensure_index_structure(self) -> None:
        if "shards" not in self.index or not isinstance(self.index["shards"], list):
            self.index["shards"] = []
        if "metadata" not in self.index or not isinstance(self.index["metadata"], dict):
            self.index["metadata"] = {}

    def _compute_next_batch_id(self) -> int:
        pattern = re.compile(rf"{re.escape(self.shard_prefix)}(\d+)")
        max_value = -1
        for entry in self.index["shards"]:
            match = pattern.fullmatch(entry.get("id", ""))
            if match:
                try:
                    candidate = int(match.group(1))
                except ValueError:
                    continue
                max_value = max(max_value, candidate)
        return max_value + 1

    # ------------------------------------------------------------------
    def format_shard_id(self, batch_number: int) -> str:
        return f"{self.shard_prefix}{int(batch_number):06d}"

    def shard_exists(self, shard_id: str) -> bool:
        return shard_id in self.known_shards

    def next_shard_id(self) -> str:
        shard_id = self.format_shard_id(self._next_batch_id)
        self._next_batch_id += 1
        return shard_id

    # ------------------------------------------------------------------
    def upload_samples_as_shard(
        self,
        samples: Sequence[Dict[str, object]],
        shard_id: str,
        commit_message: Optional[str] = None,
    ) -> UploadResult:
        if not samples:
            return UploadResult(shard_id=shard_id, uploaded_samples=0, skipped_duplicates=0, index_updated=False)

        if self.shard_exists(shard_id):
            raise ValueError(f"Shard '{shard_id}' already exists on the Hub. Choose a different identifier.")

        filtered_samples, skipped = self._filter_new_samples(samples)
        if not filtered_samples:
            # Nothing new to upload. We still mark the shard identifier as seen to avoid reusing it.
            self.known_shards.add(shard_id)
            return UploadResult(
                shard_id=shard_id,
                uploaded_samples=0,
                skipped_duplicates=skipped,
                index_updated=False,
            )

        numeric_suffix = self._extract_numeric_suffix(shard_id)
        if numeric_suffix is not None and numeric_suffix >= self._next_batch_id:
            self._next_batch_id = numeric_suffix + 1

        dataset = self._build_dataset(filtered_samples)
        path_in_repo = f"data/{shard_id}.parquet"  # Single parquet file
        message = commit_message or f"Add shard {shard_id} ({len(filtered_samples)} samples)"

        # Export to parquet and upload ONLY the file (don't touch metadata)
        # This allows incremental uploads without overwriting dataset metadata
        with tempfile.TemporaryDirectory() as tmpdir:
            local_parquet = Path(tmpdir) / f"{shard_id}.parquet"
            dataset.to_parquet(local_parquet)

            # Upload JUST the parquet file
            self.api.upload_file(
                path_or_fileobj=str(local_parquet),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=message,
            )

        shard_entry = self._build_shard_entry(shard_id, path_in_repo, filtered_samples)
        self.index["shards"].append(shard_entry)
        self.index["shards"].sort(key=lambda item: item["id"])
        self.known_shards.add(shard_id)
        self._pending_index_update = True

        return UploadResult(
            shard_id=shard_id,
            uploaded_samples=len(filtered_samples),
            skipped_duplicates=skipped,
            index_updated=True,
        )

    # ------------------------------------------------------------------
    def _filter_new_samples(
        self, samples: Sequence[Dict[str, object]]
    ) -> Tuple[List[Dict[str, object]], int]:
        filtered: List[Dict[str, object]] = []
        skipped = 0
        for sample in samples:
            key = (
                str(sample.get("source_dataset")),
                int(sample.get("original_index", -1)),
            )
            if key in self._existing_pairs:
                skipped += 1
                continue
            self._existing_pairs.add(key)
            filtered.append(sample)
        return filtered, skipped

    def _build_dataset(self, samples: Sequence[Dict[str, object]]) -> Dataset:
        """
        Build HuggingFace Dataset from samples.

        Supports both summarization ({text, summary, image}) and Q&A ({text, instruction, answer, metadata}) schemas.
        Automatically detects and includes all fields from samples.
        """
        if not samples:
            return Dataset.from_dict({})

        # Collect all unique keys from samples
        all_keys = set()
        for sample in samples:
            all_keys.update(sample.keys())

        # Build payload dynamically with all fields
        base_payload = {}
        for key in sorted(all_keys):
            base_payload[key] = [sample.get(key) for sample in samples]

        # Identify image columns (main "image" and multi-resolution "image_<name>")
        image_columns = sorted([key for key in all_keys if key.startswith("image")])

        # Create dataset
        dataset = Dataset.from_dict(base_payload)

        # Cast image columns to HuggingFace ImageFeature
        for column in image_columns:
            # Only cast if column contains images (not None)
            if any(dataset[column]):
                dataset = dataset.cast_column(column, ImageFeature())

        return dataset

    def _build_shard_entry(
        self,
        shard_id: str,
        path_in_repo: str,
        samples: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        return {
            "id": shard_id,
            "path": path_in_repo,
            "num_samples": len(samples),
            "original_indices": [
                {
                    "source_dataset": sample["source_dataset"],
                    "original_index": int(sample["original_index"]),
                }
                for sample in samples
            ],
        }

    # ------------------------------------------------------------------
    def save_index(self, commit_message: Optional[str] = None) -> bool:
        if not self._pending_index_update:
            return False

        payload = json.dumps(self.index, indent=2, sort_keys=True).encode("utf-8")
        message = commit_message or "Update shard index"
        self.api.upload_file(
            path_or_fileobj=payload,
            path_in_repo=INDEX_PATH_IN_REPO,
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self.token,
            commit_message=message,
        )
        self._pending_index_update = False
        return True

    def update_metadata(self, metadata: Dict[str, object]) -> None:
        base = self.index.get("metadata", {})
        base.update(metadata)
        self.index["metadata"] = base
        self._pending_index_update = True

    # ------------------------------------------------------------------
    def _extract_numeric_suffix(self, shard_id: str) -> Optional[int]:
        if not shard_id.startswith(self.shard_prefix):
            return None
        suffix = shard_id[len(self.shard_prefix) :]
        if not suffix.isdigit():
            return None
        try:
            return int(suffix)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    @property
    def existing_pairs(self) -> Iterable[Tuple[str, int]]:
        return iter(self._existing_pairs)

    @property
    def index_path(self) -> str:
        return INDEX_PATH_IN_REPO

