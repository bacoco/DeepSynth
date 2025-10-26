"""
Refactored global state pipeline with reduced complexity.

Splits the complex process_dataset_incremental function into smaller,
more manageable components.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from deepsynth.data.hub import HubShardManager
from deepsynth.data.transforms import TextToImageConverter
from deepsynth.utils import extract_text_summary, get_logger
from deepsynth.utils.dataset_extraction import DATASET_CONFIGS, DatasetConfig

logger = get_logger(__name__)


class RefactoredGlobalStatePipeline:
    """Refactored pipeline with better separation of concerns."""

    def __init__(
        self,
        target_dataset_name: str,
        hf_token: str,
        hf_username: str,
        batch_size: int = 10000,
        image_converter: Optional[TextToImageConverter] = None,
    ):
        """Initialize the refactored pipeline."""
        self.target_dataset_name = target_dataset_name
        self.hf_token = hf_token
        self.hf_username = hf_username
        self.batch_size = batch_size

        # Initialize components
        self.converter = image_converter or TextToImageConverter()
        self.shard_manager = HubShardManager(
            dataset_name=target_dataset_name,
            token=hf_token,
        )
        self.api = HfApi(token=hf_token)

        # State management
        self.current_batch = []
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "uploaded": 0,
        }

    def process_dataset_incremental(
        self,
        dataset_key: str,
        source_config: Dict[str, Any],
        start_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Process a dataset incrementally with reduced complexity.

        This is the refactored version of the complex function.
        """
        # Step 1: Setup and validation
        dataset_info = self._setup_dataset(dataset_key, source_config)
        if not dataset_info:
            return {"status": "error", "message": "Failed to setup dataset"}

        # Step 2: Load dataset
        dataset = self._load_dataset_split(dataset_info)
        if dataset is None:
            return {"status": "error", "message": "Failed to load dataset"}

        # Step 3: Process in batches
        result = self._process_batches(
            dataset,
            dataset_info,
            start_idx,
        )

        # Step 4: Cleanup
        self._cleanup()

        return result

    def _setup_dataset(
        self,
        dataset_key: str,
        source_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Setup dataset configuration and validation."""
        try:
            # Parse configuration
            name = source_config.get("name", dataset_key)
            subset = source_config.get("subset")
            split = source_config.get("split", "train")
            max_samples = source_config.get("max_samples")

            # Get dataset configuration
            if name.lower() in DATASET_CONFIGS:
                dataset_config = DATASET_CONFIGS[name.lower()]
            else:
                dataset_config = DatasetConfig(name=name)

            info = {
                "key": dataset_key,
                "name": name,
                "subset": subset,
                "split": split,
                "max_samples": max_samples,
                "config": dataset_config,
            }

            logger.info(
                f"Setup dataset: {name} (subset={subset}, split={split}, max={max_samples})"
            )

            return info

        except Exception as e:
            logger.error(f"Failed to setup dataset {dataset_key}: {e}")
            return None

    def _load_dataset_split(
        self,
        dataset_info: Dict[str, Any],
    ) -> Optional[Dataset]:
        """Load a dataset split with error handling."""
        try:
            name = dataset_info["name"]
            subset = dataset_info["subset"]
            split = dataset_info["split"]

            logger.info(f"Loading dataset {name}...")

            if subset:
                dataset = load_dataset(name, subset, split=split)
            else:
                dataset = load_dataset(name, split=split)

            logger.info(f"Loaded {len(dataset)} samples from {name}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None

    def _process_batches(
        self,
        dataset: Dataset,
        dataset_info: Dict[str, Any],
        start_idx: int,
    ) -> Dict[str, Any]:
        """Process dataset in batches."""
        max_samples = dataset_info.get("max_samples") or len(dataset)
        target_limit = min(len(dataset), max_samples)

        logger.info(
            f"Processing samples {start_idx} to {target_limit} "
            f"in batches of {self.batch_size}"
        )

        # Progress tracking
        pbar = tqdm(
            range(start_idx, target_limit),
            desc=f"Processing {dataset_info['name']}",
            initial=start_idx,
        )

        for idx in pbar:
            # Process single sample
            success = self._process_single_sample(
                dataset[idx],
                dataset_info,
                idx,
            )

            if success:
                self.stats["processed"] += 1
            else:
                self.stats["skipped"] += 1

            # Check if batch is ready
            if len(self.current_batch) >= self.batch_size:
                self._upload_current_batch()

            # Update progress
            if idx > 0 and idx % 1000 == 0:
                self._log_progress(idx, target_limit)

        # Upload remaining batch
        if self.current_batch:
            self._upload_current_batch()

        return {
            "status": "success",
            "stats": self.stats.copy(),
        }

    def _process_single_sample(
        self,
        sample: Dict[str, Any],
        dataset_info: Dict[str, Any],
        index: int,
    ) -> bool:
        """Process a single sample."""
        try:
            # Extract text and summary
            text, summary = extract_text_summary(
                sample,
                dataset_config=dataset_info["config"],
            )

            if not text or not summary:
                return False

            # Check for duplicates
            if self._is_duplicate(dataset_info["key"], index):
                logger.debug(f"Skipping duplicate: {dataset_info['key']}:{index}")
                return False

            # Convert text to image
            image = self._convert_to_image(text)
            if image is None:
                return False

            # Add to batch
            self.current_batch.append({
                "text": text,
                "summary": summary,
                "image": image,
                "source_dataset": dataset_info["key"],
                "original_split": dataset_info["split"],
                "original_index": index,
            })

            return True

        except Exception as e:
            logger.error(f"Error processing sample {index}: {e}")
            self.stats["errors"] += 1
            return False

    def _convert_to_image(self, text: str) -> Optional[Any]:
        """Convert text to image with error handling."""
        try:
            return self.converter.convert(text)
        except Exception as e:
            logger.error(f"Failed to convert text to image: {e}")
            return None

    def _is_duplicate(self, dataset_key: str, index: int) -> bool:
        """Check if a sample is already processed."""
        return self.shard_manager.is_duplicate(dataset_key, index)

    def _upload_current_batch(self) -> None:
        """Upload the current batch to HuggingFace."""
        if not self.current_batch:
            return

        try:
            logger.info(f"Uploading batch of {len(self.current_batch)} samples...")

            # Upload to shard
            shard_info = self.shard_manager.add_shard(self.current_batch)

            if shard_info:
                self.stats["uploaded"] += len(self.current_batch)
                logger.info(f"Successfully uploaded batch to {shard_info['path']}")
            else:
                logger.error("Failed to upload batch")

            # Clear batch
            self.current_batch.clear()

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.error(f"Error uploading batch: {e}")

    def _log_progress(self, current: int, total: int) -> None:
        """Log progress statistics."""
        progress_pct = (current / total) * 100
        logger.info(
            f"Progress: {current}/{total} ({progress_pct:.1f}%) - "
            f"Processed: {self.stats['processed']}, "
            f"Skipped: {self.stats['skipped']}, "
            f"Errors: {self.stats['errors']}, "
            f"Uploaded: {self.stats['uploaded']}"
        )

    def _cleanup(self) -> None:
        """Cleanup resources."""
        self.current_batch.clear()
        gc.collect()


class GlobalProgressTracker:
    """Separate class for global progress tracking."""

    def __init__(self, dataset_name: str, token: str):
        """Initialize progress tracker."""
        self.dataset_name = dataset_name
        self.token = token
        self.api = HfApi(token=token)
        self.progress_file = Path(f".progress_{dataset_name}.json")

    def load_progress(self) -> Dict[str, Any]:
        """Load progress from file or HuggingFace."""
        # Try local file first
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load progress file: {e}")

        # Try HuggingFace metadata
        try:
            # This would need to be implemented based on how progress
            # is stored in HF dataset metadata
            pass
        except Exception as e:
            logger.debug(f"Failed to load progress from HuggingFace: {e}")

        # Default progress
        return {
            "completed_datasets": [],
            "current_dataset": None,
            "current_index": 0,
            "total_samples": 0,
        }

    def save_progress(self, progress: Dict[str, Any]) -> None:
        """Save progress to file and HuggingFace."""
        # Save locally
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        # Save to HuggingFace
        try:
            # This would need to be implemented
            pass
        except Exception as e:
            logger.warning(f"Failed to save progress to HuggingFace: {e}")

    def mark_dataset_complete(self, dataset_key: str) -> None:
        """Mark a dataset as complete."""
        progress = self.load_progress()
        if dataset_key not in progress["completed_datasets"]:
            progress["completed_datasets"].append(dataset_key)
        progress["current_dataset"] = None
        progress["current_index"] = 0
        self.save_progress(progress)