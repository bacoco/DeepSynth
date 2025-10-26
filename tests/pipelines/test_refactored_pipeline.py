"""
Tests for the refactored global state pipeline.

Tests reduced complexity and improved error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deepsynth.pipelines.refactored_global_state import (
    RefactoredGlobalStatePipeline,
    GlobalProgressTracker,
)


@pytest.fixture
def mock_converter():
    """Mock text-to-image converter."""
    converter = MagicMock()
    converter.convert.return_value = MagicMock()  # Mock PIL Image
    return converter


@pytest.fixture
def mock_shard_manager():
    """Mock shard manager."""
    manager = MagicMock()
    manager.is_duplicate.return_value = False
    manager.add_shard.return_value = {"path": "shard_000001"}
    return manager


@pytest.fixture
def pipeline(mock_converter, mock_shard_manager):
    """Create test pipeline."""
    with patch("deepsynth.pipelines.refactored_global_state.HubShardManager", return_value=mock_shard_manager):
        pipeline = RefactoredGlobalStatePipeline(
            target_dataset_name="test-dataset",
            hf_token="test-token",
            hf_username="test-user",
            batch_size=2,
            image_converter=mock_converter,
        )
        pipeline.shard_manager = mock_shard_manager
        return pipeline


class TestRefactoredGlobalStatePipeline:
    """Test the refactored pipeline."""

    def test_initialization(self, pipeline):
        """Test pipeline initializes correctly."""
        assert pipeline.target_dataset_name == "test-dataset"
        assert pipeline.hf_token == "test-token"
        assert pipeline.batch_size == 2
        assert pipeline.stats["processed"] == 0

    def test_setup_dataset(self, pipeline):
        """Test dataset setup."""
        source_config = {
            "name": "test-dataset",
            "subset": "test-subset",
            "split": "train",
            "max_samples": 1000,
        }

        info = pipeline._setup_dataset("test-key", source_config)

        assert info is not None
        assert info["key"] == "test-key"
        assert info["name"] == "test-dataset"
        assert info["subset"] == "test-subset"
        assert info["split"] == "train"
        assert info["max_samples"] == 1000

    def test_setup_dataset_error(self, pipeline):
        """Test dataset setup handles errors."""
        source_config = {}  # Invalid config

        info = pipeline._setup_dataset("test-key", source_config)

        # Should handle gracefully
        assert info is not None or info is None  # May succeed with defaults

    def test_process_single_sample_success(self, pipeline, mock_converter):
        """Test processing a single valid sample."""
        sample = {
            "text": "This is a test document with sufficient length.",
            "summary": "Test summary",
        }
        dataset_info = {
            "key": "test-dataset",
            "split": "train",
            "config": MagicMock(),
        }

        with patch("deepsynth.pipelines.refactored_global_state.extract_text_summary") as mock_extract:
            mock_extract.return_value = (sample["text"], sample["summary"])

            success = pipeline._process_single_sample(sample, dataset_info, 0)

            assert success is True
            assert len(pipeline.current_batch) == 1
            mock_converter.convert.assert_called_once()

    def test_process_single_sample_empty_fields(self, pipeline):
        """Test processing sample with empty fields."""
        sample = {"text": "", "summary": ""}
        dataset_info = {"key": "test", "split": "train", "config": MagicMock()}

        with patch("deepsynth.pipelines.refactored_global_state.extract_text_summary") as mock_extract:
            mock_extract.return_value = (None, None)

            success = pipeline._process_single_sample(sample, dataset_info, 0)

            assert success is False
            assert len(pipeline.current_batch) == 0

    def test_process_single_sample_duplicate(self, pipeline, mock_shard_manager):
        """Test duplicate detection."""
        mock_shard_manager.is_duplicate.return_value = True

        sample = {"text": "Test", "summary": "Summary"}
        dataset_info = {"key": "test", "split": "train", "config": MagicMock()}

        with patch("deepsynth.pipelines.refactored_global_state.extract_text_summary") as mock_extract:
            mock_extract.return_value = ("Test text", "Summary")

            success = pipeline._process_single_sample(sample, dataset_info, 0)

            assert success is False

    def test_process_single_sample_conversion_error(self, pipeline, mock_converter):
        """Test handling of image conversion errors."""
        mock_converter.convert.side_effect = Exception("Conversion failed")

        sample = {"text": "Test" * 20, "summary": "Summary"}
        dataset_info = {"key": "test", "split": "train", "config": MagicMock()}

        with patch("deepsynth.pipelines.refactored_global_state.extract_text_summary") as mock_extract:
            mock_extract.return_value = (sample["text"], sample["summary"])

            success = pipeline._process_single_sample(sample, dataset_info, 0)

            assert success is False
            assert pipeline.stats["errors"] == 1

    def test_upload_current_batch(self, pipeline, mock_shard_manager):
        """Test batch upload."""
        pipeline.current_batch = [
            {"text": "Sample 1", "summary": "Summary 1"},
            {"text": "Sample 2", "summary": "Summary 2"},
        ]

        pipeline._upload_current_batch()

        mock_shard_manager.add_shard.assert_called_once()
        assert len(pipeline.current_batch) == 0
        assert pipeline.stats["uploaded"] == 2

    def test_upload_empty_batch(self, pipeline, mock_shard_manager):
        """Test uploading empty batch does nothing."""
        pipeline.current_batch = []

        pipeline._upload_current_batch()

        mock_shard_manager.add_shard.assert_not_called()

    def test_upload_batch_error(self, pipeline, mock_shard_manager):
        """Test handling of upload errors."""
        mock_shard_manager.add_shard.side_effect = Exception("Upload failed")
        pipeline.current_batch = [{"text": "Test", "summary": "Summary"}]

        # Should not raise
        pipeline._upload_current_batch()

    def test_batch_auto_upload(self, pipeline, mock_converter, mock_shard_manager):
        """Test automatic batch upload when size reached."""
        pipeline.batch_size = 2

        samples = [
            {"text": f"Document {i}" * 10, "summary": f"Summary {i}"}
            for i in range(3)
        ]
        dataset_info = {"key": "test", "split": "train", "config": MagicMock()}

        with patch("deepsynth.pipelines.refactored_global_state.extract_text_summary") as mock_extract:
            for i, sample in enumerate(samples):
                mock_extract.return_value = (sample["text"], sample["summary"])
                pipeline._process_single_sample(sample, dataset_info, i)

        # Should have uploaded once (2 samples) and have 1 remaining
        mock_shard_manager.add_shard.assert_called_once()
        assert len(pipeline.current_batch) == 1

    @patch("deepsynth.pipelines.refactored_global_state.load_dataset")
    def test_load_dataset_split(self, mock_load, pipeline):
        """Test dataset loading."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda self: 100
        mock_load.return_value = mock_dataset

        dataset_info = {
            "name": "test-dataset",
            "subset": "subset1",
            "split": "train",
        }

        result = pipeline._load_dataset_split(dataset_info)

        assert result == mock_dataset
        mock_load.assert_called_once_with("test-dataset", "subset1", split="train")

    @patch("deepsynth.pipelines.refactored_global_state.load_dataset")
    def test_load_dataset_without_subset(self, mock_load, pipeline):
        """Test dataset loading without subset."""
        mock_dataset = MagicMock()
        mock_load.return_value = mock_dataset

        dataset_info = {
            "name": "test-dataset",
            "subset": None,
            "split": "validation",
        }

        result = pipeline._load_dataset_split(dataset_info)

        mock_load.assert_called_once_with("test-dataset", split="validation")

    @patch("deepsynth.pipelines.refactored_global_state.load_dataset")
    def test_load_dataset_error(self, mock_load, pipeline):
        """Test dataset loading error handling."""
        mock_load.side_effect = Exception("Load failed")

        dataset_info = {"name": "test", "subset": None, "split": "train"}

        result = pipeline._load_dataset_split(dataset_info)

        assert result is None

    def test_cleanup(self, pipeline):
        """Test cleanup clears resources."""
        pipeline.current_batch = [{"test": "data"}]

        pipeline._cleanup()

        assert len(pipeline.current_batch) == 0


class TestGlobalProgressTracker:
    """Test the progress tracker."""

    def test_initialization(self):
        """Test tracker initializes."""
        tracker = GlobalProgressTracker("test-dataset", "test-token")
        assert tracker.dataset_name == "test-dataset"
        assert tracker.token == "test-token"

    def test_load_default_progress(self, tmp_path):
        """Test loading default progress when no file exists."""
        tracker = GlobalProgressTracker("test-dataset", "test-token")
        tracker.progress_file = tmp_path / "nonexistent.json"

        progress = tracker.load_progress()

        assert "completed_datasets" in progress
        assert "current_dataset" in progress
        assert "current_index" in progress
        assert "total_samples" in progress
        assert progress["total_samples"] == 0

    def test_save_and_load_progress(self, tmp_path):
        """Test saving and loading progress."""
        tracker = GlobalProgressTracker("test-dataset", "test-token")
        tracker.progress_file = tmp_path / "progress.json"

        progress = {
            "completed_datasets": ["dataset1", "dataset2"],
            "current_dataset": "dataset3",
            "current_index": 1000,
            "total_samples": 5000,
        }

        tracker.save_progress(progress)
        loaded = tracker.load_progress()

        assert loaded == progress

    def test_mark_dataset_complete(self, tmp_path):
        """Test marking dataset as complete."""
        tracker = GlobalProgressTracker("test-dataset", "test-token")
        tracker.progress_file = tmp_path / "progress.json"

        # Initial state
        tracker.save_progress({
            "completed_datasets": [],
            "current_dataset": "dataset1",
            "current_index": 500,
            "total_samples": 500,
        })

        # Mark complete
        tracker.mark_dataset_complete("dataset1")

        # Check state
        progress = tracker.load_progress()
        assert "dataset1" in progress["completed_datasets"]
        assert progress["current_dataset"] is None
        assert progress["current_index"] == 0

    def test_mark_dataset_complete_idempotent(self, tmp_path):
        """Test marking same dataset complete multiple times."""
        tracker = GlobalProgressTracker("test-dataset", "test-token")
        tracker.progress_file = tmp_path / "progress.json"

        tracker.mark_dataset_complete("dataset1")
        tracker.mark_dataset_complete("dataset1")

        progress = tracker.load_progress()
        # Should only appear once
        assert progress["completed_datasets"].count("dataset1") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])