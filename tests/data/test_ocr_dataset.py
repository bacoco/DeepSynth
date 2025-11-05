#!/usr/bin/env python3
"""Unit tests for OCR data module.

Tests cover:
- OCRDataset initialization
- Multi-format support (HuggingFace, WebDataset, Parquet)
- OCRDataLoader functionality
- Data collation
- WebDataset streaming
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.deepsynth.data.ocr import (
    OCRDataset,
    OCRDataLoader,
    OCRCollator,
    WebDatasetLoader,
    create_ocr_dataloader,
)


class TestOCRDataset:
    """Test OCRDataset class."""

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    def test_huggingface_dataset(self, mock_load):
        """Test loading from HuggingFace."""
        # Mock HF dataset
        mock_hf_dataset = Mock()
        mock_hf_dataset.__len__.return_value = 100
        mock_hf_dataset.__getitem__.return_value = {
            "text": "Sample text",
            "summary": "Sample summary",
        }
        mock_load.return_value = mock_hf_dataset

        # Create dataset
        dataset = OCRDataset(
            source="test/dataset",
            source_type="huggingface",
            text_field="text",
            summary_field="summary",
        )

        assert len(dataset) == 100
        sample = dataset[0]
        assert "text" in sample
        assert "summary" in sample

    @patch('src.deepsynth.data.ocr.dataset.pq')
    def test_parquet_dataset(self, mock_pq):
        """Test loading from Parquet."""
        # Mock Parquet table
        mock_table = Mock()
        mock_df = Mock()
        mock_df.__len__.return_value = 50
        mock_df.iloc = Mock()
        mock_table.to_pandas.return_value = mock_df
        mock_pq.read_table.return_value = mock_table

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            # Create dataset
            dataset = OCRDataset(
                source=f.name,
                source_type="parquet",
            )

            mock_pq.read_table.assert_called_once()

    def test_unsupported_type(self):
        """Test unsupported source type."""
        with pytest.raises(ValueError):
            OCRDataset(
                source="test",
                source_type="unsupported",
            )

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    def test_convenience_methods(self, mock_load):
        """Test convenience factory methods."""
        mock_load.return_value = Mock(__len__=lambda x: 10)

        # from_huggingface
        dataset = OCRDataset.from_huggingface("test/dataset")
        assert dataset.source_type == "huggingface"

        # from_webdataset
        with patch('src.deepsynth.data.ocr.dataset.wds'):
            dataset = OCRDataset.from_webdataset("http://example.com/data-{00..10}.tar")
            assert dataset.source_type == "webdataset"

        # from_parquet
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            with patch('src.deepsynth.data.ocr.dataset.pq'):
                dataset = OCRDataset.from_parquet(f.name)
                assert dataset.source_type == "parquet"

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    def test_dataset_info(self, mock_load):
        """Test dataset info method."""
        mock_hf_dataset = Mock(__len__=lambda x: 100)
        mock_load.return_value = mock_hf_dataset

        dataset = OCRDataset.from_huggingface("test/dataset")
        info = dataset.info()

        assert "num_samples" in info
        assert "source_type" in info
        assert info["num_samples"] == 100


class TestOCRCollator:
    """Test OCRCollator class."""

    def test_collation(self):
        """Test batch collation."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        mock_tokenizer.as_target_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        collator = OCRCollator(tokenizer=mock_tokenizer, max_length=512)

        # Create batch
        batch = [
            {"text": "Text 1", "summary": "Summary 1"},
            {"text": "Text 2", "summary": "Summary 2"},
        ]

        # Collate
        result = collator(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result


class TestOCRDataLoader:
    """Test OCRDataLoader class."""

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    def test_dataloader_creation(self, mock_load):
        """Test DataLoader creation."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 100
        mock_load.return_value = mock_dataset

        dataset = OCRDataset.from_huggingface("test/dataset")

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.as_target_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Create loader
        loader = OCRDataLoader(
            dataset,
            tokenizer=mock_tokenizer,
            batch_size=4,
            max_length=512,
        )

        assert loader.batch_size == 4
        assert loader.max_length == 512
        assert len(loader) > 0

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    def test_convenience_function(self, mock_load):
        """Test create_ocr_dataloader convenience function."""
        mock_dataset = Mock(__len__=lambda x: 100)
        mock_load.return_value = mock_dataset

        dataset = OCRDataset.from_huggingface("test/dataset")
        mock_tokenizer = Mock()
        mock_tokenizer.as_target_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Training loader
        train_loader = create_ocr_dataloader(
            dataset,
            mock_tokenizer,
            batch_size=8,
            is_training=True,
        )

        assert train_loader.batch_size == 8

        # Eval loader
        eval_loader = create_ocr_dataloader(
            dataset,
            mock_tokenizer,
            batch_size=16,
            is_training=False,
        )

        assert eval_loader.batch_size == 16


class TestWebDatasetLoader:
    """Test WebDatasetLoader class."""

    @patch('src.deepsynth.data.ocr.loader.wds')
    def test_webdataset_loader_creation(self, mock_wds):
        """Test WebDataset loader creation."""
        # Mock WebDataset
        mock_dataset = Mock()
        mock_wds.WebDataset.return_value = mock_dataset
        mock_dataset.decode.return_value = mock_dataset
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.to_tuple.return_value = mock_dataset
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.batched.return_value = mock_dataset

        mock_tokenizer = Mock()
        mock_tokenizer.as_target_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Create loader
        loader = WebDatasetLoader(
            url_pattern="http://example.com/data-{00..10}.tar",
            tokenizer=mock_tokenizer,
            batch_size=4,
            shuffle_buffer=1000,
        )

        assert loader.batch_size == 4
        mock_wds.WebDataset.assert_called_once()


class TestDataPipeline:
    """Test end-to-end data pipeline."""

    @patch('src.deepsynth.data.ocr.dataset.hf_load_dataset')
    def test_full_pipeline(self, mock_load):
        """Test complete data loading pipeline."""
        # Mock dataset
        mock_hf_dataset = []
        for i in range(10):
            mock_hf_dataset.append({
                "text": f"Text {i}",
                "summary": f"Summary {i}",
            })

        mock_dataset = Mock()
        mock_dataset.__len__.return_value = len(mock_hf_dataset)
        mock_dataset.__getitem__.side_effect = lambda i: mock_hf_dataset[i]
        mock_load.return_value = mock_dataset

        # Create dataset
        dataset = OCRDataset.from_huggingface("test/dataset")

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
        mock_tokenizer.as_target_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Create loader
        loader = create_ocr_dataloader(
            dataset,
            mock_tokenizer,
            batch_size=2,
            is_training=True,
        )

        # Verify loader is iterable
        assert hasattr(loader, '__iter__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
