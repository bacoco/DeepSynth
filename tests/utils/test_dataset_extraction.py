"""
Test suite for dataset extraction utilities.

Tests field extraction, validation, and dataset detection.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deepsynth.utils.dataset_extraction import (
    DatasetConfig,
    extract_text_summary,
    detect_dataset_type,
    batch_extract,
    DATASET_CONFIGS,
)


class TestDatasetConfig:
    """Test DatasetConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig(name="test")
        assert config.name == "test"
        assert config.text_field == "text"
        assert config.summary_field == "summary"
        assert config.validate_length == True
        assert config.min_text_length == 50
        assert config.max_text_length == 50000

    def test_custom_config(self):
        """Test custom configuration."""
        config = DatasetConfig(
            name="custom",
            text_field="document",
            summary_field="abstract",
            min_text_length=100,
            max_text_length=100000,
        )
        assert config.text_field == "document"
        assert config.summary_field == "abstract"
        assert config.min_text_length == 100


class TestExtractTextSummary:
    """Test text and summary extraction."""

    def test_standard_extraction(self):
        """Test extraction with standard fields."""
        example = {
            "text": "This is a test document with some content.",
            "summary": "Test summary",
        }
        text, summary = extract_text_summary(example)
        assert text == "This is a test document with some content."
        assert summary == "Test summary"

    def test_cnn_dailymail_extraction(self):
        """Test CNN/DailyMail specific extraction."""
        example = {
            "article": "News article content here.",
            "highlights": ["Key point 1", "Key point 2", "Key point 3"],
        }
        text, summary = extract_text_summary(example, dataset_name="cnn_dailymail")
        assert text == "News article content here."
        assert summary == "Key point 1 Key point 2 Key point 3"

    def test_xsum_extraction(self):
        """Test XSum specific extraction."""
        example = {
            "document": "Document content for XSum.",
            "summary": "XSum summary",
        }
        text, summary = extract_text_summary(example, dataset_name="xsum")
        assert text == "Document content for XSum."
        assert summary == "XSum summary"

    def test_dialogue_extraction(self):
        """Test dialogue dataset extraction."""
        example = {
            "dialogue": "Person A: Hello\nPerson B: Hi there",
            "summary": "Greeting exchange",
        }
        config = DatasetConfig(
            name="dialogue",
            dialogue_field="dialogue",
            summary_field="summary",
        )
        text, summary = extract_text_summary(example, dataset_config=config)
        assert text == "Person A: Hello\nPerson B: Hi there"
        assert summary == "Greeting exchange"

    def test_missing_fields(self):
        """Test handling of missing fields."""
        example = {"other_field": "value"}
        text, summary = extract_text_summary(example)
        assert text is None
        assert summary is None

    def test_empty_values(self):
        """Test handling of empty values."""
        example = {"text": "", "summary": "Summary"}
        text, summary = extract_text_summary(example)
        assert text is None
        assert summary is None

    def test_whitespace_trimming(self):
        """Test whitespace is trimmed."""
        example = {
            "text": "  Content with spaces  ",
            "summary": "\n\tSummary with tabs\n",
        }
        text, summary = extract_text_summary(example)
        assert text == "Content with spaces"
        assert summary == "Summary with tabs"

    def test_length_validation_too_short(self):
        """Test text that's too short is rejected."""
        example = {
            "text": "Short",
            "summary": "Summary",
        }
        config = DatasetConfig(name="test", min_text_length=10)
        text, summary = extract_text_summary(example, dataset_config=config)
        assert text is None
        assert summary is None

    def test_length_validation_too_long(self):
        """Test text that's too long is rejected."""
        example = {
            "text": "x" * 100000,
            "summary": "Summary",
        }
        config = DatasetConfig(name="test", max_text_length=50000)
        text, summary = extract_text_summary(example, dataset_config=config)
        assert text is None
        assert summary is None

    def test_length_validation_disabled(self):
        """Test length validation can be disabled."""
        example = {
            "text": "Short",
            "summary": "S",
        }
        config = DatasetConfig(name="test", validate_length=False)
        text, summary = extract_text_summary(example, dataset_config=config)
        assert text == "Short"
        assert summary == "S"

    def test_fallback_field_names(self):
        """Test fallback to common field names."""
        example = {
            "article": "Article content",
            "abstract": "Abstract content",
        }
        text, summary = extract_text_summary(example)
        assert text == "Article content"
        assert summary == "Abstract content"

    def test_list_to_string_conversion(self):
        """Test list fields are joined to strings."""
        example = {
            "text": ["Part 1", "Part 2", "Part 3"],
            "summary": ["Summary 1", "Summary 2"],
        }
        text, summary = extract_text_summary(example)
        assert text == "Part 1 Part 2 Part 3"
        assert summary == "Summary 1 Summary 2"


class TestDetectDatasetType:
    """Test dataset type detection."""

    def test_detect_cnn_dailymail(self):
        """Test CNN/DailyMail detection."""
        example = {"article": "...", "highlights": "..."}
        dataset_type = detect_dataset_type(example)
        assert dataset_type == "cnn_dailymail"

    def test_detect_xsum(self):
        """Test XSum detection."""
        example = {"document": "...", "summary": "..."}
        dataset_type = detect_dataset_type(example)
        assert dataset_type == "xsum"

    def test_detect_samsum(self):
        """Test SAMSum detection."""
        example = {"dialogue": "...", "summary": "..."}
        dataset_type = detect_dataset_type(example)
        assert dataset_type == "samsum"

    def test_detect_arxiv(self):
        """Test arXiv detection."""
        example = {"article": "...", "abstract": "..."}
        dataset_type = detect_dataset_type(example)
        assert dataset_type == "arxiv"

    def test_detect_pubmed(self):
        """Test PubMed detection."""
        example = {"article": "...", "abstract": "...", "journal": "..."}
        dataset_type = detect_dataset_type(example)
        assert dataset_type == "pubmed"

    def test_detect_generic(self):
        """Test generic dataset detection."""
        example = {"text": "...", "summary": "..."}
        dataset_type = detect_dataset_type(example)
        assert dataset_type == "generic"

    def test_detect_unknown(self):
        """Test unknown dataset returns None."""
        example = {"field1": "...", "field2": "..."}
        dataset_type = detect_dataset_type(example)
        assert dataset_type is None


class TestBatchExtract:
    """Test batch extraction."""

    def test_batch_extraction(self):
        """Test extracting multiple examples."""
        examples = [
            {"text": f"Document {i}" * 10, "summary": f"Summary {i}"}
            for i in range(5)
        ]
        results = batch_extract(examples)
        assert len(results) == 5
        assert all(text and summary for text, summary in results)

    def test_batch_with_invalid(self):
        """Test batch with some invalid examples."""
        examples = [
            {"text": "Valid document content here", "summary": "Valid summary"},
            {"text": "", "summary": "Invalid - empty text"},
            {"text": "Valid again", "summary": "Another valid"},
            {"other": "Missing required fields"},
        ]
        results = batch_extract(examples)
        assert len(results) == 4
        assert results[0][0] is not None  # First is valid
        assert results[1][0] is None  # Second is invalid
        assert results[2][0] is not None  # Third is valid
        assert results[3][0] is None  # Fourth is invalid

    def test_batch_with_auto_detection(self):
        """Test batch extraction with automatic dataset detection."""
        examples = [
            {"article": f"Article {i}", "highlights": [f"Point {i}"]}
            for i in range(3)
        ]
        results = batch_extract(examples)
        assert len(results) == 3
        assert all(text and summary for text, summary in results)

    def test_batch_with_config(self):
        """Test batch extraction with custom config."""
        config = DatasetConfig(
            name="test",
            text_field="content",
            summary_field="abstract",
        )
        examples = [
            {"content": f"Content {i}" * 10, "abstract": f"Abstract {i}"}
            for i in range(3)
        ]
        results = batch_extract(examples, dataset_config=config)
        assert len(results) == 3
        assert all(text and summary for text, summary in results)


class TestDatasetConfigs:
    """Test predefined dataset configurations."""

    def test_all_configs_have_required_fields(self):
        """Test all predefined configs have required fields."""
        for name, config in DATASET_CONFIGS.items():
            assert config.name == name
            assert config.min_text_length > 0
            assert config.max_text_length > config.min_text_length
            assert config.min_summary_length > 0
            assert config.max_summary_length > config.min_summary_length

    def test_config_lookup(self):
        """Test configuration lookup by name."""
        assert "cnn_dailymail" in DATASET_CONFIGS
        assert "xsum" in DATASET_CONFIGS
        assert "mlsum" in DATASET_CONFIGS
        assert "billsum" in DATASET_CONFIGS
        assert "samsum" in DATASET_CONFIGS

    def test_config_fields(self):
        """Test specific configurations have correct fields."""
        cnn_config = DATASET_CONFIGS["cnn_dailymail"]
        assert cnn_config.article_field == "article"
        assert cnn_config.highlights_field == "highlights"

        xsum_config = DATASET_CONFIGS["xsum"]
        assert xsum_config.text_field == "document"
        assert xsum_config.summary_field == "summary"

        samsum_config = DATASET_CONFIGS["samsum"]
        assert samsum_config.dialogue_field == "dialogue"
        assert samsum_config.summary_field == "summary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])