#!/usr/bin/env python3
"""
Test suite for HubShardManager upload structure.
Ensures datasets are uploaded with correct structure without DatasetDict wrapper.
"""

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deepsynth.data.hub import HubShardManager
from deepsynth.config import load_shared_env
from huggingface_hub import HfApi

# Load environment
load_shared_env()


def create_test_image(text="Test"):
    """Create a minimal test image"""
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill='black')
    return img


class TestHubShardStructure:
    """Test the HubShardManager upload structure"""

    @pytest.fixture
    def test_samples(self):
        """Create test samples with unique indices"""
        import time
        import random
        # Use timestamp and random to create unique indices
        base_idx = int(time.time() * 1000) + random.randint(0, 100000)
        return [
            {
                'text': f'Test document {i}',
                'summary': f'Summary {i}',
                'image': create_test_image(f"Test {i}"),
                'source_dataset': f'test_dataset_{base_idx}',
                'original_split': 'test',
                'original_index': base_idx + i
            }
            for i in range(3)
        ]

    @pytest.fixture
    def hf_token(self):
        """Get HuggingFace token from environment"""
        token = os.getenv('HF_TOKEN')
        if not token:
            pytest.skip("HF_TOKEN not available")
        return token

    def test_upload_structure_no_dataset_dict(self, test_samples, hf_token):
        """Test that uploads don't create dataset_dict.json"""
        import time
        test_repo = "baconnier/deepsynth-test-structure"

        manager = HubShardManager(
            repo_id=test_repo,
            token=hf_token,
            api=HfApi()
        )

        # Use unique shard_id
        shard_id = f"test_struct_{int(time.time() * 1000)}"

        result = manager.upload_samples_as_shard(
            samples=test_samples,
            shard_id=shard_id,
            commit_message="Test structure validation"
        )

        assert result.uploaded_samples == len(test_samples)

        manager.save_index()

        # Verify structure
        api = HfApi()
        files = api.list_repo_files(test_repo, repo_type='dataset')

        # Check expected files exist
        expected_files = [
            f"data/{shard_id}/data-00000-of-00001.arrow",
            f"data/{shard_id}/dataset_info.json",
            f"data/{shard_id}/state.json",
        ]

        for expected in expected_files:
            assert expected in files, f"Missing expected file: {expected}"

        # Check that problematic file does NOT exist
        bad_file = f"data/{shard_id}/dataset_dict.json"
        assert bad_file not in files, f"Found problematic file: {bad_file}"

    def test_upload_preserves_all_columns(self, test_samples, hf_token):
        """Test that all columns are preserved in upload"""
        import time
        test_repo = "baconnier/deepsynth-test-structure"

        manager = HubShardManager(
            repo_id=test_repo,
            token=hf_token,
            api=HfApi()
        )

        # Use unique shard_id
        shard_id = f"test_columns_{int(time.time() * 1000)}"

        result = manager.upload_samples_as_shard(
            samples=test_samples,
            shard_id=shard_id
        )

        assert result.uploaded_samples == len(test_samples)

        # Load dataset back to verify columns
        from datasets import load_dataset

        dataset = load_dataset(
            test_repo,
            split='train',
            data_files=f"data/{shard_id}/data-00000-of-00001.arrow"
        )

        # Check all expected columns exist
        expected_columns = ['text', 'summary', 'image', 'source_dataset',
                          'original_split', 'original_index']

        for col in expected_columns:
            assert col in dataset.column_names, f"Missing column: {col}"

    def test_duplicate_prevention(self, test_samples, hf_token):
        """Test that duplicate samples are prevented"""
        import time
        test_repo = "baconnier/deepsynth-test-structure"

        manager = HubShardManager(
            repo_id=test_repo,
            token=hf_token,
            api=HfApi()
        )

        # Use unique shard_ids
        base_id = int(time.time() * 1000)
        shard_id_1 = f"test_dup_{base_id}"

        # First upload
        result1 = manager.upload_samples_as_shard(
            samples=test_samples,
            shard_id=shard_id_1
        )

        assert result1.uploaded_samples == len(test_samples)
        assert result1.skipped_duplicates == 0

        # Save index
        manager.save_index()

        # Create new manager to simulate fresh session
        manager2 = HubShardManager(
            repo_id=test_repo,
            token=hf_token,
            api=HfApi()
        )

        shard_id_2 = f"test_dup_{base_id + 1}"

        # Try to upload same samples again
        result2 = manager2.upload_samples_as_shard(
            samples=test_samples,
            shard_id=shard_id_2
        )

        # Should skip all duplicates
        assert result2.uploaded_samples == 0
        assert result2.skipped_duplicates == len(test_samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
