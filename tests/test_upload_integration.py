#!/usr/bin/env python3
"""
Integration tests for the complete upload pipeline.
Tests the flow from dataset processing to HuggingFace upload.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepsynth.pipelines._dataset_processor import OptimizedDatasetPipeline
from deepsynth.config import load_shared_env
from huggingface_hub import HfApi, login

# Load environment
load_shared_env()


class TestUploadIntegration:
    """Integration tests for upload pipeline"""

    @pytest.fixture
    def hf_token(self):
        """Get HuggingFace token"""
        token = os.getenv('HF_TOKEN')
        if not token:
            pytest.skip("HF_TOKEN not available")
        login(token=token)
        return token

    @pytest.fixture
    def test_pipeline(self):
        """Create a test pipeline"""
        work_dir = "./test_work_integration"
        pipeline = OptimizedDatasetPipeline(
            work_dir=work_dir,
            batch_size=10,
            auto_upload=True
        )
        yield pipeline

        # Cleanup
        import shutil
        if Path(work_dir).exists():
            shutil.rmtree(work_dir)

    def test_pipeline_creates_correct_structure(self, test_pipeline, hf_token):
        """Test that pipeline creates correct dataset structure"""
        from huggingface_hub import whoami
        username = whoami()['name']

        # Process a tiny dataset
        test_pipeline.process_and_batch_dataset(
            name='billsum',
            subset=None,
            text_field='text',
            summary_field='summary',
            username=username,
            max_samples=5
        )

        # Verify structure on HuggingFace
        repo_name = f"{username}/deepsynth-en-legal"
        api = HfApi()

        files = api.list_repo_files(repo_name, repo_type='dataset')

        # Should have shards.json
        assert 'data/shards.json' in files

        # Should have at least one batch folder
        batch_files = [f for f in files if f.startswith('data/batch_')]
        assert len(batch_files) > 0

        # Should NOT have dataset_dict.json
        bad_files = [f for f in files if 'dataset_dict.json' in f]
        assert len(bad_files) == 0, f"Found dataset_dict.json files: {bad_files}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
