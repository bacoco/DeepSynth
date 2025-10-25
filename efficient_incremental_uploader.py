#!/usr/bin/env python3
"""
Efficient incremental uploader that uploads every 100 batches (~5000 samples)
and cleans up batch files after successful upload to save disk space.
Maintains the same dataset name with incremental updates.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import login, whoami, HfApi
import time

# Load environment variables
env_file = Path('.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

class EfficientIncrementalUploader:
    def __init__(self, work_dir="./work", batches_per_upload=100):
        self.work_dir = Path(work_dir)
        self.samples_dir = self.work_dir / "samples"
        self.uploaded_dir = self.work_dir / "uploaded"  # Archive uploaded batches
        self.upload_progress_file = self.work_dir / "upload_progress.json"
        self.batches_per_upload = batches_per_upload  # ~5000 samples

        self.uploaded_dir.mkdir(exist_ok=True)

        self.hf_token = os.getenv('HF_TOKEN')
        login(token=self.hf_token)
        self.username = whoami()['name']
        self.api = HfApi()
        self.dataset_name = f"{self.username}/deepseek-vision-complete"

        self.upload_progress = self.load_upload_progress()

    def load_upload_progress(self):
        if self.upload_progress_file.exists():
            with open(self.upload_progress_file, 'r') as f:
                progress = json.load(f)
                # Handle old format
                if 'total_uploaded' in progress and 'total_uploaded_samples' not in progress:
                    progress['total_uploaded_samples'] = progress['total_uploaded']
                if 'dataset_created' not in progress:
                    progress['dataset_created'] = progress.get('upload_count', 0) > 0
                return progress
        return {
            'last_uploaded_batch': -1,
            'total_uploaded_samples': 0,
            'upload_count': 0,
            'dataset_created': False
        }

    def save_upload_progress(self):
        with open(self.upload_progress_file, 'w') as f:
            json.dump(self.upload_progress, f, indent=2)

    def get_pending_batches(self):
        """Get list of batch files that haven't been uploaded yet"""
        batch_files = sorted(self.samples_dir.glob("batch_*.pkl"))
        last_uploaded = self.upload_progress['last_uploaded_batch']

        pending = []
        for batch_file in batch_files:
            batch_id = int(batch_file.stem.split('_')[1])
            if batch_id > last_uploaded:
                pending.append((batch_id, batch_file))

        return pending

    def load_batch_range(self, batch_files):
        """Load samples from a range of batch files"""
        all_samples = []

        for batch_id, batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    samples = pickle.load(f)
                    all_samples.extend(samples)
                    print(f"  ✓ Loaded batch {batch_id}: {len(samples)} samples")
            except Exception as e:
                print(f"  ✗ Failed to load batch {batch_id}: {e}")
                return None

        return all_samples

    def create_dataset_from_samples(self, samples):
        """Create HuggingFace dataset from samples"""
        if not samples:
            return None

        dataset = Dataset.from_dict({
            'text': [s['text'] for s in samples],
            'summary': [s['summary'] for s in samples],
            'image': [s['image'] for s in samples],
            'source_dataset': [s['source_dataset'] for s in samples],
            'original_split': [s['original_split'] for s in samples],
            'original_index': [s['original_index'] for s in samples],
        })

        from datasets import Image as ImageFeature
        dataset = dataset.cast_c