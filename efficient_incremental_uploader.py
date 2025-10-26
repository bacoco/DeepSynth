#!/usr/bin/env python3
"""
Efficient incremental uploader that uploads every 100 batches (~5000 samples)
and cleans up batch files after successful upload to save disk space.
Maintains the same dataset name with incremental updates.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import login, whoami, HfApi

from hf_shard_uploader import HubShardManager

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
        self.shard_manager = HubShardManager(
            repo_id=self.dataset_name,
            token=self.hf_token,
            api=self.api,
        )
        self.upload_progress['dataset_created'] = True

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

    def load_single_batch(self, batch_id: int, batch_file: Path) -> Tuple[Optional[List[dict]], bool]:
        """Load samples from a single batch file."""

        if self.shard_manager.shard_exists(self.shard_manager.format_shard_id(batch_id)):
            print(f"  ⚠ Shard for batch {batch_id} already exists on the Hub, skipping upload")
            return None, True

        try:
            with open(batch_file, 'rb') as f:
                samples = pickle.load(f)
                print(f"  ✓ Loaded batch {batch_id}: {len(samples)} samples")
                return samples, False
        except Exception as e:
            print(f"  ✗ Failed to load batch {batch_id}: {e}")
            return None, False

    def upload_batch_chunk(self, batch_files):
        """Upload a chunk of batches and clean up after success"""
        print(f"\n📤 Uploading {len(batch_files)} batches...")

        total_uploaded = 0
        total_skipped = 0
        shards_uploaded = []

        for batch_id, batch_file in batch_files:
            samples, skipped_existing = self.load_single_batch(batch_id, batch_file)
            if skipped_existing:
                continue
            if samples is None:
                print(f"  ✗ Unable to read batch {batch_id}, aborting chunk")
                return False
            if not samples:
                print(f"  ⚠ Batch {batch_id} contains no samples, skipping")
                continue

            shard_id = self.shard_manager.format_shard_id(batch_id)
            try:
                result = self.shard_manager.upload_samples_as_shard(
                    samples,
                    shard_id=shard_id,
                    commit_message=f"Add shard {shard_id} from efficient uploader",
                )
            except Exception as exc:
                print(f"  ✗ Upload failed for batch {batch_id}: {exc}")
                return False

            total_uploaded += result.uploaded_samples
            total_skipped += result.skipped_duplicates

            if result.uploaded_samples:
                shards_uploaded.append(result.shard_id)
                print(f"  ✅ Uploaded shard {result.shard_id}: {result.uploaded_samples} samples")
            else:
                print(f"  ℹ️ Batch {batch_id} contained only duplicates ({result.skipped_duplicates} skipped)")

        if shards_uploaded:
            index_commit_message = f"Update shard index ({len(shards_uploaded)} new shards via efficient uploader)"
            try:
                self.shard_manager.save_index(commit_message=index_commit_message)
            except Exception as exc:
                print(f"  ✗ Failed to update shard index: {exc}")
                return False

        if total_uploaded == 0 and total_skipped == 0 and not shards_uploaded:
            print("  ⚠ Nothing to upload in this chunk")

        last_batch_id = max(batch_id for batch_id, _ in batch_files)
        self.upload_progress['last_uploaded_batch'] = last_batch_id
        self.upload_progress['total_uploaded_samples'] += total_uploaded
        if shards_uploaded:
            self.upload_progress['upload_count'] += len(shards_uploaded)
        self.save_upload_progress()

        self.cleanup_uploaded_batches(batch_files)

        print(f"  📈 Chunk summary: uploaded {total_uploaded} new samples (skipped {total_skipped} duplicates)")
        print(f"  ✅ Dataset: https://huggingface.co/datasets/{self.dataset_name}")
        return True

    def cleanup_uploaded_batches(self, batch_files):
        """DELETE uploaded batch files to save disk space"""
        print(f"  🧹 Deleting {len(batch_files)} batch files to save space...")

        for batch_id, batch_file in batch_files:
            try:
                # DELETE the file completely to save disk space
                batch_file.unlink()
                print(f"    ✓ Deleted batch {batch_id}")
            except Exception as e:
                print(f"    ✗ Failed to delete batch {batch_id}: {e}")

    def upload_all_pending(self):
        """Upload all pending batches in chunks"""
        print("🎯 EFFICIENT INCREMENTAL UPLOAD")
        print("=" * 50)
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"📊 Upload every {self.batches_per_upload} batches (~{self.batches_per_upload * 50} samples)")

        pending_batches = self.get_pending_batches()

        if not pending_batches:
            print("⚠ No pending batches to upload")
            return

        print(f"📊 Found {len(pending_batches)} pending batches")

        # Upload in chunks
        for i in range(0, len(pending_batches), self.batches_per_upload):
            chunk = pending_batches[i:i + self.batches_per_upload]

            print(f"\\n📦 Chunk {i//self.batches_per_upload + 1}: batches {chunk[0][0]} to {chunk[-1][0]}")

            success = self.upload_batch_chunk(chunk)

            if not success:
                print(f"❌ Failed to upload chunk, stopping")
                break

            # Small delay to avoid rate limiting
            time.sleep(3)

        print(f"\\n🎉 Upload complete!")
        print(f"📊 Total uploads: {self.upload_progress['upload_count']}")
        print(f"📊 Total samples: {self.upload_progress['total_uploaded_samples']}")
        print(f"📊 Dataset: https://huggingface.co/datasets/{self.dataset_name}")

    def should_upload_now(self):
        """Check if we should upload now (every batches_per_upload batches)"""
        pending = self.get_pending_batches()
        return len(pending) >= self.batches_per_upload

    def upload_if_ready(self):
        """Upload if we have enough pending batches"""
        if self.should_upload_now():
            print(f"\\n🔄 Auto-upload triggered ({self.batches_per_upload} batches ready)")

            pending_batches = self.get_pending_batches()
            chunk = pending_batches[:self.batches_per_upload]

            return self.upload_batch_chunk(chunk)

        return False

def main():
    uploader = EfficientIncrementalUploader(batches_per_upload=100)
    uploader.upload_all_pending()

if __name__ == "__main__":
    main()
