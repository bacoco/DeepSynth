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
from huggingface_hub import login, whoami, HfApi, hf_hub_download
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

    def load_index_registry(self):
        """Load the remote index registry tracking processed samples."""
        try:
            registry_path = hf_hub_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                filename="metadata/index_registry.json",
                token=self.hf_token,
            )
            with open(registry_path, "r") as f:
                raw_registry = json.load(f)

            registry = {}
            for source, splits in raw_registry.items():
                registry[source] = {}
                for split, max_index in splits.items():
                    try:
                        registry[source][split] = int(max_index)
                    except (TypeError, ValueError):
                        registry[source][split] = -1
            return registry
        except Exception:
            return {}

    def upload_index_registry(self, registry, commit_message):
        """Persist the updated index registry to HuggingFace."""
        metadata_dir = self.work_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        local_path = metadata_dir / "index_registry.json"
        with open(local_path, "w") as f:
            json.dump(registry, f, indent=2, sort_keys=True)

        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="metadata/index_registry.json",
            repo_id=self.dataset_name,
            repo_type="dataset",
            token=self.hf_token,
            commit_message=commit_message,
        )

    def filter_duplicate_samples(self, samples, registry):
        """Remove samples already present according to the registry."""
        if not samples:
            return samples, registry, 0

        filtered = []
        seen_in_chunk = set()
        updates = {}
        skipped = 0

        for sample in samples:
            source = sample.get("source_dataset")
            split = sample.get("original_split", "train") or "train"
            original_index = sample.get("original_index")

            # If we can't identify the sample, keep it but don't update registry
            if source is None or original_index is None:
                filtered.append(sample)
                continue

            try:
                original_index = int(original_index)
            except (TypeError, ValueError):
                filtered.append(sample)
                continue

            key = (source, split, original_index)

            if key in seen_in_chunk:
                skipped += 1
                continue

            seen_in_chunk.add(key)

            existing_max = registry.get(source, {}).get(split, -1)
            if original_index <= existing_max:
                skipped += 1
                continue

            filtered.append(sample)

            new_max = max(updates.get((source, split), existing_max), original_index)
            updates[(source, split)] = new_max

        if updates:
            for (source, split), max_index in updates.items():
                registry.setdefault(source, {})
                registry[source][split] = max(registry[source].get(split, -1), max_index)

        return filtered, registry, skipped

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
        dataset = dataset.cast_column("image", ImageFeature())

        return DatasetDict({'train': dataset})

    def upload_batch_chunk(self, batch_files):
        """Upload a chunk of batches and clean up after success"""
        print(f"\\n📤 Uploading {len(batch_files)} batches...")

        # Load samples
        samples = self.load_batch_range(batch_files)
        if not samples:
            print("  ✗ Failed to load samples")
            return False

        print(f"  📊 Total samples in chunk: {len(samples)}")

        registry = self.load_index_registry()
        samples, registry, skipped = self.filter_duplicate_samples(samples, registry)

        if skipped:
            print(f"  ⚖️  Skipped {skipped} duplicate samples already present on HuggingFace")

        if not samples:
            print("  ℹ️ No new unique samples to upload after duplicate filtering")
            last_batch_id = max(batch_id for batch_id, _ in batch_files)
            self.upload_progress['last_uploaded_batch'] = last_batch_id
            self.save_upload_progress()
            self.cleanup_uploaded_batches(batch_files)
            return True

        # Create dataset from new samples
        new_dataset_dict = self.create_dataset_from_samples(samples)
        if not new_dataset_dict:
            print("  ✗ Failed to create dataset")
            return False

        try:
            print(f"  🚀 Uploading to: {self.dataset_name}")

            # For first upload, create new dataset
            if not self.upload_progress['dataset_created']:
                try:
                    self.api.delete_repo(repo_id=self.dataset_name, repo_type='dataset')
                    print("  🗑️ Deleted existing dataset")
                except:
                    print("  ℹ️ No existing dataset to delete")

                # Upload new dataset
                final_dataset = new_dataset_dict
                print("  📝 Creating new dataset")
            else:
                # APPEND to existing dataset
                print("  📝 Loading existing dataset to append...")
                try:
                    from datasets import load_dataset
                    existing_dataset = load_dataset(self.dataset_name, token=self.hf_token)

                    # Combine existing and new data
                    from datasets import concatenate_datasets
                    combined_train = concatenate_datasets([
                        existing_dataset['train'],
                        new_dataset_dict['train']
                    ])
                    final_dataset = DatasetDict({'train': combined_train})
                    print(f"  ✅ Combined: {len(existing_dataset['train'])} + {len(new_dataset_dict['train'])} = {len(combined_train)} samples")

                except Exception as e:
                    print(f"  ⚠ Failed to load existing dataset, creating new: {e}")
                    final_dataset = new_dataset_dict

            # Upload combined dataset
            final_dataset.push_to_hub(
                self.dataset_name,
                private=False,
                token=self.hf_token,
                commit_message=f"Incremental update: +{len(samples)} samples (upload #{self.upload_progress['upload_count'] + 1})"
            )

            print(f"  ✅ Upload successful: https://huggingface.co/datasets/{self.dataset_name}")

            # Persist updated registry to Hub
            try:
                self.upload_index_registry(
                    registry,
                    commit_message=f"Update registry after upload #{self.upload_progress['upload_count'] + 1}"
                )
            except Exception as registry_error:
                print(f"  ⚠ Failed to update index registry: {registry_error}")

            # Update progress
            last_batch_id = max(batch_id for batch_id, _ in batch_files)
            self.upload_progress['last_uploaded_batch'] = last_batch_id
            self.upload_progress['total_uploaded_samples'] += len(samples)
            self.upload_progress['upload_count'] += 1
            self.upload_progress['dataset_created'] = True
            self.save_upload_progress()

            # Clean up batch files after successful upload
            self.cleanup_uploaded_batches(batch_files)

            print(f"  📈 Total uploaded: {self.upload_progress['total_uploaded_samples']} samples")
            return True

        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
            return False

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
