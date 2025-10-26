#!/usr/bin/env python3
"""
GLOBAL INCREMENTAL BUILDER - Works across multiple computers
Stores progress metadata in HuggingFace dataset to prevent duplicates
and enable seamless continuation from any machine.
"""

import os
from pathlib import Path

from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, login, whoami

from deepsynth.config import load_shared_env
from deepsynth.data.hub import HubShardManager
from deepsynth.data.loaders import MLSUMLoader
from deepsynth.data.transforms import TextToImageConverter

__all__ = ["GlobalIncrementalPipeline", "run_global_incremental_pipeline"]

# Load environment variables
load_shared_env()

class GlobalIncrementalPipeline:
    def __init__(self):
        self.hf_token = os.getenv('HF_TOKEN')
        login(token=self.hf_token)
        self.username = whoami()['name']
        self.api = HfApi()
        self.dataset_name = f"{self.username}/deepsynth-vision-complete"

        # Text-to-image converter
        unicode_font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        self.converter = TextToImageConverter(
            font_path=unicode_font_path,
            font_size=12, max_width=1600, max_height=2400, margin=40,
            background_color=(255, 255, 255), text_color=(0, 0, 0)
        )

        # Determine sampling strategy for large-scale datasets
        try:
            arxiv_limit = int(os.getenv('ARXIV_IMAGE_SAMPLES', '50000'))
        except ValueError:
            print("⚠ Invalid ARXIV_IMAGE_SAMPLES value. Falling back to 50000 samples.")
            arxiv_limit = 50000
        if arxiv_limit <= 0:
            print("⚠ ARXIV_IMAGE_SAMPLES must be positive. Falling back to 10000 samples.")
            arxiv_limit = 10000

        # Dataset sources configuration (name, subset, text_field, summary_field, expected_count, max_samples?)
        self.sources = [
            ('MLSUM', 'fr', 'text', 'summary', 392902),     # French
            ('MLSUM', 'es', 'text', 'summary', 266367),     # Spanish
            ('MLSUM', 'de', 'text', 'summary', 220748),     # German
            ('cnn_dailymail', '3.0.0', 'article', 'highlights', 287113),  # English news
            ('ccdv/arxiv-summarization', None, 'article', 'abstract', arxiv_limit, arxiv_limit),  # Scientific papers
            ('Rexhaif/xsum_reduced', None, 'text', 'target', 50000),      # English BBC
            ('billsum', None, 'text', 'summary', 22218),    # Legal documents
        ]

    def get_global_progress(self):
        """Load global progress from the shard index metadata."""

        metadata = self.shard_manager.index.get('metadata', {})
        progress = metadata.get('global_progress')

        if progress:
            print(f"📊 Loaded global progress from shard index: {progress}")
            return progress

        total_samples = sum(entry.get('num_samples', 0) for entry in self.shard_manager.index.get('shards', []))
        print(f"📊 No stored progress metadata found. Reconstructing from shard index ({total_samples} samples)")

        completed_datasets = []
        remaining_samples = total_samples

        for source in self.sources:
            name, subset, _, _, expected_count, *rest = source
            dataset_key = f"{name}_{subset}" if subset else name
            if remaining_samples >= expected_count:
                completed_datasets.append(dataset_key)
                remaining_samples -= expected_count
                print(f"  ✅ {dataset_key}: Complete ({expected_count} samples)")
            else:
                print(f"  🔄 {dataset_key}: Partial ({remaining_samples}/{expected_count} samples)")
                break

        return {
            'completed_datasets': completed_datasets,
            'current_dataset': None,
            'current_index': remaining_samples,
            'total_samples': total_samples
        }

    def save_global_progress(self, progress):
        """Save progress to HuggingFace dataset metadata"""
        try:
            # Create metadata description with progress
            metadata = {
                'global_progress': progress,
                'description': 'DeepSynth multilingual summarization dataset with text-image pairs',
                'total_expected': sum(entry[4] for entry in self.sources)
            }

            # Update dataset card with progress
            card_content = f"""# DeepSynth Multilingual Summarization Dataset

## Storage Layout
- Data is stored as independent shards under `data/`.
- The file `{self.shard_manager.index_path}` lists every shard along with the `(source_dataset, original_index)` pairs they contain.
- Use the repository script `scripts/check_shards_duplicates.py` to stream shards locally and verify integrity.

### Loading example
```python
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_from_disk

repo_id = "{self.dataset_name}"
index_path = hf_hub_download(repo_id, filename="{self.shard_manager.index_path}", repo_type="dataset")

with open(index_path, "r", encoding="utf-8") as handle:
    index = json.load(handle)

for shard in index["shards"]:
    repo_dir = snapshot_download(repo_id, repo_type="dataset", allow_patterns=[f"{shard['path']}/*"])
    dataset = load_from_disk(Path(repo_dir) / shard["path"])
    for row in dataset["train"]:
        ...  # consume the samples
```

## Progress Status
- **Total Samples**: {progress['total_samples']:,}
- **Completed Datasets**: {', '.join(progress['completed_datasets']) or 'None'}
- **Current Dataset**: {progress.get('current_dataset', 'None')}
- **Current Index**: {progress['current_index']:,}

## Dataset Sources
{chr(10).join([f"- {name} {subset or ''}: {count:,} samples" for name, subset, _, _, count, *rest in self.sources])}

**Auto-generated by global_incremental_builder.py**
"""

            # Save metadata
            self.api.upload_file(
                path_or_fileobj=card_content.encode(),
                path_in_repo="README.md",
                repo_id=self.dataset_name,
                repo_type="dataset",
                commit_message=f"Update progress: {progress['total_samples']} samples"
            )

            print(f"💾 Saved global progress: {progress['total_samples']} samples")

            # Persist the shard index metadata update
            self.shard_manager.save_index(commit_message="Update shard index metadata")

        except Exception as e:
            print(f"⚠ Failed to save progress metadata: {e}")

    def process_dataset_incremental(self, name, subset, text_field, summary_field, expected_count, max_samples=None):
        """Process dataset incrementally with global state tracking"""
        dataset_key = f"{name}_{subset}" if subset else name

        # Get current progress
        progress = self.get_global_progress()

        # Check if this dataset is already completed
        if dataset_key in progress['completed_datasets']:
            print(f"✅ {dataset_key} already completed, skipping")
            return

        # Determine starting index
        if progress['current_dataset'] == dataset_key:
            start_idx = progress['current_index']
        else:
            start_idx = 0

        print(f"🔄 Processing {dataset_key} from index {start_idx}")

        # Load dataset
        if name == 'MLSUM':
            mlsum_loader = MLSUMLoader()
            dataset_dict = mlsum_loader.load_language(subset)
            dataset = dataset_dict['train']
        else:
            from datasets import load_dataset
            if subset:
                dataset = load_dataset(name, subset, split='train')
            else:
                dataset = load_dataset(name, split='train')

        target_limit = len(dataset)
        if max_samples is not None:
            target_limit = min(target_limit, max_samples)

        print(f"📊 Dataset loaded: {len(dataset)} samples")
        if target_limit < len(dataset):
            print(f"🎯 Sampling first {target_limit:,} examples for image conversion")

        if start_idx >= target_limit:
            print(f"✅ {dataset_key} already processed up to configured limit ({target_limit} samples)")
            if dataset_key not in progress['completed_datasets']:
                progress['completed_datasets'].append(dataset_key)
            progress['current_dataset'] = None
            progress['current_index'] = 0
            self.save_global_progress(progress)
            return True

        # Process in batches - Large batch size since we clean up after upload
        batch_size = 10000  # 10K samples per batch for maximum efficiency
        batch_samples = []

        for idx in range(start_idx, target_limit):
            example = dataset[idx]

            # Extract text and summary
            if name == 'cnn_dailymail':
                text, summary = example.get('article', ''), example.get('highlights', '')
            elif name == 'billsum':
                text, summary = example.get('text', ''), example.get('summary', '')
            elif 'xsum' in name.lower():
                text = example.get('text', example.get('document', ''))
                summary = example.get('target', example.get('summary', ''))
            else:
                text, summary = example.get(text_field, ''), example.get(summary_field, '')

            if not text or not summary:
                continue

            # Convert to image
            try:
                image = self.converter.convert(text)
                batch_samples.append({
                    'text': text,
                    'summary': summary,
                    'image': image,
                    'source_dataset': name,
                    'original_split': 'train',
                    'original_index': idx
                })
            except Exception as e:
                print(f"⚠ Error processing sample {idx}: {e}")
                continue

            # Upload batch when ready
            if len(batch_samples) >= batch_size:
                print(f"🚀 Uploading batch: {len(batch_samples)} samples (Memory efficient)")
                uploaded_count = self.upload_batch_append(batch_samples)
                if uploaded_count is not None:
                    # Update progress
                    progress['current_dataset'] = dataset_key
                    progress['current_index'] = idx + 1
                    progress['total_samples'] += uploaded_count_count
                    self.save_global_progress(progress)

                    # Clear batch to free memory
                    batch_samples.clear()
                    print(f"📈 Progress: {idx + 1:,}/{target_limit:,} ({(idx + 1)/target_limit*100:.1f}%) - {progress['total_samples']:,} total samples")

                    # Force garbage collection for memory efficiency
                    import gc
                    gc.collect()
                else:
                    print(f"❌ Upload failed, stopping at index {idx}")
                    return False

                # Update progress
                progress['current_dataset'] = dataset_key
                progress['current_index'] = idx + 1
                progress['total_samples'] += uploaded_count
                self.save_global_progress(progress)

                # Clear batch to free memory
                batch_samples.clear()
                print(f"📈 Progress: {idx + 1:,}/{target_limit:,} ({(idx + 1)/target_limit*100:.1f}%) - {progress['total_samples']:,} total samples")

                # Force garbage collection for memory efficiency
                import gc
                gc.collect()

            # Progress reporting every 1000 samples
            if (idx + 1) % 1000 == 0:
                print(f"  📊 Processed {idx + 1:,}/{target_limit:,} samples ({len(batch_samples)} in current batch)")

        # Upload remaining samples
        if batch_samples:
            uploaded_count = self.upload_batch_append(batch_samples)
            if uploaded_count is None:
                return False
            progress['total_samples'] += uploaded_count_count

        # Mark dataset as completed
        if dataset_key not in progress['completed_datasets']:
            progress['completed_datasets'].append(dataset_key)
        progress['current_dataset'] = None
        progress['current_index'] = 0
        self.save_global_progress(progress)

        print(f"✅ {dataset_key} completed!")
        return True

    def load_index_registry(self):
        if self.index_registry is not None:
            return self.index_registry

        try:
            registry_path = hf_hub_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                filename="metadata/index_registry.json",
                token=self.hf_token,
            )
            with open(registry_path, 'r') as f:
                raw_registry = json.load(f)

            self.index_registry = {}
            for source, splits in raw_registry.items():
                self.index_registry[source] = {}
                for split, max_index in splits.items():
                    try:
                        self.index_registry[source][split] = int(max_index)
                    except (TypeError, ValueError):
                        self.index_registry[source][split] = -1
        except Exception:
            self.index_registry = {}

        return self.index_registry

    def persist_index_registry(self, commit_message):
        if self.index_registry is None:
            return

        metadata_dir = Path("./global_metadata")
        metadata_dir.mkdir(exist_ok=True)
        local_path = metadata_dir / "index_registry.json"
        with open(local_path, 'w') as f:
            json.dump(self.index_registry, f, indent=2, sort_keys=True)

        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="metadata/index_registry.json",
            repo_id=self.dataset_name,
            repo_type="dataset",
            token=self.hf_token,
            commit_message=commit_message,
        )

    def filter_duplicates(self, batch_samples):
        registry = self.load_index_registry()
        filtered = []
        seen = set()
        skipped = 0
        updates = {}

        for sample in batch_samples:
            source = sample.get('source_dataset')
            split = sample.get('original_split', 'train') or 'train'
            index = sample.get('original_index')

            if source is None or index is None:
                filtered.append(sample)
                continue

            try:
                index = int(index)
            except (TypeError, ValueError):
                filtered.append(sample)
                continue

            key = (source, split, index)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)

            existing_max = registry.get(source, {}).get(split, -1)
            if index <= existing_max:
                skipped += 1
                continue

            filtered.append(sample)
            updates[(source, split)] = max(updates.get((source, split), existing_max), index)

        for (source, split), max_index in updates.items():
            registry.setdefault(source, {})
            registry[source][split] = max(registry[source].get(split, -1), max_index)

        if skipped:
            print(f"⚖️  {skipped} duplicate samples skipped (already uploaded)")

        return filtered

    def upload_batch_append(self, batch_samples):
        """Upload a batch as a new shard and update the shard index."""

        try:
            print(f"📤 Uploading batch of {len(batch_samples)} samples...")
            shard_id = self.shard_manager.next_shard_id()
            result = self.shard_manager.upload_samples_as_shard(
                batch_samples,
                shard_id=shard_id,
                commit_message=f"Add shard {shard_id} from global builder",
            )

            if result.index_updated:
                self.shard_manager.save_index(
                    commit_message=f"Update shard index with {shard_id}"
                )

            if result.uploaded_samples == 0:
                print(
                    f"ℹ️ Shard {shard_id} only contained duplicates ({result.skipped_duplicates} skipped)."
                )
            else:
                print(
                    f"✅ Upload successful: {result.uploaded_samples} new samples in {shard_id}"
                )

            return result.uploaded_samples

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None

    def run_complete_pipeline(self):
        """Run the complete multilingual pipeline"""
        print("🌍 GLOBAL INCREMENTAL MULTILINGUAL PIPELINE")
        print("=" * 60)
        print(f"📊 Dataset: {self.dataset_name}")
        print("🔄 Cross-computer resumable with global state tracking")
        print("🚫 Duplicate-proof with HuggingFace metadata")

        # Get initial progress
        progress = self.get_global_progress()
        print(f"📊 Starting from: {progress['total_samples']} existing samples")

        # Process each dataset
        for entry in self.sources:
            name, subset, text_field, summary_field, expected_count, *rest = entry
            max_samples = rest[0] if rest else None
            success = self.process_dataset_incremental(name, subset, text_field, summary_field, expected_count, max_samples)
            if not success:
                print(f"❌ Pipeline stopped at {name}")
                return False

        print(f"🎉 PIPELINE COMPLETED!")
        print(f"📊 Final dataset: https://huggingface.co/datasets/{self.dataset_name}")
        return True

def run_global_incremental_pipeline():
    builder = GlobalIncrementalPipeline()
    builder.run_complete_pipeline()

if __name__ == "__main__":
    run_global_incremental_pipeline()
