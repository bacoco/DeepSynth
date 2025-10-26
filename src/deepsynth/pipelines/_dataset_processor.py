#!/usr/bin/env python3
"""
Pipeline optimisé pour créer des datasets séparés par langue sur HuggingFace.
VERSION 2: Upload incrémental par batches de 5000, pas de splits, métadonnées seulement.
"""

import os
import json
import pickle
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, login, whoami

from deepsynth.config import load_shared_env
from deepsynth.data.loaders import MLSUMLoader
from deepsynth.data.transforms import TextToImageConverter
from deepsynth.pipelines.uploaders.incremental import EfficientIncrementalUploader

__all__ = ["OptimizedDatasetPipeline", "run_optimized_pipeline"]

load_shared_env()

# Mapping des datasets vers leurs noms de sortie
DATASET_NAMING = {
    ('MLSUM', 'fr'): 'deepsynth-fr',
    ('MLSUM', 'es'): 'deepsynth-es',
    ('MLSUM', 'de'): 'deepsynth-de',
    ('cnn_dailymail', '3.0.0'): 'deepsynth-en-news',
    ('ccdv/arxiv-summarization', None): 'deepsynth-en-arxiv',
    ('Rexhaif/xsum_reduced', None): 'deepsynth-en-xsum',
    ('billsum', None): 'deepsynth-en-legal',
}

class OptimizedConverter(TextToImageConverter):
    def __init__(self):
        # Auto-detect Unicode font for multilingual support
        unicode_font_path = self._find_unicode_font()
        super().__init__(
            font_path=unicode_font_path,
            font_size=12, max_width=1600, max_height=2400, margin=40,
            background_color=(255, 255, 255), text_color=(0, 0, 0)
        )

    @staticmethod
    def _find_unicode_font():
        """Find a Unicode-capable font on the system."""
        import os
        import platform

        font_paths = []
        system = platform.system()

        if system == 'Darwin':  # macOS
            font_paths = [
                '/Library/Fonts/DejaVuSans.ttf',
                '/System/Library/Fonts/Helvetica.ttc',
                '/System/Library/Fonts/SFNSText.ttf',
                '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            ]
        elif system == 'Linux':
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/liberation/LiberationSans-Regular.ttf',
            ]
        elif system == 'Windows':
            font_paths = [
                'C:\\Windows\\Fonts\\arial.ttf',
                'C:\\Windows\\Fonts\\calibri.ttf',
            ]

        for path in font_paths:
            if os.path.exists(path):
                print(f"    ✅ Using Unicode font: {os.path.basename(path)}")
                return path

        print("    ⚠️  No Unicode font found, using PIL default")
        return None


class OptimizedDatasetPipeline:
    """
    Pipeline optimisé avec:
    - Upload incrémental par batches de 5000
    - Téléchargement métadonnées seulement (pas d'images)
    - Un seul dataset (pas de splits train/test/validation)
    - Nettoyage automatique des fichiers locaux
    """

    def __init__(self, work_dir="./work_separate", batch_size=50):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)

        self.samples_dir = self.work_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        self.progress_file = self.work_dir / "progress.json"
        self.converter = OptimizedConverter()
        self.progress = self.load_progress()

        self.batch_size = batch_size  # Samples per local batch file
        self.current_batch = []
        self.batch_counter = 0

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'completed_datasets': [],
            'processed_samples': {}  # {dataset_key: {(split, idx): True}}
        }

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def _extract_text_and_summary(self, example, name, text_field, summary_field):
        """Extract text and summary from example based on dataset type"""
        try:
            if name == 'cnn_dailymail':
                return example.get('article', ''), example.get('highlights', '')
            elif name == 'billsum':
                return example.get('text', ''), example.get('summary', '')
            elif 'xsum' in name.lower():
                text = example.get('text', example.get('document', ''))
                summary = example.get('target', example.get('summary', ''))
                return text, summary
            else:
                return example.get(text_field, ''), example.get(summary_field, '')
        except Exception as e:
            print(f"      ⚠ Error extracting fields: {e}")
            return '', ''

    def get_dataset_output_name(self, name, subset):
        """Get the output dataset name for HuggingFace"""
        key = (name, subset)
        return DATASET_NAMING.get(key, f"deepsynth-{subset or name.replace('/', '-')}")

    def check_processed_indices(self, repo_name):
        """
        OPTIMIZED: Download ONLY metadata (indices), NOT images.
        Returns set of (split, index) tuples already processed.
        """
        try:
            print(f"    🔍 Checking processed samples: {repo_name}")

            # Download ONLY the metadata columns (no images!)
            existing_metadata = load_dataset(
                repo_name,
                split='train',
                columns=['original_index', 'original_split']
            )

            processed = set()
            for row in existing_metadata:
                split = row.get('original_split', 'train')
                idx = row['original_index']
                processed.add((split, idx))

            print(f"    📊 Found {len(processed)} already processed (metadata only)")
            return processed

        except Exception as e:
            print(f"    ℹ️  No existing dataset (will create new)")
            return set()

    def save_batch_to_disk(self):
        """Save current batch to disk as pickle file"""
        if not self.current_batch:
            return

        batch_file = self.samples_dir / f"batch_{self.batch_counter:06d}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(self.current_batch, f)

        print(f"      💾 Saved batch {self.batch_counter} ({len(self.current_batch)} samples)")
        self.batch_counter += 1
        self.current_batch = []

    def process_and_batch_dataset(self, name, subset, text_field, summary_field, username, *, max_samples=None):
        """
        Process dataset and save to batches (NO immediate upload).
        Upload will be done incrementally by EfficientIncrementalUploader.
        """
        dataset_key = f"{name}_{subset}" if subset else name

        # Check if already completed
        if dataset_key in self.progress['completed_datasets']:
            print(f"✅ {dataset_key} already completed")
            return

        output_name = self.get_dataset_output_name(name, subset)
        repo_name = f"{username}/{output_name}"

        print(f"\n📥 Processing: {name} ({subset}) → {repo_name}")

        # Check what's already processed (metadata only, fast!)
        processed_keys = self.check_processed_indices(repo_name)

        # Source splits (to load raw data)
        # NOTE: Final dataset has NO splits - everything in 'train'
        splits_map = {
            'MLSUM': ['train', 'validation', 'test'],
            'billsum': ['train', 'test'],
            'cnn_dailymail': ['train', 'validation', 'test'],
            'ccdv/arxiv-summarization': ['train'],
            'Rexhaif/xsum_reduced': ['train'],
        }
        splits = splits_map.get(name, ['train'])

        total_new = 0

        for split in splits:
            print(f"  📂 Processing source split: {split}")

            try:
                # Load source dataset
                if name == 'MLSUM':
                    mlsum_loader = MLSUMLoader()
                    dataset_dict = mlsum_loader.load_language(subset)
                    dataset = dataset_dict[split] if split in dataset_dict else None
                    if dataset is None:
                        print(f"    ❌ Split {split} not found")
                        continue
                else:
                    try:
                        if subset:
                            dataset = load_dataset(name, subset, split=split)
                        else:
                            dataset = load_dataset(name, split=split)
                    except:
                        if subset:
                            dataset = load_dataset(name, subset, split=split, trust_remote_code=True)
                        else:
                            dataset = load_dataset(name, split=split, trust_remote_code=True)

                total = len(dataset)
                limit = min(total, max_samples) if max_samples else total

                # Filter already processed
                remaining = [idx for idx in range(limit) if (split, idx) not in processed_keys]

                if not remaining:
                    print(f"    ✅ {split} already complete ({limit} samples)")
                    continue

                print(f"    📊 {len(remaining)}/{limit} samples to process ({len(processed_keys)} already done)")

                # Process samples
                for count, idx in enumerate(remaining, 1):
                    if count % 1000 == 0:
                        print(f"      📈 {count}/{len(remaining)} ({count/len(remaining)*100:.1f}%)")

                    example = dataset[idx]
                    text, summary = self._extract_text_and_summary(example, name, text_field, summary_field)

                    if not text or not summary:
                        continue

                    try:
                        image = self.converter.convert(text)

                        # Add to current batch
                        self.current_batch.append({
                            'text': text,
                            'summary': summary,
                            'image': image,
                            'source_dataset': name,
                            'original_split': split,  # Source split (for tracking only)
                            'original_index': idx
                        })

                        total_new += 1

                        # Save batch when full
                        if len(self.current_batch) >= self.batch_size:
                            self.save_batch_to_disk()

                    except Exception as e:
                        print(f"      ❌ Sample {idx}: {e}")
                        continue

                print(f"    ✅ {split} processed: {len(remaining)} new samples")

            except Exception as e:
                print(f"    ❌ Error processing {split}: {e}")
                continue

        # Save remaining samples
        if self.current_batch:
            self.save_batch_to_disk()

        if total_new > 0:
            print(f"\n  ✅ Total new samples batched: {total_new}")
            print(f"  📦 Batches saved: {self.batch_counter}")
        else:
            print(f"\n  ℹ️  No new samples to process")

        # Mark as completed
        self.progress['completed_datasets'].append(dataset_key)
        self.save_progress()


def run_optimized_pipeline():
    """Run the optimized pipeline with incremental uploads"""
    print("🎯 PIPELINE OPTIMISÉ - UPLOAD INCRÉMENTAL")
    print("=" * 70)

    login(token=os.getenv('HF_TOKEN'))
    username = whoami()['name']

    print(f"Username: {username}")
    print("\n📊 Configuration:")
    print("  • Upload incrémental par batches de ~5000 samples")
    print("  • Téléchargement métadonnées seulement (pas d'images)")
    print("  • Un seul dataset (pas de splits train/test/validation)")
    print("  • Nettoyage automatique après upload")

    pipeline = OptimizedDatasetPipeline()

    # Get arXiv limit
    try:
        arxiv_limit = int(os.getenv('ARXIV_IMAGE_SAMPLES', '50000'))
    except ValueError:
        arxiv_limit = 50000

    # Define sources (priority order)
    sources = [
        ('cnn_dailymail', '3.0.0', 'article', 'highlights', None),
        ('ccdv/arxiv-summarization', None, 'article', 'abstract', arxiv_limit),
        ('Rexhaif/xsum_reduced', None, 'text', 'target', None),
        ('MLSUM', 'fr', 'text', 'summary', None),
        ('MLSUM', 'es', 'text', 'summary', None),
        ('MLSUM', 'de', 'text', 'summary', None),
        ('billsum', None, 'text', 'summary', None),
    ]

    try:
        for name, subset, text_field, summary_field, max_samples in sources:
            pipeline.process_and_batch_dataset(
                name, subset, text_field, summary_field, username,
                max_samples=max_samples
            )

        # Upload batches incrementally
        print("\n" + "=" * 70)
        print("📤 UPLOAD INCRÉMENTAL DES BATCHES")
        print("=" * 70)

        uploader = EfficientIncrementalUploader(
            work_dir=str(pipeline.work_dir),
            batches_per_upload=100  # Upload every ~5000 samples
        )
        uploader.upload_all_pending()

        print("\n🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")

    except KeyboardInterrupt:
        print("\n⏸️  Interrupted - relaunch to continue")
        pipeline.save_progress()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        pipeline.save_progress()
        raise


if __name__ == "__main__":
    run_optimized_pipeline()
