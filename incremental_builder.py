#!/usr/bin/env python3
"""
Pipeline incrémental avec sauvegarde : traite tous les datasets avec reprise possible.
"""

import os
import json
import pickle
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login, whoami, HfApi
from data.text_to_image import TextToImageConverter
from mlsum_loader import MLSUMLoader

# Load environment variables
env_file = Path('.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

class OptimizedConverter(TextToImageConverter):
    def __init__(self):
        super().__init__(
            font_size=16, max_width=800, max_height=600, margin=30,
            background_color=(255, 255, 255), text_color=(0, 0, 0)
        )

class IncrementalBuilder:
    def __init__(self, work_dir="./work"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.progress_file = self.work_dir / "progress.json"
        self.samples_dir = self.work_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        self.converter = OptimizedConverter()
        self.progress = self.load_progress()

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed': [], 'current': None, 'split': None, 'index': 0, 'total': 0}

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def save_batch(self, samples, batch_id):
        batch_file = self.samples_dir / f"batch_{batch_id:06d}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(samples, f)
        print(f"💾 Batch {batch_id}: {len(samples)} échantillons")

    def load_all_samples(self):
        all_samples = []
        for batch_file in sorted(self.samples_dir.glob("batch_*.pkl")):
            with open(batch_file, 'rb') as f:
                all_samples.extend(pickle.load(f))
        return all_samples

    def _try_alternative_dataset_names(self, name, subset, split):
        """Try alternative loading methods for problematic datasets"""
        # For scientific_papers, we'll skip it since it requires custom script
        # Most other datasets should work with direct loading now
        return None

    def _extract_text_and_summary(self, example, name, text_field, summary_field):
        """Extract text and summary from example based on dataset type"""
        try:
            if name == 'cnn_dailymail':
                return example.get('article', ''), example.get('highlights', '')
            elif name == 'billsum':
                return example.get('text', ''), example.get('summary', '')
            else:
                # Standard processing using provided field names (works for MLSUM)
                return example.get(text_field, ''), example.get(summary_field, '')
        except Exception as e:
            print(f"      ⚠ Error extracting fields: {e}")
            return '', ''

    def process_dataset(self, name, subset, text_field, summary_field, batch_size=50):
        if name in self.progress['completed']:
            print(f"✅ {name} déjà complété")
            return

        print(f"\\n📥 Traitement: {name}")

        # Déterminer les splits (using correct dataset names)
        splits_map = {
            'MLSUM': ['train', 'validation', 'test'],
            'billsum': ['train', 'test'],
            'cnn_dailymail': ['train', 'validation', 'test'],
        }
        splits = splits_map.get(name, ['train'])

        batch_counter = len(list(self.samples_dir.glob("batch_*.pkl")))

        for split in splits:
            # Vérifier reprise
            if (self.progress['current'] == name and self.progress['split'] == split):
                start_idx = self.progress['index']
            elif (self.progress['current'] == name and splits.index(split) < splits.index(self.progress['split'])):
                continue
            else:
                start_idx = 0

            print(f"  📂 {split} (depuis {start_idx})")

            try:
                # Special handling for MLSUM dataset
                if name == 'MLSUM':
                    print(f"    🎯 Loading MLSUM {subset} using custom loader")
                    mlsum_loader = MLSUMLoader()
                    dataset_dict = mlsum_loader.load_language(subset)

                    if split in dataset_dict:
                        dataset = dataset_dict[split]
                        print(f"    ✅ MLSUM {subset} {split} loaded successfully")
                    else:
                        print(f"    ❌ Split {split} not found for MLSUM {subset}")
                        continue
                else:
                    # Try multiple loading strategies with robust error handling
                    dataset = None
                    loading_errors = []

                    # Strategy 1: Direct loading
                    try:
                        if subset:
                            dataset = load_dataset(name, subset, split=split)
                        else:
                            dataset = load_dataset(name, split=split)
                        print(f"    ✅ Direct loading successful")
                    except Exception as e:
                        loading_errors.append(f"Direct: {str(e)[:100]}")

                    # Strategy 2: With trust_remote_code for custom scripts
                    if dataset is None:
                        try:
                            if subset:
                                dataset = load_dataset(name, subset, split=split, trust_remote_code=True)
                            else:
                                dataset = load_dataset(name, split=split, trust_remote_code=True)
                            print(f"    ✅ Trust remote code successful")
                        except Exception as e:
                            loading_errors.append(f"Trust remote: {str(e)[:100]}")

                    # Strategy 3: Try alternative dataset names
                    if dataset is None:
                        try:
                            alt_dataset = self._try_alternative_dataset_names(name, subset, split)
                            if alt_dataset is not None:
                                dataset = alt_dataset
                                print(f"    ✅ Alternative name successful")
                        except Exception as e:
                            loading_errors.append(f"Alternative: {str(e)[:100]}")

                    if dataset is None:
                        print(f"    ❌ All loading strategies failed for {name}:")
                        for error in loading_errors:
                            print(f"      - {error}")
                        continue  # Skip this dataset and continue with others

                total = len(dataset)
                print(f"    📊 {total} échantillons")

                batch = []
                for idx in range(start_idx, total):
                    self.progress.update({'current': name, 'split': split, 'index': idx})

                    example = dataset[idx]

                    # Extract text and summary using robust helper method
                    text, summary = self._extract_text_and_summary(example, name, text_field, summary_field)

                    if not text or not summary:
                        continue

                    try:
                        image = self.converter.convert(text)
                        batch.append({
                            'text': text, 'summary': summary, 'image': image,
                            'source_dataset': name, 'original_split': split, 'original_index': idx
                        })
                        self.progress['total'] += 1
                    except Exception as e:
                        print(f"      ❌ Échantillon {idx}: {e}")
                        continue

                    # Sauvegarde par batch
                    if len(batch) >= batch_size:
                        self.save_batch(batch, batch_counter)
                        batch_counter += 1
                        batch = []
                        self.save_progress()

                    if (idx + 1) % 500 == 0:
                        print(f"      📈 {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")

                # Dernier batch
                if batch:
                    self.save_batch(batch, batch_counter)
                    batch_counter += 1

                print(f"    ✅ {split} complété")

            except Exception as e:
                print(f"    ❌ Erreur {split}: {e}")
                self.save_progress()
                raise

        # Marquer comme complété
        self.progress['completed'].append(name)
        self.progress.update({'current': None, 'split': None, 'index': 0})
        self.save_progress()
        print(f"✅ {name} terminé")

    def create_final_dataset(self, repo_name):
        print("\\n🔧 Création du dataset final...")
        samples = self.load_all_samples()

        if not samples:
            raise RuntimeError("Aucun échantillon")

        print(f"📊 {len(samples)} échantillons total")

        # Créer dataset
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
        dataset_dict = DatasetDict({'train': dataset})

        # Stats
        stats = {}
        for s in samples:
            key = f"{s['source_dataset']}_{s['original_split']}"
            stats[key] = stats.get(key, 0) + 1

        print("\\n📊 Distribution:")
        for key, count in stats.items():
            print(f"  - {key}: {count:,}")

        # Upload
        try:
            HfApi().delete_repo(repo_id=repo_name, repo_type='dataset')
        except:
            pass

        print(f"\\n📤 Upload: {repo_name}")
        dataset_dict.push_to_hub(repo_name, private=False, token=os.getenv('HF_TOKEN'))
        return repo_name

def main():
    print("🎯 TRAITEMENT INCRÉMENTAL COMPLET")
    print("=" * 50)

    login(token=os.getenv('HF_TOKEN'))
    username = whoami()['name']
    repo_name = f"{username}/deepseek-vision-complete"

    print(f"Dataset: {repo_name}")
    print("🔄 Reprise automatique si interruption")

    builder = IncrementalBuilder()

    sources = [
        # PRIORITY: MLSUM Multilingual Summarization (as requested - using downloaded data)
        ('MLSUM', 'fr', 'text', 'summary'),                          # French news summarization - 392k examples
        ('MLSUM', 'es', 'text', 'summary'),                          # Spanish news summarization - 266k examples
        ('MLSUM', 'de', 'text', 'summary'),                          # German news summarization - 220k examples

        # High-Quality Text Summarization Datasets
        ('billsum', None, 'text', 'summary'),                        # Legal documents - US bills (~22k examples)
        ('cnn_dailymail', '3.0.0', 'article', 'highlights'),         # News articles - 287k examples
    ]

    try:
        for name, subset, text_field, summary_field in sources:
            builder.process_dataset(name, subset, text_field, summary_field)

        final_repo = builder.create_final_dataset(repo_name)
        print(f"\\n🎉 SUCCÈS: https://huggingface.co/datasets/{final_repo}")

        # Cleanup
        import shutil
        shutil.rmtree(builder.work_dir)
        print("🧹 Nettoyage terminé")

    except KeyboardInterrupt:
        print("\\n⏸️ Interruption - relancez pour continuer")
        builder.save_progress()
    except Exception as e:
        print(f"\\n❌ Erreur: {e}")
        builder.save_progress()
        raise

if __name__ == "__main__":
    main()
