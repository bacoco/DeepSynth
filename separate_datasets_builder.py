#!/usr/bin/env python3
"""
Pipeline pour créer des datasets séparés par langue/type sur HuggingFace.
Chaque dataset aura un nom de la forme: deepsynth-fr, deepsynth-es, etc.
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

# Mapping des datasets vers leurs noms de sortie
DATASET_NAMING = {
    ('MLSUM', 'fr'): 'deepsynth-fr',                    # Français - 392k exemples
    ('MLSUM', 'es'): 'deepsynth-es',                    # Espagnol - 266k exemples
    ('MLSUM', 'de'): 'deepsynth-de',                    # Allemand - 220k exemples
    ('cnn_dailymail', '3.0.0'): 'deepsynth-en-news',   # Anglais actualités - 287k exemples
    ('ccdv/arxiv-summarization', None): 'deepsynth-en-arxiv',  # Anglais scientifique - 50k exemples
    ('Rexhaif/xsum_reduced', None): 'deepsynth-en-xsum',       # Anglais BBC - 50k exemples
    ('billsum', None): 'deepsynth-en-legal',                   # Anglais juridique - 22k exemples
}

class OptimizedConverter(TextToImageConverter):
    def __init__(self):
        # Use DejaVu Sans font for proper French character support
        unicode_font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        super().__init__(
            font_path=unicode_font_path,
            font_size=12, max_width=1600, max_height=2400, margin=40,
            background_color=(255, 255, 255), text_color=(0, 0, 0)
        )

class SeparateDatasetBuilder:
    def __init__(self, work_dir="./work_separate"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.progress_file = self.work_dir / "progress_separate.json"
        self.converter = OptimizedConverter()
        self.progress = self.load_progress()

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed_datasets': []}

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
                # Standard processing using provided field names (works for MLSUM)
                return example.get(text_field, ''), example.get(summary_field, '')
        except Exception as e:
            print(f"      ⚠ Error extracting fields: {e}")
            return '', ''

    def get_dataset_output_name(self, name, subset):
        """Get the output dataset name for HuggingFace"""
        key = (name, subset)
        if key in DATASET_NAMING:
            return DATASET_NAMING[key]
        else:
            # Fallback naming
            if subset:
                return f"deepsynth-{subset}"
            else:
                return f"deepsynth-{name.replace('/', '-')}"

    def check_existing_dataset_progress(self, repo_name):
        """Check existing dataset on HuggingFace and return progress info"""
        try:
            print(f"    🔍 Vérification du dataset existant: {repo_name}")

            # Try to load existing dataset
            existing_dataset = load_dataset(repo_name, split='train')

            if 'original_index' in existing_dataset.column_names:
                existing_indices = set(existing_dataset['original_index'])
                last_index = max(existing_indices) if existing_indices else -1
                total_processed = len(existing_indices)

                print(f"    📊 Dataset existant trouvé:")
                print(f"      - {total_processed} échantillons déjà traités")
                print(f"      - Dernier index traité: {last_index}")

                return {
                    'exists': True,
                    'processed_indices': existing_indices,
                    'last_index': last_index,
                    'total_processed': total_processed,
                    'existing_dataset': existing_dataset
                }
            else:
                print(f"    ⚠️ Dataset existant sans colonne 'original_index', recommence depuis le début")
                return {'exists': False}

        except Exception as e:
            print(f"    ℹ️ Aucun dataset existant trouvé (normal pour nouveau dataset)")
            return {'exists': False}

    def process_and_upload_dataset(self, name, subset, text_field, summary_field, username, *, max_samples=None):
        """Process a single dataset and upload it immediately to HuggingFace"""

        # Check if already completed
        dataset_key = f"{name}_{subset}" if subset else name
        if dataset_key in self.progress['completed_datasets']:
            print(f"✅ {dataset_key} déjà complété et uploadé")
            return

        output_name = self.get_dataset_output_name(name, subset)
        repo_name = f"{username}/{output_name}"

        print(f"\\n📥 Traitement: {name} ({subset}) → {repo_name}")

        # Check existing progress on HuggingFace
        progress_info = self.check_existing_dataset_progress(repo_name)

        # Déterminer les splits
        splits_map = {
            'MLSUM': ['train', 'validation', 'test'],
            'billsum': ['train', 'test'],
            'cnn_dailymail': ['train', 'validation', 'test'],
            'ccdv/arxiv-summarization': ['train'],
            'Rexhaif/xsum_reduced': ['train'],
        }
        splits = splits_map.get(name, ['train'])

        # Start with existing samples if any
        if progress_info['exists']:
            all_samples = []
            existing_dataset = progress_info['existing_dataset']

            # Convert existing dataset back to list of samples
            for i in range(len(existing_dataset)):
                all_samples.append({
                    'text': existing_dataset[i]['text'],
                    'summary': existing_dataset[i]['summary'],
                    'image': existing_dataset[i]['image'],
                    'source_dataset': existing_dataset[i]['source_dataset'],
                    'original_split': existing_dataset[i]['original_split'],
                    'original_index': existing_dataset[i]['original_index'],
                })

            processed_indices = progress_info['processed_indices']
            last_index = progress_info['last_index']
            print(f"    🔄 Reprise à partir de l'index {last_index + 1}")
        else:
            all_samples = []
            processed_indices = set()
            last_index = -1

        for split in splits:
            print(f"  📂 Processing split: {split}")

            try:
                # Load dataset
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
                    # Try loading with different strategies
                    dataset = None
                    try:
                        if subset:
                            dataset = load_dataset(name, subset, split=split)
                        else:
                            dataset = load_dataset(name, split=split)
                        print(f"    ✅ Direct loading successful")
                    except Exception as e:
                        try:
                            if subset:
                                dataset = load_dataset(name, subset, split=split, trust_remote_code=True)
                            else:
                                dataset = load_dataset(name, split=split, trust_remote_code=True)
                            print(f"    ✅ Trust remote code successful")
                        except Exception as e2:
                            print(f"    ❌ Failed to load {name}: {e2}")
                            continue

                total = len(dataset)
                limit = min(total, max_samples) if max_samples is not None else total

                # Calculate remaining work
                remaining_indices = []
                for idx in range(limit):
                    # Create a unique key for this sample across all splits
                    sample_key = f"{split}_{idx}"
                    if idx not in processed_indices:
                        remaining_indices.append(idx)

                if not remaining_indices:
                    print(f"    ✅ {split} déjà complètement traité ({limit} échantillons)")
                    continue

                print(f"    📊 {len(remaining_indices)} échantillons restants à traiter sur {limit}")
                print(f"    📈 Progression: {((limit - len(remaining_indices)) / limit * 100):.1f}% déjà fait")

                # Process only remaining samples
                split_samples = []
                processed_count = 0

                for idx in remaining_indices:
                    processed_count += 1
                    if processed_count % 1000 == 0:
                        print(f"      📈 {processed_count}/{len(remaining_indices)} ({processed_count/len(remaining_indices)*100:.1f}%)")

                    example = dataset[idx]
                    text, summary = self._extract_text_and_summary(example, name, text_field, summary_field)

                    if not text or not summary:
                        continue

                    try:
                        image = self.converter.convert(text)
                        split_samples.append({
                            'text': text,
                            'summary': summary,
                            'image': image,
                            'source_dataset': name,
                            'original_split': split,
                            'original_index': idx
                        })
                    except Exception as e:
                        print(f"      ❌ Sample {idx}: {e}")
                        continue

                all_samples.extend(split_samples)
                print(f"    ✅ {split} completed: {len(split_samples)} nouveaux échantillons traités")

            except Exception as e:
                print(f"    ❌ Error processing {split}: {e}")
                continue

        if not all_samples:
            print(f"    ❌ No samples processed for {name}")
            return

        print(f"\\n🔧 Creating dataset with {len(all_samples)} samples...")

        # Create dataset
        dataset = Dataset.from_dict({
            'text': [s['text'] for s in all_samples],
            'summary': [s['summary'] for s in all_samples],
            'image': [s['image'] for s in all_samples],
            'source_dataset': [s['source_dataset'] for s in all_samples],
            'original_split': [s['original_split'] for s in all_samples],
            'original_index': [s['original_index'] for s in all_samples],
        })

        from datasets import Image as ImageFeature
        dataset = dataset.cast_column("image", ImageFeature())

        # Create train/validation splits if we have enough data
        if len(all_samples) > 1000:
            # Use 95% for train, 5% for validation
            train_size = int(0.95 * len(all_samples))
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, len(all_samples)))
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
        else:
            # Small dataset, keep as single train split
            dataset_dict = DatasetDict({'train': dataset})

        # Upload to HuggingFace
        try:
            if progress_info['exists']:
                print(f"\\n📤 Mise à jour du dataset existant: {repo_name}")
                print(f"    📊 Total final: {len(all_samples)} échantillons")
            else:
                print(f"\\n📤 Création du nouveau dataset: {repo_name}")
                print(f"    📊 Total: {len(all_samples)} échantillons")

            dataset_dict.push_to_hub(repo_name, private=False, token=os.getenv('HF_TOKEN'))

            print(f"✅ Successfully uploaded: https://huggingface.co/datasets/{repo_name}")

            # Mark as completed
            self.progress['completed_datasets'].append(dataset_key)
            self.save_progress()

        except Exception as e:
            print(f"❌ Failed to upload {repo_name}: {e}")
            raise

def main():
    print("🎯 CRÉATION DE DATASETS SÉPARÉS PAR LANGUE")
    print("=" * 60)

    login(token=os.getenv('HF_TOKEN'))
    username = whoami()['name']

    print(f"Username: {username}")
    print("\\n📊 Datasets à créer:")
    for (name, subset), output_name in DATASET_NAMING.items():
        print(f"  - {name} ({subset}) → {username}/{output_name}")

    builder = SeparateDatasetBuilder()

    # Get arXiv limit
    try:
        arxiv_limit = int(os.getenv('ARXIV_IMAGE_SAMPLES', '50000'))
    except ValueError:
        print("⚠️  ARXIV_IMAGE_SAMPLES invalide. Utilisation de 50000 par défaut.")
        arxiv_limit = 50000
    if arxiv_limit <= 0:
        print("⚠️  ARXIV_IMAGE_SAMPLES doit être positif. Utilisation de 10000 par défaut.")
        arxiv_limit = 10000

    # Define sources with their configurations - ORDRE DE PRIORITÉ
    sources = [
        # PRIORITÉ 1: CNN/DailyMail (Anglais actualités) - Le plus demandé
        ('cnn_dailymail', '3.0.0', 'article', 'highlights', None),

        # PRIORITÉ 2: arXiv (Anglais scientifique) - Recherche
        ('ccdv/arxiv-summarization', None, 'article', 'abstract', arxiv_limit),

        # PRIORITÉ 3: XSum BBC (Anglais BBC News) - Actualités courtes
        ('Rexhaif/xsum_reduced', None, 'text', 'target', None),

        # PRIORITÉ 4: Français MLSUM - Multilingue commence ici
        ('MLSUM', 'fr', 'text', 'summary', None),

        # PRIORITÉ 5: Autres langues MLSUM
        ('MLSUM', 'es', 'text', 'summary', None),
        ('MLSUM', 'de', 'text', 'summary', None),

        # PRIORITÉ 6: Juridique (plus spécialisé)
        ('billsum', None, 'text', 'summary', None),
    ]

    try:
        for name, subset, text_field, summary_field, max_samples in sources:
            builder.process_and_upload_dataset(
                name, subset, text_field, summary_field, username,
                max_samples=max_samples
            )

        print(f"\\n🎉 TOUS LES DATASETS CRÉÉS AVEC SUCCÈS!")
        print("\\n📊 Datasets disponibles:")
        for (name, subset), output_name in DATASET_NAMING.items():
            print(f"  🔗 https://huggingface.co/datasets/{username}/{output_name}")

        # Cleanup
        import shutil
        if builder.work_dir.exists():
            shutil.rmtree(builder.work_dir)
            print("\\n🧹 Nettoyage terminé")

    except KeyboardInterrupt:
        print("\\n⏸️ Interruption - relancez pour continuer")
        builder.save_progress()
    except Exception as e:
        print(f"\\n❌ Erreur: {e}")
        builder.save_progress()
        raise

if __name__ == "__main__":
    main()
