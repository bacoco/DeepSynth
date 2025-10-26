#!/usr/bin/env python3
"""
Générateur de Dataset Cloud - Séparé du Fine-Tuning

Ce script génère UNIQUEMENT les images et uploade vers HuggingFace.
Le fine-tuning est fait séparément avec les datasets cloud.

Architecture:
1. Ce script: CPU-only, génération d'images → Upload HF
2. Training script: GPU, charge datasets HF → Fine-tune

Avantages:
- Génération parallélisable sur plusieurs machines
- Datasets réutilisables
- Pas besoin de GPU pour preprocessing
- Partage facile entre équipes
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

from deepsynth.config import Config
from deepsynth.data.transforms import TextToImageConverter
from deepsynth.utils import extract_text_summary, get_logger, setup_global_logging

logger = get_logger(__name__)


class CloudDatasetGenerator:
    """Génère des datasets avec images et les uploade sur HuggingFace."""

    # Configurations des datasets supportés
    DATASET_CONFIGS = {
        "mlsum_fr": {
            "source": "mlsum",
            "subset": "fr",
            "splits": ["train", "validation", "test"],
            "target_name": "deepsynth-mlsum-fr-images",
            "description": "MLSUM French with pre-rendered images for DeepSynth",
        },
        "mlsum_es": {
            "source": "mlsum",
            "subset": "es",
            "splits": ["train", "validation", "test"],
            "target_name": "deepsynth-mlsum-es-images",
            "description": "MLSUM Spanish with pre-rendered images for DeepSynth",
        },
        "mlsum_de": {
            "source": "mlsum",
            "subset": "de",
            "splits": ["train", "validation", "test"],
            "target_name": "deepsynth-mlsum-de-images",
            "description": "MLSUM German with pre-rendered images for DeepSynth",
        },
        "cnn_dailymail": {
            "source": "ccdv/cnn_dailymail",
            "subset": "3.0.0",
            "splits": ["train", "validation", "test"],
            "target_name": "deepsynth-cnn-dailymail-images",
            "description": "CNN/DailyMail with pre-rendered images for DeepSynth",
        },
        "xsum": {
            "source": "EdinburghNLP/xsum",
            "subset": None,
            "splits": ["train", "validation", "test"],
            "target_name": "deepsynth-xsum-images",
            "description": "XSum with pre-rendered images for DeepSynth",
        },
        "billsum": {
            "source": "billsum",
            "subset": None,
            "splits": ["train", "test"],
            "target_name": "deepsynth-billsum-images",
            "description": "BillSum with pre-rendered images for DeepSynth",
        },
    }

    def __init__(
        self,
        hf_token: str,
        hf_username: str,
        converter: Optional[TextToImageConverter] = None,
    ):
        """Initialize cloud dataset generator."""
        self.hf_token = hf_token
        self.hf_username = hf_username
        self.converter = converter or TextToImageConverter()
        self.api = HfApi(token=hf_token)

    def generate_and_upload(
        self,
        dataset_key: str,
        max_samples_per_split: Optional[int] = None,
        private: bool = False,
    ) -> str:
        """
        Génère un dataset complet et l'uploade sur HuggingFace.

        Args:
            dataset_key: Clé du dataset (ex: "mlsum_fr")
            max_samples_per_split: Limite par split (None = tous)
            private: Dataset privé ou public

        Returns:
            Nom complet du dataset uploadé (username/dataset-name)
        """
        if dataset_key not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Dataset inconnu: {dataset_key}. "
                f"Disponibles: {list(self.DATASET_CONFIGS.keys())}"
            )

        config = self.DATASET_CONFIGS[dataset_key]
        logger.info(f"🚀 Génération du dataset: {dataset_key}")
        logger.info(f"   Source: {config['source']}")
        logger.info(f"   Target: {config['target_name']}")

        # Créer le repo HuggingFace
        repo_id = f"{self.hf_username}/{config['target_name']}"
        self._create_repo(repo_id, config["description"], private)

        # Traiter chaque split
        dataset_dict = {}
        for split in config["splits"]:
            logger.info(f"\n📊 Processing split: {split}")
            processed_split = self._process_split(
                config["source"],
                config["subset"],
                split,
                dataset_key,
                max_samples_per_split,
            )
            dataset_dict[split] = processed_split

        # Upload vers HuggingFace
        logger.info(f"\n⬆️  Uploading to HuggingFace: {repo_id}")
        dataset_dict_obj = DatasetDict(dataset_dict)
        dataset_dict_obj.push_to_hub(
            repo_id,
            token=self.hf_token,
            private=private,
        )

        logger.info(f"\n✅ Dataset uploadé: https://huggingface.co/datasets/{repo_id}")
        return repo_id

    def _create_repo(self, repo_id: str, description: str, private: bool):
        """Crée le repo HuggingFace s'il n'existe pas."""
        try:
            create_repo(
                repo_id,
                token=self.hf_token,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            logger.info(f"✅ Repo créé/vérifié: {repo_id}")

            # Ajouter README
            readme = f"""# {repo_id}

{description}

## Dataset Structure

This dataset contains pre-rendered images for DeepSynth training.

### Fields

- `text`: Original document text
- `summary`: Human-written summary
- `image`: Pre-rendered PNG image of the text
- `source_dataset`: Original dataset name
- `original_split`: Original split (train/validation/test)
- `original_index`: Index in original dataset

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access samples
sample = dataset['train'][0]
print(sample['text'])
print(sample['summary'])
sample['image'].show()  # Display image
```

## Generation

Generated using DeepSynth preprocessing pipeline.
"""
            self.api.upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.hf_token,
            )

        except Exception as e:
            logger.warning(f"⚠️  Repo creation issue: {e}")

    def _process_split(
        self,
        source: str,
        subset: Optional[str],
        split: str,
        dataset_key: str,
        max_samples: Optional[int],
    ) -> Dataset:
        """Traite un split du dataset."""
        # Charger le dataset source
        logger.info(f"   Loading {source} ({subset or 'default'}) - {split}")

        if subset:
            dataset = load_dataset(source, subset, split=split)
        else:
            dataset = load_dataset(source, split=split)

        # Limiter si nécessaire
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"   Limited to {max_samples} samples")

        logger.info(f"   Processing {len(dataset)} samples...")

        # Traiter chaque échantillon
        processed_samples = []
        errors = 0

        for idx in tqdm(range(len(dataset)), desc=f"   {split}"):
            try:
                sample = dataset[idx]

                # Extraire text et summary
                text, summary = extract_text_summary(sample, dataset_name=dataset_key)

                if not text or not summary:
                    continue

                # Convertir en image
                image = self.converter.convert(text)

                processed_samples.append({
                    "text": text,
                    "summary": summary,
                    "image": image,
                    "source_dataset": dataset_key,
                    "original_split": split,
                    "original_index": idx,
                })

            except Exception as e:
                errors += 1
                if errors <= 5:  # Log only first 5 errors
                    logger.warning(f"   Error processing sample {idx}: {e}")

        logger.info(
            f"   ✅ Processed: {len(processed_samples)}/{len(dataset)} "
            f"(errors: {errors})"
        )

        # Créer le dataset HuggingFace
        return Dataset.from_list(processed_samples)

    def list_generated_datasets(self) -> List[str]:
        """Liste tous les datasets générés par cet utilisateur."""
        logger.info("📋 Datasets générés:")

        datasets = []
        for key, config in self.DATASET_CONFIGS.items():
            repo_id = f"{self.hf_username}/{config['target_name']}"
            try:
                # Vérifier si le dataset existe
                self.api.dataset_info(repo_id)
                datasets.append(repo_id)
                logger.info(f"   ✅ {repo_id}")
            except:
                logger.info(f"   ❌ {repo_id} (pas encore généré)")

        return datasets


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Générer et uploader des datasets avec images vers HuggingFace"
    )
    parser.add_argument(
        "dataset",
        choices=list(CloudDatasetGenerator.DATASET_CONFIGS.keys()) + ["all"],
        help="Dataset à générer",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Nombre max d'échantillons par split (pour tests)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Créer un dataset privé",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les datasets déjà générés",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token (ou utiliser HF_TOKEN env var)",
    )
    parser.add_argument(
        "--hf-username",
        help="HuggingFace username (ou utiliser HF_USERNAME env var)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_global_logging()

    # Charger config
    if args.hf_token and args.hf_username:
        hf_token = args.hf_token
        hf_username = args.hf_username
    else:
        config = Config.from_env()
        hf_token = config.hf_token
        hf_username = config.hf_username

    # Créer le générateur
    generator = CloudDatasetGenerator(hf_token, hf_username)

    # Lister ou générer
    if args.list:
        generator.list_generated_datasets()
        return 0

    # Générer dataset(s)
    start_time = time.time()

    if args.dataset == "all":
        logger.info("🚀 Génération de TOUS les datasets\n")
        for dataset_key in CloudDatasetGenerator.DATASET_CONFIGS.keys():
            try:
                generator.generate_and_upload(
                    dataset_key,
                    max_samples_per_split=args.max_samples,
                    private=args.private,
                )
                logger.info("")
            except Exception as e:
                logger.error(f"❌ Erreur avec {dataset_key}: {e}")
                continue
    else:
        generator.generate_and_upload(
            args.dataset,
            max_samples_per_split=args.max_samples,
            private=args.private,
        )

    elapsed = time.time() - start_time
    logger.info(f"\n⏱️  Temps total: {elapsed/60:.1f} minutes")

    return 0


if __name__ == "__main__":
    exit(main())