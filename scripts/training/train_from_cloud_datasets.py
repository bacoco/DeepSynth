#!/usr/bin/env python3
"""
Training from Cloud Datasets - Séparé de la Génération

Ce script charge des datasets PRÉ-GÉNÉRÉS depuis HuggingFace
et lance le fine-tuning. Aucune génération d'image n'est faite ici.

Architecture:
1. Dataset generation (autre script): CPU → HuggingFace
2. Ce script: HuggingFace → GPU → Fine-tuning

Avantages:
- Pas besoin de régénérer les images
- Entraînement plus rapide (pas de preprocessing)
- Peut combiner plusieurs datasets facilement
- GPU utilisé uniquement pour le training
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datasets import load_dataset, concatenate_datasets, DatasetDict
from deepsynth.config import Config
from deepsynth.training.optimized_trainer import (
    OptimizedTrainerConfig,
    OptimizedDeepSynthTrainer,
    create_trainer,
)
from deepsynth.utils import get_logger, setup_global_logging

logger = get_logger(__name__)


class CloudDatasetTrainer:
    """Entraîne un modèle à partir de datasets cloud."""

    # Mapping des datasets pré-générés
    AVAILABLE_DATASETS = {
        "mlsum_fr": "deepsynth-mlsum-fr-images",
        "mlsum_es": "deepsynth-mlsum-es-images",
        "mlsum_de": "deepsynth-mlsum-de-images",
        "cnn_dailymail": "deepsynth-cnn-dailymail-images",
        "xsum": "deepsynth-xsum-images",
        "billsum": "deepsynth-billsum-images",
    }

    def __init__(self, hf_username: str, hf_token: str):
        """Initialize cloud dataset trainer."""
        self.hf_username = hf_username
        self.hf_token = hf_token

    def load_datasets(
        self,
        dataset_keys: List[str],
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ) -> DatasetDict:
        """
        Charge et combine plusieurs datasets depuis HuggingFace.

        Args:
            dataset_keys: Liste de clés de datasets (ex: ["mlsum_fr", "cnn_dailymail"])
            max_train_samples: Limite pour le split train
            max_val_samples: Limite pour le split validation

        Returns:
            DatasetDict avec splits train, validation, test
        """
        logger.info(f"📥 Chargement de {len(dataset_keys)} dataset(s) depuis HuggingFace")

        all_train = []
        all_val = []
        all_test = []

        for key in dataset_keys:
            if key not in self.AVAILABLE_DATASETS:
                logger.warning(f"⚠️  Dataset inconnu: {key}, ignoré")
                continue

            dataset_name = self.AVAILABLE_DATASETS[key]
            repo_id = f"{self.hf_username}/{dataset_name}"

            logger.info(f"   Loading {repo_id}...")

            try:
                # Charger le dataset
                dataset = load_dataset(
                    repo_id,
                    token=self.hf_token,
                )

                # Collecter les splits
                if "train" in dataset:
                    all_train.append(dataset["train"])
                if "validation" in dataset:
                    all_val.append(dataset["validation"])
                if "test" in dataset:
                    all_test.append(dataset["test"])

                logger.info(f"   ✅ Loaded: {len(dataset)} splits")

            except Exception as e:
                logger.error(f"   ❌ Erreur chargement {repo_id}: {e}")
                continue

        # Combiner les datasets
        logger.info("\n🔗 Combinaison des datasets...")

        combined = {}

        if all_train:
            combined["train"] = concatenate_datasets(all_train)
            if max_train_samples and len(combined["train"]) > max_train_samples:
                combined["train"] = combined["train"].select(range(max_train_samples))
            logger.info(f"   Train: {len(combined['train'])} samples")

        if all_val:
            combined["validation"] = concatenate_datasets(all_val)
            if max_val_samples and len(combined["validation"]) > max_val_samples:
                combined["validation"] = combined["validation"].select(range(max_val_samples))
            logger.info(f"   Validation: {len(combined['validation'])} samples")

        if all_test:
            combined["test"] = concatenate_datasets(all_test)
            logger.info(f"   Test: {len(combined['test'])} samples")

        return DatasetDict(combined)

    def train(
        self,
        datasets: DatasetDict,
        trainer_config: OptimizedTrainerConfig,
    ):
        """
        Lance l'entraînement avec les datasets chargés.

        Args:
            datasets: DatasetDict avec les données
            trainer_config: Configuration du trainer
        """
        logger.info("\n🚀 Démarrage de l'entraînement")
        logger.info(f"   Train samples: {len(datasets['train'])}")
        logger.info(f"   Val samples: {len(datasets.get('validation', []))}")
        logger.info(f"   Batch size: {trainer_config.batch_size}")
        logger.info(f"   Epochs: {trainer_config.num_epochs}")
        logger.info(f"   Mixed precision: {trainer_config.mixed_precision}")

        # Créer le trainer
        trainer = create_trainer(config=trainer_config)

        # Entraîner
        stats = trainer.train(
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
        )

        logger.info("\n✅ Entraînement terminé!")
        logger.info(f"   Best loss: {trainer.best_loss:.4f}")
        logger.info(f"   Global step: {trainer.global_step}")

        return stats


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Train DeepSynth from pre-generated cloud datasets"
    )

    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(CloudDatasetTrainer.AVAILABLE_DATASETS.keys()) + ["all"],
        default=["mlsum_fr"],
        help="Datasets à utiliser pour l'entraînement",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Limite nombre d'échantillons d'entraînement",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        help="Limite nombre d'échantillons de validation",
    )

    # Training config
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Nombre d'epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["bf16", "fp16", "no"],
        default="bf16",
        help="Mixed precision training",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/cloud-training",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-OCR",
        help="Base model to fine-tune",
    )

    # Advanced
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    # Setup
    setup_global_logging()

    # Load config
    config = Config.from_env()

    # Expand "all" datasets
    if "all" in args.datasets:
        dataset_keys = list(CloudDatasetTrainer.AVAILABLE_DATASETS.keys())
    else:
        dataset_keys = args.datasets

    logger.info("=" * 60)
    logger.info("DEEPSYNTH TRAINING FROM CLOUD DATASETS")
    logger.info("=" * 60)
    logger.info(f"\nDatasets: {', '.join(dataset_keys)}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {args.output_dir}\n")

    # Créer le trainer
    cloud_trainer = CloudDatasetTrainer(
        hf_username=config.hf_username,
        hf_token=config.hf_token,
    )

    # Charger les datasets
    datasets = cloud_trainer.load_datasets(
        dataset_keys,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    if not datasets:
        logger.error("❌ Aucun dataset chargé. Vérifier que les datasets existent sur HF.")
        return 1

    # Configuration du trainer
    trainer_config = OptimizedTrainerConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_gradient_scaling=args.mixed_precision == "fp16",
    )

    # Entraîner
    stats = cloud_trainer.train(datasets, trainer_config)

    logger.info(f"\n🎉 Training completed!")
    logger.info(f"Checkpoints saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())