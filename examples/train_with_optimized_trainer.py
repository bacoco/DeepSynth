#!/usr/bin/env python3
"""
Exemple d'utilisation du OptimizedTrainer.

Démontre comment utiliser le nouveau trainer avec toutes les optimisations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from deepsynth.training.optimized_trainer import (
    OptimizedDeepSynthTrainer,
    OptimizedTrainerConfig,
    create_trainer,
)
from deepsynth.utils.logging_config import setup_global_logging


def example_1_quick_start():
    """Exemple 1: Démarrage rapide avec configuration par défaut."""
    print("\n" + "=" * 60)
    print("EXEMPLE 1: Démarrage Rapide")
    print("=" * 60 + "\n")

    # Setup logging
    setup_global_logging()

    # Créer le trainer avec les paramètres par défaut depuis .env
    trainer = create_trainer()

    # Charger un petit dataset pour l'exemple
    print("Chargement du dataset...")
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train[:100]")

    # Entraîner
    print("Démarrage de l'entraînement...")
    stats = trainer.train(dataset)

    print(f"\n✅ Entraînement terminé! Stats: {stats}")


def example_2_custom_config():
    """Exemple 2: Configuration personnalisée."""
    print("\n" + "=" * 60)
    print("EXEMPLE 2: Configuration Personnalisée")
    print("=" * 60 + "\n")

    # Configuration détaillée
    config = OptimizedTrainerConfig(
        # Model
        model_name="deepseek-ai/DeepSeek-OCR",
        output_dir="./checkpoints/custom_run",

        # Training
        batch_size=8,
        gradient_accumulation_steps=4,  # Batch effectif = 8 * 4 = 32
        num_epochs=3,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # Optimization
        mixed_precision="bf16",  # Utiliser bf16 au lieu de fp16
        use_gradient_scaling=True,  # Gradient scaling automatique
        compile_model=False,  # PyTorch 2.0 compilation

        # DataLoader (optimisations de performance)
        num_workers=4,  # Parallélisation du chargement
        prefetch_factor=2,  # Prefetch 2 batches à l'avance
        pin_memory=True,  # Pin memory pour GPU
        drop_last=True,  # Drop le dernier batch incomplet

        # Checkpointing
        save_interval=500,  # Sauvegarder tous les 500 steps
        save_total_limit=3,  # Garder seulement les 3 derniers checkpoints
        eval_interval=100,  # Évaluer tous les 100 steps

        # Logging
        log_interval=10,  # Logger tous les 10 steps
        use_tensorboard=True,
        use_wandb=False,
    )

    trainer = OptimizedDeepSynthTrainer(config)

    print(f"✅ Trainer configuré:")
    print(f"  - Batch effectif: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  - Précision mixte: {config.mixed_precision}")
    print(f"  - Workers: {config.num_workers}")
    print(f"  - Gradient scaling: {config.use_gradient_scaling}")


def example_3_training_with_eval():
    """Exemple 3: Entraînement avec évaluation."""
    print("\n" + "=" * 60)
    print("EXEMPLE 3: Entraînement avec Évaluation")
    print("=" * 60 + "\n")

    config = OptimizedTrainerConfig(
        batch_size=4,
        num_epochs=2,
        eval_interval=50,
        num_workers=2,
    )

    trainer = create_trainer(config=config)

    # Charger train et validation
    print("Chargement des datasets...")
    train_data = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="train[:200]")
    eval_data = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="validation[:50]")

    # Entraîner avec évaluation
    print("Entraînement avec évaluation...")
    stats = trainer.train(
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    print(f"\n✅ Entraînement terminé!")
    print(f"  - Epochs: {len(stats['epochs'])}")
    print(f"  - Best loss: {trainer.best_loss:.4f}")


def example_4_resume_from_checkpoint():
    """Exemple 4: Reprendre depuis un checkpoint."""
    print("\n" + "=" * 60)
    print("EXEMPLE 4: Reprendre depuis un Checkpoint")
    print("=" * 60 + "\n")

    # Configuration avec checkpoint
    config = OptimizedTrainerConfig(
        output_dir="./checkpoints/resumable",
        resume_from_checkpoint="./checkpoints/resumable/epoch_0",  # Reprendre ici
        batch_size=4,
        num_epochs=5,
    )

    try:
        trainer = OptimizedDeepSynthTrainer(config)
        print(f"✅ Checkpoint chargé:")
        print(f"  - Epoch actuel: {trainer.current_epoch}")
        print(f"  - Step global: {trainer.global_step}")
        print(f"  - Meilleur loss: {trainer.best_loss:.4f}")
    except FileNotFoundError as e:
        print(f"⚠️  Checkpoint non trouvé: {e}")
        print("  → Créer d'abord un checkpoint avec example_3")


def example_5_fp16_with_gradient_scaling():
    """Exemple 5: FP16 avec gradient scaling."""
    print("\n" + "=" * 60)
    print("EXEMPLE 5: FP16 avec Gradient Scaling")
    print("=" * 60 + "\n")

    config = OptimizedTrainerConfig(
        mixed_precision="fp16",  # FP16 pour plus de vitesse
        use_gradient_scaling=True,  # CRITIQUE pour stabilité FP16
        batch_size=16,  # Peut utiliser plus gros batch avec FP16
        max_grad_norm=1.0,  # Gradient clipping
    )

    trainer = OptimizedDeepSynthTrainer(config)

    print(f"✅ Configuration FP16:")
    print(f"  - Mixed precision: {config.mixed_precision}")
    print(f"  - Gradient scaler: {trainer.scaler is not None}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Gradient clipping: {config.max_grad_norm}")

    if trainer.scaler:
        print("\n  Le gradient scaler gère automatiquement:")
        print("  1. scaler.scale(loss).backward()")
        print("  2. scaler.unscale_(optimizer)")
        print("  3. gradient clipping")
        print("  4. scaler.step(optimizer)")
        print("  5. scaler.update()")


def example_6_distributed_training():
    """Exemple 6: Entraînement distribué."""
    print("\n" + "=" * 60)
    print("EXEMPLE 6: Entraînement Distribué")
    print("=" * 60 + "\n")

    print("Pour l'entraînement distribué multi-GPU:")
    print("\n1. Utiliser accelerate launch:")
    print("   $ accelerate config")
    print("   $ accelerate launch train_script.py")
    print("\n2. Le trainer détecte automatiquement l'environnement distribué")
    print("\n3. Code identique - aucune modification nécessaire!")

    config = OptimizedTrainerConfig(
        batch_size=4,  # Par GPU
        num_epochs=3,
    )

    trainer = OptimizedDeepSynthTrainer(config)

    print(f"\n✅ Configuration:")
    print(f"  - Nombre de processus: {trainer.accelerator.num_processes}")
    print(f"  - Process index: {trainer.accelerator.process_index}")
    print(f"  - Est main process: {trainer.accelerator.is_main_process}")


def example_7_custom_dataset():
    """Exemple 7: Dataset personnalisé."""
    print("\n" + "=" * 60)
    print("EXEMPLE 7: Dataset Personnalisé")
    print("=" * 60 + "\n")

    from deepsynth.training.optimized_trainer import DeepSynthDataset
    from transformers import AutoTokenizer

    # Créer des données custom
    custom_data = [
        {
            "text": "Premier document de test avec du contenu substantiel.",
            "summary": "Premier résumé",
        },
        {
            "text": "Deuxième document avec plus de contenu pour tester.",
            "summary": "Deuxième résumé",
        },
        {
            "text": "Troisième document contenant des informations importantes.",
            "summary": "Troisième résumé",
        },
    ]

    # Créer le dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = DeepSynthDataset(
        custom_data,
        tokenizer,
        max_length=128,
        cache_encodings=True,  # Pre-encode pour plus de vitesse
    )

    print(f"✅ Dataset créé:")
    print(f"  - Nombre d'exemples: {len(dataset)}")
    print(f"  - Encodages cachés: {len(dataset._encoding_cache)}")

    # Afficher un exemple
    item = dataset[0]
    print(f"\n  Exemple d'item:")
    print(f"  - input_ids shape: {item['input_ids'].shape}")
    print(f"  - attention_mask shape: {item['attention_mask'].shape}")
    print(f"  - labels shape: {item['labels'].shape}")


def example_8_performance_comparison():
    """Exemple 8: Comparaison de performance."""
    print("\n" + "=" * 60)
    print("EXEMPLE 8: Comparaison de Performance")
    print("=" * 60 + "\n")

    import time

    # Configuration de base
    base_config = OptimizedTrainerConfig(
        batch_size=4,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=False,
    )

    # Configuration optimisée
    optimized_config = OptimizedTrainerConfig(
        batch_size=4,
        num_workers=4,  # Parallélisation
        prefetch_factor=2,  # Prefetching
        pin_memory=True,  # Pin memory
        mixed_precision="bf16",  # Mixed precision
        use_gradient_scaling=True,
    )

    print("Configuration de base:")
    print(f"  - Workers: {base_config.num_workers}")
    print(f"  - Pin memory: {base_config.pin_memory}")
    print(f"  - Mixed precision: {base_config.mixed_precision}")

    print("\nConfiguration optimisée:")
    print(f"  - Workers: {optimized_config.num_workers}")
    print(f"  - Pin memory: {optimized_config.pin_memory}")
    print(f"  - Mixed precision: {optimized_config.mixed_precision}")

    print("\n📊 Gains attendus:")
    print("  - DataLoader parallèle: +40% vitesse")
    print("  - Pin memory: +10% vitesse GPU")
    print("  - Mixed precision: +30% vitesse + -50% mémoire")
    print("  - Gradient scaling: Stabilité numérique pour fp16")


def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Exemples OptimizedTrainer")
    parser.add_argument(
        "--example",
        "-e",
        type=int,
        choices=range(1, 9),
        help="Numéro de l'exemple à exécuter (1-8)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécuter tous les exemples (sauf entraînement complet)",
    )

    args = parser.parse_args()

    examples = {
        1: ("Démarrage Rapide", example_1_quick_start),
        2: ("Configuration Personnalisée", example_2_custom_config),
        3: ("Training avec Évaluation", example_3_training_with_eval),
        4: ("Reprendre depuis Checkpoint", example_4_resume_from_checkpoint),
        5: ("FP16 + Gradient Scaling", example_5_fp16_with_gradient_scaling),
        6: ("Entraînement Distribué", example_6_distributed_training),
        7: ("Dataset Personnalisé", example_7_custom_dataset),
        8: ("Comparaison Performance", example_8_performance_comparison),
    }

    if args.all:
        # Exécuter tous les exemples sauf 1 et 3 (qui font vraiment l'entraînement)
        for num, (name, func) in examples.items():
            if num not in [1, 3]:
                try:
                    func()
                except Exception as e:
                    print(f"❌ Erreur dans exemple {num}: {e}")
    elif args.example:
        name, func = examples[args.example]
        print(f"\n🚀 Exécution: {name}")
        func()
    else:
        print("\n📚 Exemples Disponibles:\n")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\nUtilisation:")
        print("  python train_with_optimized_trainer.py --example 2")
        print("  python train_with_optimized_trainer.py --all")


if __name__ == "__main__":
    main()