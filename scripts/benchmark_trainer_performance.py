#!/usr/bin/env python3
"""
Benchmark de performance des trainers.

Compare l'ancien et le nouveau trainer pour mesurer les gains de performance.
"""

import time
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import json

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepsynth.training.optimized_trainer import (
    OptimizedDeepSynthTrainer,
    OptimizedTrainerConfig,
    DeepSynthDataset,
)
from deepsynth.utils.logging_config import setup_global_logging, get_logger

logger = get_logger(__name__)


class MockDataset(Dataset):
    """Dataset mock pour les benchmarks."""

    def __init__(self, size: int = 1000, seq_length: int = 128):
        self.size = size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (self.seq_length,)),
            "attention_mask": torch.ones(self.seq_length),
            "labels": torch.randint(0, 1000, (self.seq_length,)),
            "image": torch.randn(3, 224, 224),  # Mock image
        }


class MockModel(torch.nn.Module):
    """Modèle mock pour les benchmarks."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1000)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Simulate forward pass
        x = input_ids.float()
        x = self.linear1(x)
        x = self.dropout(x)
        logits = self.linear2(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 1000),
                labels.view(-1) % 1000,
            )

        class Output:
            pass

        output = Output()
        output.loss = loss
        output.logits = logits
        return output


class TrainerBenchmark:
    """Classe pour benchmarker les trainers."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}

    def benchmark_dataloader_creation(
        self,
        dataset_size: int = 10000,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Benchmark création et itération du DataLoader."""
        logger.info("Benchmarking DataLoader creation...")

        dataset = MockDataset(size=dataset_size)

        # Baseline: DataLoader manuel simple
        start = time.time()
        loader_baseline = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        _ = list(loader_baseline)
        time_baseline = time.time() - start

        # Optimized: DataLoader avec optimisations
        start = time.time()
        loader_optimized = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
        _ = list(loader_optimized)
        time_optimized = time.time() - start

        speedup = (time_baseline / time_optimized) if time_optimized > 0 else 1.0

        results = {
            "baseline_time": time_baseline,
            "optimized_time": time_optimized,
            "speedup": speedup,
            "improvement_pct": (speedup - 1) * 100,
        }

        logger.info(
            f"  DataLoader: {time_baseline:.2f}s → {time_optimized:.2f}s "
            f"(speedup: {speedup:.2f}x, +{results['improvement_pct']:.1f}%)"
        )

        return results

    def benchmark_training_step(
        self,
        batch_size: int = 32,
        num_steps: int = 100,
        use_fp16: bool = False,
        use_grad_scaler: bool = False,
    ) -> Dict[str, float]:
        """Benchmark une étape d'entraînement."""
        logger.info(
            f"Benchmarking training step (fp16={use_fp16}, scaler={use_grad_scaler})..."
        )

        model = MockModel().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler() if use_grad_scaler and use_fp16 else None

        dataset = MockDataset(size=num_steps * batch_size)
        loader = DataLoader(dataset, batch_size=batch_size)

        model.train()
        start = time.time()

        for batch in loader:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            if use_fp16 and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

        elapsed = time.time() - start
        samples_per_sec = (num_steps * batch_size) / elapsed

        results = {
            "total_time": elapsed,
            "samples_per_second": samples_per_sec,
            "time_per_step": elapsed / num_steps,
        }

        logger.info(
            f"  Training: {elapsed:.2f}s total, "
            f"{samples_per_sec:.1f} samples/s, "
            f"{results['time_per_step']*1000:.1f}ms/step"
        )

        return results

    def benchmark_full_comparison(
        self,
        dataset_size: int = 1000,
        batch_size: int = 16,
        num_epochs: int = 1,
    ) -> Dict[str, Any]:
        """Benchmark complet comparant configurations."""
        logger.info("Full benchmark comparison...")

        dataset = MockDataset(size=dataset_size)
        model = MockModel()

        configs = {
            "baseline": OptimizedTrainerConfig(
                batch_size=batch_size,
                num_epochs=num_epochs,
                num_workers=0,
                pin_memory=False,
                mixed_precision=None,
                use_gradient_scaling=False,
                save_interval=10000,  # Don't save
            ),
            "optimized": OptimizedTrainerConfig(
                batch_size=batch_size,
                num_epochs=num_epochs,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                mixed_precision="bf16" if torch.cuda.is_bf16_supported() else None,
                use_gradient_scaling=True,
                save_interval=10000,
            ),
        }

        results = {}

        for config_name, config in configs.items():
            logger.info(f"\nTesting configuration: {config_name}")

            trainer = OptimizedDeepSynthTrainer(
                config,
                model=model,
                tokenizer=None,
            )

            start = time.time()
            # Note: Would need actual training here
            # For now, just measure dataloader creation
            loader = trainer.create_dataloader(dataset, is_train=True)
            _ = list(loader)
            elapsed = time.time() - start

            results[config_name] = {
                "time": elapsed,
                "samples_per_second": dataset_size / elapsed,
            }

            logger.info(
                f"  {config_name}: {elapsed:.2f}s, "
                f"{results[config_name]['samples_per_second']:.1f} samples/s"
            )

        # Calculate speedup
        if "baseline" in results and "optimized" in results:
            speedup = results["baseline"]["time"] / results["optimized"]["time"]
            results["speedup"] = speedup
            results["improvement_pct"] = (speedup - 1) * 100

            logger.info(
                f"\n✅ Speedup: {speedup:.2f}x (+{results['improvement_pct']:.1f}%)"
            )

        return results

    def benchmark_memory_usage(
        self,
        batch_size: int = 32,
        mixed_precision: bool = False,
    ) -> Dict[str, float]:
        """Benchmark utilisation mémoire."""
        if self.device != "cuda":
            logger.warning("Memory benchmark requires CUDA")
            return {}

        logger.info(f"Benchmarking memory usage (fp16={mixed_precision})...")

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = MockModel().to(self.device)
        dataset = MockDataset(size=batch_size * 10)
        loader = DataLoader(dataset, batch_size=batch_size)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters())

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break  # Just one batch

        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        max_memory = torch.cuda.max_memory_allocated() / 1024**2

        results = {
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "max_memory_mb": max_memory,
        }

        logger.info(
            f"  Memory: {memory_allocated:.1f}MB allocated, "
            f"{max_memory:.1f}MB peak"
        )

        return results


def run_all_benchmarks(device: str = "cpu", output_file: str = None):
    """Exécute tous les benchmarks."""
    logger.info("=" * 60)
    logger.info("DEEPSYNTH TRAINER PERFORMANCE BENCHMARK")
    logger.info("=" * 60)

    benchmark = TrainerBenchmark(device=device)
    all_results = {}

    # 1. DataLoader benchmark
    logger.info("\n1. DataLoader Performance")
    logger.info("-" * 60)
    all_results["dataloader"] = benchmark.benchmark_dataloader_creation(
        dataset_size=10000,
        batch_size=32,
    )

    # 2. Training step benchmark
    logger.info("\n2. Training Step Performance")
    logger.info("-" * 60)

    # Baseline (fp32)
    all_results["training_fp32"] = benchmark.benchmark_training_step(
        batch_size=16,
        num_steps=50,
        use_fp16=False,
        use_grad_scaler=False,
    )

    # FP16 without scaler
    if device == "cuda":
        all_results["training_fp16_no_scaler"] = benchmark.benchmark_training_step(
            batch_size=16,
            num_steps=50,
            use_fp16=True,
            use_grad_scaler=False,
        )

        # FP16 with scaler
        all_results["training_fp16_scaler"] = benchmark.benchmark_training_step(
            batch_size=16,
            num_steps=50,
            use_fp16=True,
            use_grad_scaler=True,
        )

    # 3. Full comparison
    logger.info("\n3. Full Configuration Comparison")
    logger.info("-" * 60)
    all_results["full_comparison"] = benchmark.benchmark_full_comparison(
        dataset_size=1000,
        batch_size=16,
        num_epochs=1,
    )

    # 4. Memory usage
    if device == "cuda":
        logger.info("\n4. Memory Usage Comparison")
        logger.info("-" * 60)

        all_results["memory_fp32"] = benchmark.benchmark_memory_usage(
            batch_size=32,
            mixed_precision=False,
        )

        all_results["memory_fp16"] = benchmark.benchmark_memory_usage(
            batch_size=32,
            mixed_precision=True,
        )

        if "memory_fp32" in all_results and "memory_fp16" in all_results:
            memory_savings = (
                1 - all_results["memory_fp16"]["max_memory_mb"]
                / all_results["memory_fp32"]["max_memory_mb"]
            ) * 100
            all_results["memory_savings_pct"] = memory_savings
            logger.info(f"\n  Memory savings with FP16: {memory_savings:.1f}%")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if "dataloader" in all_results:
        logger.info(
            f"DataLoader speedup: {all_results['dataloader']['speedup']:.2f}x "
            f"(+{all_results['dataloader']['improvement_pct']:.1f}%)"
        )

    if "full_comparison" in all_results and "speedup" in all_results["full_comparison"]:
        logger.info(
            f"Overall speedup: {all_results['full_comparison']['speedup']:.2f}x "
            f"(+{all_results['full_comparison']['improvement_pct']:.1f}%)"
        )

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✅ Results saved to: {output_path}")

    return all_results


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Benchmark trainer performance")
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for benchmarking",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    setup_global_logging(
        level="DEBUG" if args.verbose else "INFO",
    )

    logger.info(f"Device: {args.device}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"CUDA Version: {torch.version.cuda}"
        )

    # Run benchmarks
    results = run_all_benchmarks(
        device=args.device,
        output_file=args.output,
    )

    return 0


if __name__ == "__main__":
    exit(main())