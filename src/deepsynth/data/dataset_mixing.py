"""
Dataset Mixing Utilities.

Combine multiple Q&A datasets with:
- Stratified sampling
- Deduplication
- Balanced mixing
- Quality filtering
"""

import hashlib
import logging
import random
from collections import Counter
from typing import Dict, List, Optional

from datasets import Dataset, concatenate_datasets

from .instruction_dataset import InstructionDataset

LOGGER = logging.getLogger(__name__)


def mix_datasets(
    datasets: List[InstructionDataset],
    weights: Optional[List[float]] = None,
    max_samples_per_dataset: Optional[int] = None,
    shuffle: bool = True,
    deduplicate: bool = True,
    seed: int = 42,
) -> InstructionDataset:
    """
    Mix multiple datasets with optional weights and deduplication.

    Args:
        datasets: List of InstructionDataset objects
        weights: Optional weights for each dataset (None = equal weights)
        max_samples_per_dataset: Maximum samples from each dataset
        shuffle: Shuffle final dataset
        deduplicate: Remove duplicate questions
        seed: Random seed for reproducibility

    Returns:
        Mixed InstructionDataset

    Example:
        >>> squad = create_instruction_dataset_from_qa(...)
        >>> nq = convert_natural_questions(...)
        >>> mixed = mix_datasets([squad, nq], weights=[0.7, 0.3])
        >>> len(mixed)
        100000
    """
    random.seed(seed)

    if not datasets:
        raise ValueError("No datasets provided")

    # Default to equal weights
    if weights is None:
        weights = [1.0 / len(datasets)] * len(datasets)

    if len(weights) != len(datasets):
        raise ValueError(f"weights ({len(weights)}) must match datasets ({len(datasets)})")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    LOGGER.info(f"Mixing {len(datasets)} datasets with weights: {weights}")

    # Sample from each dataset
    all_samples = []
    dataset_stats = []

    for idx, (dataset, weight) in enumerate(zip(datasets, weights)):
        # Determine number of samples to take
        dataset_size = len(dataset)

        if max_samples_per_dataset:
            num_samples = min(max_samples_per_dataset, dataset_size)
        else:
            num_samples = dataset_size

        # Apply weight-based sampling
        weighted_samples = int(num_samples * weight * len(datasets))
        num_samples = min(weighted_samples, dataset_size)

        LOGGER.info(f"Dataset {idx}: Taking {num_samples}/{dataset_size} samples (weight: {weight:.2%})")

        # Sample indices
        indices = list(range(dataset_size))
        if num_samples < dataset_size:
            indices = random.sample(indices, num_samples)

        # Extract samples
        for i in indices:
            sample = dataset[i]
            all_samples.append(sample)

        dataset_stats.append({
            "index": idx,
            "original_size": dataset_size,
            "sampled_size": num_samples,
            "weight": weight,
        })

    LOGGER.info(f"Total samples before deduplication: {len(all_samples)}")

    # Deduplicate
    if deduplicate:
        all_samples = _deduplicate_samples(all_samples)
        LOGGER.info(f"Total samples after deduplication: {len(all_samples)}")

    # Shuffle
    if shuffle:
        random.shuffle(all_samples)

    # Print statistics
    _print_dataset_stats(dataset_stats, len(all_samples))

    return InstructionDataset(all_samples, split="train", use_augmentation=False)


def _deduplicate_samples(samples: List[dict]) -> List[dict]:
    """
    Remove duplicate samples based on instruction hash.

    Args:
        samples: List of sample dictionaries

    Returns:
        Deduplicated samples
    """
    seen_hashes = set()
    unique_samples = []
    duplicates = 0

    for sample in samples:
        # Create hash from instruction (questions should be unique)
        instruction = sample.get("instruction", "")
        sample_hash = hashlib.md5(instruction.lower().strip().encode()).hexdigest()

        if sample_hash not in seen_hashes:
            seen_hashes.add(sample_hash)
            unique_samples.append(sample)
        else:
            duplicates += 1

    if duplicates > 0:
        LOGGER.info(f"Removed {duplicates} duplicate samples")

    return unique_samples


def _print_dataset_stats(dataset_stats: List[dict], final_size: int):
    """Print dataset mixing statistics."""
    LOGGER.info("=" * 80)
    LOGGER.info("Dataset Mixing Statistics")
    LOGGER.info("=" * 80)

    for stats in dataset_stats:
        LOGGER.info(
            f"Dataset {stats['index']}: "
            f"{stats['sampled_size']}/{stats['original_size']} samples "
            f"({stats['sampled_size']/final_size*100:.1f}% of final)"
        )

    LOGGER.info(f"Final dataset size: {final_size}")
    LOGGER.info("=" * 80)


def stratified_sample(
    dataset: InstructionDataset,
    num_samples: int,
    stratify_by: str = "source",
    seed: int = 42,
) -> InstructionDataset:
    """
    Perform stratified sampling from a dataset.

    Args:
        dataset: InstructionDataset
        num_samples: Number of samples to draw
        stratify_by: Metadata field to stratify by
        seed: Random seed

    Returns:
        Stratified sample dataset

    Example:
        >>> sampled = stratified_sample(dataset, num_samples=1000, stratify_by="source")
    """
    random.seed(seed)

    # Group samples by stratification key
    strata = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        key = sample.get("metadata", {}).get(stratify_by, "unknown")

        if key not in strata:
            strata[key] = []
        strata[key].append(sample)

    LOGGER.info(f"Found {len(strata)} strata: {list(strata.keys())}")

    # Calculate samples per stratum
    total_samples = sum(len(samples) for samples in strata.values())
    samples_per_stratum = {
        key: max(1, int(len(samples) / total_samples * num_samples))
        for key, samples in strata.items()
    }

    # Sample from each stratum
    sampled = []
    for key, samples in strata.items():
        n = min(samples_per_stratum[key], len(samples))
        sampled.extend(random.sample(samples, n))

    LOGGER.info(f"Stratified sample: {len(sampled)} samples")

    return InstructionDataset(sampled, split=dataset.split, use_augmentation=False)


__all__ = ["mix_datasets", "stratified_sample"]
