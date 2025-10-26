#!/usr/bin/env python3
"""Script de lancement principal pour le traitement parallèle des datasets.

Usage:
    deepsynth-parallel
"""

from deepsynth.pipelines import run_parallel_datasets_pipeline


def main() -> int:
    """Execute the parallel processing pipeline."""

    return run_parallel_datasets_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())