#!/usr/bin/env python3
"""
Script de lancement principal pour le traitement parallèle des datasets
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsynth.pipelines.parallel.run_parallel_datasets import run_parallel_datasets_cli

if __name__ == "__main__":
    exit_code = run_parallel_datasets_cli()
    sys.exit(exit_code)