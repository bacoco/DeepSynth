#!/usr/bin/env python3
"""
Script de lancement principal pour le traitement parallèle des datasets
"""

import sys

from deepsynth.pipelines.parallel_cli import main

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)