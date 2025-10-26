#!/usr/bin/env python3
"""
Script de lancement principal pour le traitement parallèle des datasets
"""

import sys
import os

# Ajouter le répertoire du projet au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importer et lancer le script principal
from parallel_processing.run_parallel_datasets import main

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)