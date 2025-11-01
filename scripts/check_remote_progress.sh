#!/bin/bash
# Script pour monitorer le processus distant

echo "=== ÉTAT DU DATASET HuggingFace ==="
huggingface-cli repo ls baconnier/deepsynth-qa --repo-type dataset 2>&1 | grep -E "batch_|parquet" | tail -20

echo ""
echo "=== STATISTIQUES ==="
PYTHONPATH=./src python3 -c "
from datasets import load_dataset
try:
    ds = load_dataset('baconnier/deepsynth-qa', split='train', streaming=True)
    count = 0
    for i, _ in enumerate(ds):
        count = i + 1
        if count >= 100:  # Échantillonnage rapide
            print(f'✅ Au moins {count} échantillons visibles')
            break
    if count < 100:
        print(f'📊 Total échantillons: {count}')
except Exception as e:
    print(f'❌ Erreur: {e}')
"

echo ""
echo "=== DERNIERS COMMITS ==="
huggingface-cli repo info baconnier/deepsynth-qa --repo-type dataset 2>&1 | grep -A 5 "Last updated"
