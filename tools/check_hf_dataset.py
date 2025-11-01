#!/usr/bin/env python3
"""Vérification rapide de l'état du dataset sur HuggingFace"""
import sys
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET = "baconnier/deepsynth-qa"

print("=" * 70)
print(f"🔍 VÉRIFICATION DATASET: {DATASET}")
print("=" * 70)

# 1. Info du repo
try:
    api = HfApi()
    info = api.repo_info(repo_id=DATASET, repo_type="dataset")
    print(f"\n📅 Dernière mise à jour: {info.last_modified}")
    print(f"📦 Taille: {info.size_on_disk / (1024**3):.2f} GB" if hasattr(info, 'size_on_disk') else "📦 Taille: N/A")
except Exception as e:
    print(f"❌ Erreur info: {e}")
    sys.exit(1)

# 2. Liste des fichiers
try:
    files = list(api.list_repo_files(repo_id=DATASET, repo_type="dataset"))
    parquet_files = [f for f in files if f.endswith('.parquet')]
    batch_dirs = set([f.split('/')[0] for f in files if 'batch_' in f])
    
    print(f"\n📊 Fichiers Parquet: {len(parquet_files)}")
    print(f"📁 Batches: {len(batch_dirs)}")
    if parquet_files:
        print("\n📄 Derniers fichiers Parquet:")
        for f in sorted(parquet_files)[-5:]:
            print(f"   - {f}")
except Exception as e:
    print(f"⚠️  Erreur liste fichiers: {e}")

# 3. Comptage des échantillons (streaming rapide)
try:
    print(f"\n🔢 Comptage des échantillons (échantillonnage)...")
    ds = load_dataset(DATASET, split='train', streaming=True)
    
    # Échantillonnage rapide
    count = 0
    for i, sample in enumerate(ds):
        count = i + 1
        if count % 1000 == 0:
            print(f"   ... {count} échantillons vus")
        if count >= 10000:  # Limite pour rapidité
            print(f"   ✅ Au moins {count} échantillons présents")
            break
    
    if count < 10000:
        print(f"   📊 Total exact: {count} échantillons")
    
    # Affiche un exemple
    if count > 0:
        print(f"\n📝 Exemple de métadonnées:")
        example = next(iter(ds))
        if 'metadata' in example:
            meta = example['metadata']
            print(f"   Source: {meta.get('source', 'N/A')}")
            print(f"   Language: {meta.get('language', 'N/A')}")
            print(f"   Quality: {meta.get('quality', 'N/A')}")
        
except Exception as e:
    print(f"❌ Erreur comptage: {e}")

print("\n" + "=" * 70)
print("✅ Vérification terminée")
print("=" * 70)
