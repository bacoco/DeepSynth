#!/usr/bin/env python3
"""Vérifie ce qui a été uploadé sur HuggingFace."""

import sys
sys.path.insert(0, "./src")

from deepsynth.config import Config
from huggingface_hub import HfApi, hf_hub_download
import json

print("=" * 60)
print("VÉRIFICATION UPLOAD HUGGINGFACE")
print("=" * 60)

config = Config.from_env()
api = HfApi(token=config.hf_token)

dataset_name = f"{config.hf_username}/deepsynth-qa"

print(f"\n🔍 Dataset: {dataset_name}")

try:
    # Lister les fichiers
    print("\n📁 FICHIERS SUR HUGGINGFACE:")
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")

    if not files:
        print("   ⚠️  Aucun fichier trouvé!")
    else:
        for f in sorted(files):
            print(f"   - {f}")

    # Vérifier l'index des shards
    print("\n📊 INDEX DES SHARDS:")
    try:
        index_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename="data/shards.json",
            token=config.hf_token
        )
        with open(index_path) as f:
            index = json.load(f)

        shards = index.get("shards", [])
        print(f"   Nombre de shards: {len(shards)}")
        for shard in shards:
            print(f"   - {shard['id']}: {shard['num_samples']} samples")
    except Exception as e:
        print(f"   ⚠️  Pas d'index shards.json: {e}")

    # Info du repo
    print("\n📈 INFO DU REPO:")
    info = api.repo_info(repo_id=dataset_name, repo_type="dataset")
    print(f"   Dernière modification: {info.last_modified}")
    print(f"   Privé: {info.private}")

except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    print("\n💡 Le dataset existe-t-il sur HuggingFace?")
    print(f"   https://huggingface.co/datasets/{dataset_name}")

print("\n" + "=" * 60)
