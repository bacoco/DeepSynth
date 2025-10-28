#!/usr/bin/env python3
"""
Test complet Q&A avec upload sur HuggingFace.
Génère un petit dataset de test et l'uploade pour validation.
"""

import sys
import os
sys.path.insert(0, "./src")

from datasets import Dataset, Features, Value, Image as HFImage
from deepsynth.config import Config
from deepsynth.data.dataset_converters import convert_natural_questions, convert_ms_marco

print("🧪 TEST COMPLET Q&A AVEC UPLOAD HUGGINGFACE")
print("=" * 80)

# Load config
config = Config.from_env()

if not config.hf_token:
    print("❌ HF_TOKEN not found in .env")
    sys.exit(1)

print(f"✅ HuggingFace config loaded:")
print(f"   Username: {config.hf_username}")
print(f"   Token: {'*' * 10}")
print()

# Configuration du test
TEST_SAMPLES_PER_SOURCE = 10
DATASET_NAME = "deepsynth-qa-test"
REPO_ID = f"{config.hf_username}/{DATASET_NAME}"

print(f"📊 Configuration:")
print(f"   Samples per source: {TEST_SAMPLES_PER_SOURCE}")
print(f"   Output dataset: {REPO_ID}")
print(f"   Resolution: gundam (1600px)")
print()

# Collecter tous les samples
all_samples = []

# Natural Questions
print("=" * 80)
print("📖 PROCESSING NATURAL QUESTIONS")
print("=" * 80)

nq_dataset = convert_natural_questions(
    split="train",
    max_samples=TEST_SAMPLES_PER_SOURCE,
    streaming=True,
    target_resolution="gundam"
)

nq_count = 0
for sample in nq_dataset:
    all_samples.append(sample)
    nq_count += 1
    print(f"✓ Processed Natural Questions sample {nq_count}")
    if nq_count >= TEST_SAMPLES_PER_SOURCE:
        break

print(f"✅ Natural Questions: {nq_count} samples collected")
print()

# MS MARCO
print("=" * 80)
print("📚 PROCESSING MS MARCO")
print("=" * 80)

marco_dataset = convert_ms_marco(
    config="v2.1",
    split="train",
    max_samples=TEST_SAMPLES_PER_SOURCE,
    streaming=True,
    target_resolution="gundam"
)

marco_count = 0
for sample in marco_dataset:
    all_samples.append(sample)
    marco_count += 1
    print(f"✓ Processed MS MARCO sample {marco_count}")
    if marco_count >= TEST_SAMPLES_PER_SOURCE:
        break

print(f"✅ MS MARCO: {marco_count} samples collected")
print()

# Statistiques
print("=" * 80)
print("📊 DATASET STATISTICS")
print("=" * 80)
print(f"Total samples: {len(all_samples)}")
print(f"  - Natural Questions: {nq_count}")
print(f"  - MS MARCO: {marco_count}")
print()

# Vérifier sources
nq_samples = [s for s in all_samples if s['metadata']['source'] == 'natural_questions']
marco_samples = [s for s in all_samples if s['metadata']['source'] == 'ms_marco']
print(f"Source distribution:")
print(f"  - natural_questions: {len(nq_samples)}")
print(f"  - ms_marco: {len(marco_samples)}")
print()

# Sample inspection
if all_samples:
    sample = all_samples[0]
    print(f"Sample structure:")
    print(f"  - text: {len(sample['text'])} chars")
    print(f"  - instruction: {sample['instruction'][:60]}...")
    print(f"  - answer: {sample['answer'][:60]}...")
    print(f"  - short_answer: {sample.get('short_answer', 'N/A')[:40]}...")
    print(f"  - long_answer: {sample.get('long_answer', 'N/A')[:40]}...")
    print(f"  - image: {type(sample['image'])}")
    print(f"  - quality: {sample['quality']}")
    print(f"  - token_count: {sample['token_count']}")
    print(f"  - extracted_token_count: {sample.get('extracted_token_count', 'N/A')}")
    print(f"  - metadata.source: {sample['metadata']['source']}")
    print(f"  - metadata.generation_resolution: {sample['metadata']['generation_resolution']}")
    print()

# Créer le dataset HuggingFace
print("=" * 80)
print("🚀 CREATING HUGGINGFACE DATASET")
print("=" * 80)

# Préparer les samples pour HuggingFace (enlever les tensors, garder PIL Images)
print("Preparing samples for upload...")
from PIL import Image
import torch

prepared_samples = []
for sample in all_samples:
    # Convertir tensor en PIL Image si nécessaire
    img = sample['image']
    if isinstance(img, torch.Tensor):
        # Skip tensor conversion, use original from converter
        print(f"⚠️  Sample has tensor image, getting original...")
        continue  # For now skip tensors

    prepared_sample = {
        'text': sample['text'],
        'instruction': sample['instruction'],
        'answer': sample['answer'],
        'short_answer': sample.get('short_answer', ''),
        'long_answer': sample.get('long_answer', ''),
        'answer_start_token': sample.get('answer_start_token', -1) or -1,
        'answer_end_token': sample.get('answer_end_token', -1) or -1,
        'image': img if not isinstance(img, torch.Tensor) else None,
        'quality': sample['quality'],
        'estimated_height': sample['estimated_height'],
        'token_count': sample['token_count'],
        'extracted_token_count': sample.get('extracted_token_count', sample['token_count']),
        'source': sample['metadata']['source'],
        'generation_resolution': sample['metadata']['generation_resolution'],
        'extraction_method': sample['metadata'].get('extraction_method', 'unknown'),
    }

    if prepared_sample['image'] is not None:
        prepared_samples.append(prepared_sample)

print(f"Prepared {len(prepared_samples)} samples for upload")

# Créer le dataset sans features explicites (auto-infer)
print("Creating dataset from samples...")
hf_dataset = Dataset.from_list(prepared_samples)
print(f"✅ Dataset created: {len(hf_dataset)} samples")
print()

# Upload vers HuggingFace
print("=" * 80)
print("📤 UPLOADING TO HUGGINGFACE")
print("=" * 80)
print(f"Repository: {REPO_ID}")
print(f"Uploading {len(hf_dataset)} samples...")
print()

try:
    hf_dataset.push_to_hub(
        REPO_ID,
        token=config.hf_token,
        private=False,
        commit_message=f"Test Q&A dataset with {len(hf_dataset)} samples (Natural Questions + MS MARCO)"
    )

    print("✅ UPLOAD SUCCESSFUL!")
    print()
    print("=" * 80)
    print("🎉 TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print(f"📊 Dataset Info:")
    print(f"   - Name: {DATASET_NAME}")
    print(f"   - Repository: {REPO_ID}")
    print(f"   - URL: https://huggingface.co/datasets/{REPO_ID}")
    print(f"   - Total samples: {len(hf_dataset)}")
    print(f"   - Natural Questions: {len(nq_samples)}")
    print(f"   - MS MARCO: {len(marco_samples)}")
    print()
    print(f"✨ View your dataset at:")
    print(f"   https://huggingface.co/datasets/{REPO_ID}")
    print()
    print("=" * 80)

except Exception as e:
    print(f"❌ UPLOAD FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
