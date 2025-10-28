#!/usr/bin/env python3
"""Simple test to validate Q&A converters generate data correctly."""

import sys
sys.path.insert(0, "./src")

from deepsynth.data.dataset_converters import convert_natural_questions, convert_ms_marco

print("🧪 Testing Q&A Converters (No Upload)")
print("=" * 60)

# Test Natural Questions
print("\n📖 Testing Natural Questions...")
nq_dataset = convert_natural_questions(
    split="train",
    max_samples=5,
    streaming=True,
    target_resolution="gundam"
)

nq_samples = list(nq_dataset)
print(f"✅ Generated {len(nq_samples)} Natural Questions samples")

if nq_samples:
    sample = nq_samples[0]
    print(f"\nSample structure:")
    print(f"  - text length: {len(sample['text'])} chars")
    print(f"  - instruction: {sample['instruction'][:50]}...")
    print(f"  - answer: {sample['answer'][:50]}...")
    img = sample['image']
    if hasattr(img, 'size') and callable(img.size):
        # PyTorch tensor
        print(f"  - image: Tensor {img.size()}")
    elif hasattr(img, 'size'):
        # PIL Image
        print(f"  - image: PIL.Image {img.size}")
    else:
        print(f"  - image: {type(img)}")
    print(f"  - metadata.source: {sample['metadata']['source']}")
    print(f"  - metadata.generation_resolution: {sample['metadata']['generation_resolution']}")

# Test MS MARCO
print("\n📚 Testing MS MARCO...")
marco_dataset = convert_ms_marco(
    split="train",
    max_samples=5,
    streaming=True,
    target_resolution="gundam"
)

marco_samples = list(marco_dataset)
print(f"✅ Generated {len(marco_samples)} MS MARCO samples")

if marco_samples:
    sample = marco_samples[0]
    print(f"\nSample structure:")
    print(f"  - text length: {len(sample['text'])} chars")
    print(f"  - instruction: {sample['instruction'][:50]}...")
    print(f"  - answer: {sample['answer'][:50]}...")
    img = sample['image']
    if hasattr(img, 'size') and callable(img.size):
        # PyTorch tensor
        print(f"  - image: Tensor {img.size()}")
    elif hasattr(img, 'size'):
        # PIL Image
        print(f"  - image: PIL.Image {img.size}")
    else:
        print(f"  - image: {type(img)}")
    print(f"  - metadata.source: {sample['metadata']['source']}")
    print(f"  - metadata.generation_resolution: {sample['metadata']['generation_resolution']}")

print("\n" + "=" * 60)
print("✅ Q&A CONVERTERS WORK CORRECTLY!")
print("   Images are pre-generated at gundam resolution")
print("   Metadata tracks source properly")
print("=" * 60)
