#!/usr/bin/env python3
"""Test minimal pour diagnostiquer le problème de streaming Natural Questions."""

import sys
import time
sys.path.insert(0, "./src")

print("=" * 60)
print("TEST MINIMAL STREAMING NATURAL QUESTIONS")
print("=" * 60)

print("\n1️⃣ Testing dataset loading...")
start = time.time()

try:
    from datasets import load_dataset

    print("   Loading Natural Questions (streaming=True)...")
    dataset = load_dataset("natural_questions", split="train", streaming=True)

    print(f"   ✅ Dataset loaded in {time.time() - start:.1f}s")
    print(f"   Dataset type: {type(dataset)}")

    print("\n2️⃣ Testing iteration (first 5 samples)...")
    iter_start = time.time()

    for i, sample in enumerate(dataset):
        elapsed = time.time() - iter_start
        print(f"   Sample {i+1}: {elapsed:.1f}s elapsed")

        # Check sample structure
        if i == 0:
            print(f"   Keys: {list(sample.keys())}")

        if i >= 4:
            break

    print(f"\n✅ Iteration successful! Total time: {time.time() - start:.1f}s")
    print("\n💡 If this works, the problem is likely in the converter logic.")

except KeyboardInterrupt:
    print("\n\n⚠️  INTERRUPTED - This suggests the dataset download is stuck")
    print("   Possible causes:")
    print("   • Slow internet connection")
    print("   • HuggingFace servers slow")
    print("   • Dataset shard download stuck")
    print("\n   Try: export HF_DATASETS_OFFLINE=0")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
