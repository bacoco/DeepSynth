#!/usr/bin/env python3
"""Create a mini dataset - 1000 samples."""

import os
from datasets import load_dataset, Dataset

def create_mini_dataset():
    print("Loading 1000 samples in streaming mode...")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required")

    # Stream and take 1000 samples
    ds_stream = load_dataset("baconnier/deepsynth-fr", split="train", streaming=True, token=token)

    samples = []
    for i, sample in enumerate(ds_stream):
        samples.append(sample)
        if i >= 999:
            break
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/1000...")

    print(f"Creating dataset with {len(samples)} samples...")
    ds_mini = Dataset.from_list(samples)

    print("Pushing to HuggingFace...")
    ds_mini.push_to_hub(
        "baconnier/deepsynth-fr-mini",
        token=token,
        private=False,
        commit_message="Mini dataset - 1000 samples"
    )

    print("Done!")

if __name__ == "__main__":
    create_mini_dataset()
