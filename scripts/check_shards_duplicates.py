#!/usr/bin/env python3
"""Utility to validate shard integrity on the HuggingFace Hub.

The script downloads the shard index (``data/shards.json``), streams each
shard listed in the file, and verifies that no
``(source_dataset, original_index)`` pair appears twice.  It also prints a
small per-source summary of the total samples processed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Tuple

from datasets import DatasetDict, load_from_disk
from huggingface_hub import hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace dataset repository id (e.g. username/dataset)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HuggingFace token. Falls back to HF_TOKEN env var if unset.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional path to reuse the huggingface_hub cache",
    )
    return parser.parse_args()


def iter_shard_examples(shard_dir: Path):
    dataset = load_from_disk(str(shard_dir))
    if isinstance(dataset, DatasetDict):
        iterable = dataset["train"]
    else:
        iterable = dataset
    for example in iterable:
        yield example


def main() -> int:
    args = parse_args()

    index_path = hf_hub_download(
        repo_id=args.repo,
        filename="data/shards.json",
        repo_type="dataset",
        token=args.token,
        cache_dir=args.cache_dir,
    )

    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)

    shards = index.get("shards", [])
    if not shards:
        print("⚠️ No shards listed in data/shards.json")
        return 0

    seen: set[Tuple[str, int]] = set()
    duplicates: list[Tuple[str, int, str]] = []
    per_source: Counter[str] = Counter()
    total_samples = 0

    for shard in shards:
        path = shard.get("path")
        shard_id = shard.get("id", path)
        if not path:
            print(f"⚠️ Skipping shard without path information: {shard}")
            continue

        snapshot_path = snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            allow_patterns=[f"{path}/*"],
            token=args.token,
            cache_dir=args.cache_dir,
        )
        shard_dir = Path(snapshot_path) / path

        shard_total = 0
        for example in iter_shard_examples(shard_dir):
            key = (str(example["source_dataset"]), int(example["original_index"]))
            if key in seen:
                duplicates.append((key[0], key[1], shard_id))
            else:
                seen.add(key)
            per_source[key[0]] += 1
            shard_total += 1
            total_samples += 1

        print(f"✅ Processed {shard_id}: {shard_total} samples")

    print("\n📊 Aggregate counts per source dataset:")
    for source, count in sorted(per_source.items()):
        print(f"  - {source}: {count:,}")

    print(f"\n📦 Total samples streamed: {total_samples:,}")

    if duplicates:
        print("\n❌ Duplicate (source_dataset, original_index) pairs detected:")
        for source, index_value, shard_id in duplicates[:10]:
            print(f"  - {source} #{index_value} (also seen in shard {shard_id})")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
        return 1

    print("\n✅ No duplicate pairs detected across shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
