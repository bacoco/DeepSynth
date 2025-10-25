#!/usr/bin/env python3
"""
Custom loader for reciTAL/mlsum dataset using the downloaded Google Drive data.
Works with the actual MLSUM dataset files.
"""

import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import Dict, List, Any

class MLSUMLoader:
    """Custom loader for MLSUM dataset using downloaded data files."""

    def __init__(self, data_dir="./mlsum_data/MLSUM"):
        self.data_dir = Path(data_dir)

        # Available languages in the dataset (fr, es, de order as requested, removed ru, tu)
        self.languages = ['fr', 'es', 'de']  # French, Spanish, German

        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MLSUM data directory not found: {self.data_dir}")

    def get_available_splits(self, language: str) -> List[str]:
        """Get available splits for a language."""
        lang_dir = self.data_dir / language
        if not lang_dir.exists():
            return []

        splits = []
        for split_name in ['train', 'val', 'test']:
            src_file = lang_dir / f"{split_name}.txt.src"
            tgt_file = lang_dir / f"{split_name}.txt.tgt"
            if src_file.exists() and tgt_file.exists():
                splits.append(split_name)

        return splits

    def load_split_files(self, language: str, split: str) -> Dict[str, List[str]]:
        """Load articles and summaries for a specific language and split."""
        lang_dir = self.data_dir / language

        # File paths
        articles_file = lang_dir / f"{split}.txt.src"
        summaries_file = lang_dir / f"{split}.txt.tgt"
        urls_file = lang_dir / f"{split}.txt.urls"

        # Check if files exist
        if not articles_file.exists() or not summaries_file.exists():
            raise FileNotFoundError(f"Split files not found for {language}/{split}")

        # Read articles
        with open(articles_file, 'r', encoding='utf-8') as f:
            articles = [line.strip() for line in f if line.strip()]

        # Read summaries
        with open(summaries_file, 'r', encoding='utf-8') as f:
            summaries = [line.strip() for line in f if line.strip()]

        # Read URLs if available
        urls = []
        if urls_file.exists():
            with open(urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]

        # Ensure equal lengths
        min_length = min(len(articles), len(summaries))
        articles = articles[:min_length]
        summaries = summaries[:min_length]
        urls = urls[:min_length] if urls else [''] * min_length

        return {
            'articles': articles,
            'summaries': summaries,
            'urls': urls
        }

    def load_language(self, language: str, max_samples: int = None) -> DatasetDict:
        """Load MLSUM data for a specific language."""
        if language not in self.languages:
            raise ValueError(f"Language {language} not supported. Available: {self.languages}")

        print(f"📊 Loading MLSUM {language.upper()} dataset...")

        available_splits = self.get_available_splits(language)
        if not available_splits:
            raise FileNotFoundError(f"No splits found for language {language}")

        print(f"  📂 Available splits: {available_splits}")

        dataset_dict = {}

        for split in available_splits:
            print(f"  📂 Processing {split} split...")

            try:
                # Load split data
                split_data = self.load_split_files(language, split)

                articles = split_data['articles']
                summaries = split_data['summaries']
                urls = split_data['urls']

                # Apply max_samples limit
                if max_samples and len(articles) > max_samples:
                    articles = articles[:max_samples]
                    summaries = summaries[:max_samples]
                    urls = urls[:max_samples]
                    print(f"    📊 Limited to {max_samples} samples")

                # Create dataset in HuggingFace format
                processed_data = {
                    'text': articles,
                    'summary': summaries,
                    'url': urls,
                    'topic': [''] * len(articles),  # Not available in this format
                    'title': [''] * len(articles),  # Not available in this format
                    'date': [''] * len(articles),   # Not available in this format
                }

                # Create dataset
                split_dataset = Dataset.from_dict(processed_data)

                # Map split names (val -> validation)
                hf_split_name = 'validation' if split == 'val' else split
                dataset_dict[hf_split_name] = split_dataset

                print(f"    ✅ {hf_split_name}: {len(split_dataset)} samples")

            except Exception as e:
                print(f"    ❌ Error processing {split}: {e}")
                continue

        return DatasetDict(dataset_dict)

    def test_loading(self, language: str = 'fr', max_samples: int = 5):
        """Test loading a small sample."""
        print(f"🧪 Testing MLSUM {language} loading...")

        try:
            dataset_dict = self.load_language(language, max_samples)

            print(f"✅ Test successful!")
            print(f"📊 Splits: {list(dataset_dict.keys())}")

            if 'train' in dataset_dict:
                train_data = dataset_dict['train']
                print(f"📋 Columns: {list(train_data.features.keys())}")

                if len(train_data) > 0:
                    example = train_data[0]
                    text = example['text']
                    summary = example['summary']

                    print(f"📝 Sample text length: {len(text)} chars")
                    print(f"📝 Sample summary length: {len(summary)} chars")
                    print(f"📝 Text preview: {text[:100]}...")
                    print(f"📝 Summary preview: {summary[:100]}...")

            return True

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

def main():
    """Test the custom MLSUM loader."""
    loader = MLSUMLoader()

    # Test each language
    for lang in ['fr', 'de', 'es']:
        print(f"\n{'='*60}")
        success = loader.test_loading(lang, max_samples=2)
        if success:
            print(f"✅ {lang.upper()} works!")
        else:
            print(f"❌ {lang.upper()} failed")

if __name__ == "__main__":
    main()
