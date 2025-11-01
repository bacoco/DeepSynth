#!/usr/bin/env python3
"""
Custom loader for reciTAL/mlsum dataset using the downloaded Google Drive data.
Works with the actual MLSUM dataset files.
"""

import os
import requests
import zipfile
import time
try:
    import fcntl  # Unix-only
    HAS_FCNTL = True
except Exception:
    HAS_FCNTL = False
    class _FakeFcntl:
        LOCK_EX = 1
        LOCK_NB = 2
        def flock(self, *args, **kwargs):
            return None
    fcntl = _FakeFcntl()  # type: ignore
from pathlib import Path
from datasets import Dataset, DatasetDict
from typing import Dict, List, Any
from tqdm import tqdm

class MLSUMLoader:
    """Custom loader for MLSUM dataset using downloaded data files."""

    def __init__(self, data_dir="./mlsum_data"):
        self.base_dir = Path(data_dir)
        self.data_dir = self.base_dir / "MLSUM"

        # Available languages in the dataset (fr, es, de order as requested, removed ru, tu)
        self.languages = ['fr', 'es', 'de']  # French, Spanish, German

        # Auto-download if data doesn't exist (with locking for parallel workers)
        if not self.data_dir.exists():
            print(f"📥 MLSUM data not found, downloading automatically...")
            self._download_mlsum_data_with_lock()

        print(f"✅ MLSUM data ready at: {self.data_dir}")

    def _download_mlsum_data_with_lock(self):
        """Download MLSUM dataset with file locking for parallel workers."""
        lock_file = self.base_dir / ".mlsum_download.lock"

        # Create base directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)

        # Open lock file (create if doesn't exist)
        with open(lock_file, 'w') as lock_fd:
            try:
                # Try to acquire exclusive lock (non-blocking first to show status)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    print("🔒 Acquired download lock, starting download...")
                    is_downloader = True
                except BlockingIOError:
                    print("⏳ Another worker is downloading MLSUM data, waiting...")
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Block until available
                    print("✅ Download lock released by other worker")
                    is_downloader = False

                # Check if data exists now (another worker might have downloaded it)
                if self.data_dir.exists():
                    print("✅ MLSUM data is now available (downloaded by another worker)")
                    return

                # If we're the downloader or data still doesn't exist, download
                if is_downloader:
                    self._download_mlsum_data()
                else:
                    # Double-check after lock release
                    if not self.data_dir.exists():
                        print("⚠️  Data still missing after lock release, downloading...")
                        self._download_mlsum_data()

            finally:
                # Release lock
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                # Clean up lock file
                try:
                    lock_file.unlink()
                except (FileNotFoundError, PermissionError):
                    pass

    def _download_mlsum_data(self):
        """Download MLSUM dataset from Google Drive if not exists."""
        print("🌐 Downloading MLSUM dataset from Google Drive (3.2GB)...")
        print("⏳ This may take several minutes...")

        # Create base directory
        self.base_dir.mkdir(exist_ok=True)

        # Working GitLab URL for MLSUM (3.3GB)
        working_url = "https://gitlab.lip6.fr/scialom/mlsum_data/-/raw/master/MLSUM.zip"

        try:
            # Download with progress bar
            response = requests.get(working_url, stream=True)

            if response.status_code == 200:
                zip_file = self.base_dir / "mlsum.zip"

                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))

                with open(zip_file, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                print(f"✅ Downloaded to: {zip_file}")

                # Extract the zip file
                print("📦 Extracting MLSUM data...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir)

                print("✅ MLSUM data extracted successfully!")

                # Clean up zip file
                zip_file.unlink()
                print("🧹 Cleaned up zip file")

            else:
                raise Exception(f"Download failed with status: {response.status_code}")

        except Exception as e:
            print(f"❌ Auto-download failed: {e}")
            print("\n💡 Manual download required:")
            print("1. Go to: https://gitlab.lip6.fr/scialom/mlsum_data/-/raw/master/MLSUM.zip")
            print("2. Download the MLSUM.zip file (3.3GB)")
            print(f"3. Extract to {self.base_dir} directory")
            raise FileNotFoundError(f"MLSUM data not available at {self.data_dir}")

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
