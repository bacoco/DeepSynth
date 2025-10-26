"""
XLSum Dataset Loader - Direct Download from GitHub
Downloads JSONL files directly from the original XLSum repository to bypass HuggingFace script issues.
"""

import os
import json
import requests
from typing import Iterator, Dict, Any
from tqdm import tqdm
import gzip

class XLSumLoader:
    """Loads XLSum dataset directly from GitHub repository JSONL files."""
    
    # Direct download URLs from XLSum GitHub repository
    DOWNLOAD_URLS = {
        'english': {
            'train': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/english_train.jsonl',
            'validation': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/english_val.jsonl', 
            'test': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/english_test.jsonl'
        },
        'chinese_simplified': {
            'train': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/chinese_simplified_train.jsonl',
            'validation': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/chinese_simplified_val.jsonl',
            'test': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/chinese_simplified_test.jsonl'
        },
        'chinese_traditional': {
            'train': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/chinese_traditional_train.jsonl',
            'validation': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/chinese_traditional_val.jsonl',
            'test': 'https://github.com/csebuetnlp/xl-sum/raw/master/data/chinese_traditional_test.jsonl'
        }
    }
    
    def __init__(self, cache_dir: str = "./xlsum_cache"):
        """Initialize XLSum loader with cache directory."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> str:
        """Download a file from URL to cache directory."""
        filepath = os.path.join(self.cache_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"✅ Using cached file: {filename}")
            return filepath
        
        print(f"📥 Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            raise
    
    def load_jsonl(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Load data from JSONL file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
                            continue
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            raise
    
    def load_language(self, language: str, split: str = 'train') -> Iterator[Dict[str, Any]]:
        """
        Load XLSum data for a specific language and split.
        
        Args:
            language: Language code ('english', 'chinese_simplified', 'chinese_traditional')
            split: Data split ('train', 'validation', 'test')
        
        Yields:
            Dict with keys: id, url, title, summary, text
        """
        if language not in self.DOWNLOAD_URLS:
            available = list(self.DOWNLOAD_URLS.keys())
            raise ValueError(f"Language '{language}' not available. Available: {available}")
        
        if split not in self.DOWNLOAD_URLS[language]:
            available = list(self.DOWNLOAD_URLS[language].keys())
            raise ValueError(f"Split '{split}' not available. Available: {available}")
        
        # Download file
        url = self.DOWNLOAD_URLS[language][split]
        filename = f"xlsum_{language}_{split}.jsonl"
        filepath = self.download_file(url, filename)
        
        # Load and yield data
        print(f"📖 Loading {language} {split} data...")
        count = 0
        for example in self.load_jsonl(filepath):
            # Standardize field names
            yield {
                'id': example.get('id', ''),
                'url': example.get('url', ''),
                'title': example.get('title', ''),
                'summary': example.get('summary', ''),
                'text': example.get('text', ''),
                'language': language
            }
            count += 1
        
        print(f"✅ Loaded {count} examples from {language} {split}")
    
    def get_dataset_info(self, language: str) -> Dict[str, int]:
        """Get dataset size information for a language."""
        info = {}
        for split in ['train', 'validation', 'test']:
            try:
                count = sum(1 for _ in self.load_language(language, split))
                info[split] = count
            except Exception as e:
                print(f"⚠️ Could not get info for {language} {split}: {e}")
                info[split] = 0
        return info


def test_xlsum_loader():
    """Test the XLSum loader."""
    print("🧪 Testing XLSum Loader...")
    
    loader = XLSumLoader()
    
    # Test each language
    for language in ['english', 'chinese_simplified']:
        try:
            print(f"\n📊 Testing {language}...")
            
            # Load a few examples
            examples = list(loader.load_language(language, 'train'))[:3]
            
            if examples:
                print(f"✅ Successfully loaded {len(examples)} examples")
                
                # Show first example
                example = examples[0]
                print(f"📝 Sample example:")
                print(f"  ID: {example['id']}")
                print(f"  Title: {example['title'][:100]}...")
                print(f"  Text length: {len(example['text'])} chars")
                print(f"  Summary length: {len(example['summary'])} chars")
                print(f"  Text preview: {example['text'][:150]}...")
                print(f"  Summary preview: {example['summary'][:150]}...")
                print(f"🎯 {language.upper()} WORKS!")
            else:
                print(f"❌ No examples loaded for {language}")
                
        except Exception as e:
            print(f"❌ {language} failed: {e}")


if __name__ == "__main__":
    test_xlsum_loader()