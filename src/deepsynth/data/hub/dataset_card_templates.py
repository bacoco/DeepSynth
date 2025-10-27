#!/usr/bin/env python3
"""
Professional HuggingFace Dataset Card Templates for DeepSynth
"""

DATASET_CARD_TEMPLATE = """---
language:
{languages}
size_categories:
{size_categories}
task_categories:
- summarization
- image-to-text
- text-generation
tags:
- summarization
- vision
- DeepSeek-OCR
- multilingual
- multi-resolution
- visual-text-encoding
library_name: datasets
license: {license}
pretty_name: "{pretty_name}"
dataset_info:
  features:
  - name: text
    dtype: string
  - name: summary
    dtype: string
  - name: image
    dtype: image
  - name: image_tiny
    dtype: image
  - name: image_small
    dtype: image
  - name: image_base
    dtype: image
  - name: image_large
    dtype: image
  - name: image_gundam
    dtype: image
  - name: source_dataset
    dtype: string
  - name: original_split
    dtype: string
  - name: original_index
    dtype: int64
  splits:
  - name: train
    num_examples: {num_examples}
---

# {dataset_name}

## Dataset Description

{description}

This dataset is part of the **DeepSynth** project, which uses visual text encoding for multilingual summarization with the DeepSeek-OCR vision-language model. Text documents are converted into images and processed through a frozen 380M parameter visual encoder, enabling 20x token compression while preserving document layout and structure.

### Key Features

- **Multi-Resolution Images**: Each sample includes 6 different image resolutions optimized for DeepSeek-OCR training
- **Visual Text Encoding**: 20x compression ratio (1 visual token ≈ 20 text tokens)
- **Document Structure Preservation**: Layout and formatting maintained through image representation
- **Human-Written Summaries**: High-quality reference summaries for each document
- **Deduplication Tracking**: Source dataset and index tracking prevents duplicates

### Dataset Statistics

- **Total Samples**: ~{total_samples}
- **Language(s)**: {language_names}
- **Domain**: {domain}
- **Average Document Length**: {avg_doc_length}
- **Average Summary Length**: {avg_summary_length}

### Source Dataset

{source_info}

## Multi-Resolution Image Format

Each sample contains **6 image representations** of the same text document at different resolutions:

| Field Name | Resolution | Use Case |
|------------|-----------|----------|
| `image` | Original (1600×2200 max) | Full document, highest quality |
| `image_tiny` | 512×512 | Fast training, mobile inference |
| `image_small` | 640×640 | Balanced speed/quality |
| `image_base` | 1024×1024 | Standard DeepSeek-OCR training |
| `image_large` | 1280×1280 | High-quality inference |
| `image_gundam` | 1600×1600 | Maximum quality, research |

All images use aspect-ratio preservation with padding to maintain text readability.

## Dataset Structure

### Data Fields

- `text` (string): Original document text
- `summary` (string): Human-written summary
- `image` (PIL.Image): Original full-size rendered document image
- `image_tiny` (PIL.Image): 512×512 resolution
- `image_small` (PIL.Image): 640×640 resolution
- `image_base` (PIL.Image): 1024×1024 resolution
- `image_large` (PIL.Image): 1280×1280 resolution
- `image_gundam` (PIL.Image): 1600×1600 resolution
- `source_dataset` (string): Origin dataset name
- `original_split` (string): Source split (train/validation/test)
- `original_index` (int): Original sample index for deduplication

### Data Example

```python
{{
    'text': '{example_text}',
    'summary': '{example_summary}',
    'image': <PIL.Image>,
    'image_tiny': <PIL.Image (512×512)>,
    'image_small': <PIL.Image (640×640)>,
    'image_base': <PIL.Image (1024×1024)>,
    'image_large': <PIL.Image (1280×1280)>,
    'image_gundam': <PIL.Image (1600×1600)>,
    'source_dataset': '{source_dataset}',
    'original_split': 'train',
    'original_index': 0
}}
```

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("{repo_id}")

# Load specific resolution only (faster, less memory)
dataset = load_dataset("{repo_id}", columns=['text', 'summary', 'image_base'])

# Streaming for large datasets
dataset = load_dataset("{repo_id}", streaming=True)
```

### Training Example with DeepSeek-OCR

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained("deepseek-ai/DeepSeek-OCR")
processor = AutoProcessor.from_pretrained("deepseek-ai/DeepSeek-OCR")

# Load dataset (use appropriate resolution)
dataset = load_dataset("{repo_id}")

# Process sample
sample = dataset['train'][0]
inputs = processor(
    images=sample['image_base'],  # Use base resolution for training
    text=sample['text'],
    return_tensors="pt"
)

# Fine-tune decoder only (freeze encoder)
for param in model.encoder.parameters():
    param.requires_grad = False

# Training loop...
```

### Resolution Selection Guidelines

| Training Stage | Recommended Resolution | Rationale |
|----------------|----------------------|-----------|
| Initial training | `image_small` (640×640) | Fast iteration, baseline |
| Standard training | `image_base` (1024×1024) | Best speed/quality balance |
| High-quality training | `image_large` (1280×1280) | Better detail capture |
| Research/SOTA | `image_gundam` (1600×1600) | Maximum quality |
| Mobile deployment | `image_tiny` (512×512) | Inference speed |

## Training Recommendations

### DeepSeek-OCR Fine-Tuning

```python
# Recommended hyperparameters
training_args = {{
    "learning_rate": 2e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "mixed_precision": "bf16",
    "freeze_encoder": True,  # IMPORTANT: Only fine-tune decoder
    "image_resolution": "base"  # Use image_base field
}}
```

### Expected Performance

- **Baseline (text-to-text)**: ROUGE-1 ~40-42
- **DeepSeek-OCR (visual)**: ROUGE-1 ~44-47 (typical SOTA)
- **Training Time**: ~6-8 hours on A100 (80GB) for full dataset
- **GPU Memory**: ~40GB with batch_size=4, mixed_precision=bf16

## Dataset Creation

This dataset was created using the **DeepSynth** pipeline:

1. **Source Loading**: Original text documents from {source_dataset}
2. **Text-to-Image Conversion**: Documents rendered as PNG images (DejaVu Sans 12pt, Unicode support)
3. **Multi-Resolution Generation**: 6 resolutions generated with aspect-ratio preservation
4. **Incremental Upload**: Batches of 5,000 samples uploaded to HuggingFace Hub
5. **Deduplication**: Source tracking prevents duplicate samples

### Rendering Specifications

- **Font**: DejaVu Sans 12pt (full Unicode support for multilingual text)
- **Line Wrapping**: 100 characters per line
- **Margin**: 40px
- **Background**: White (255, 255, 255)
- **Text Color**: Black (0, 0, 0)
- **Format**: PNG with lossless compression

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{deepsynth-{dataset_id},
    title={{{{DeepSynth {pretty_name}: Multi-Resolution Visual Text Encoding for Summarization}}}},
    author={{Baconnier}},
    year={{2025}},
    publisher={{HuggingFace}},
    howpublished={{\\url{{{repo_url}}}}}
}}
```

### Source Dataset Citation

{source_citation}

## License

{license_text}

**Note**: This dataset inherits the license from the original source dataset. Please review the source license before commercial use.

## Limitations and Bias

{limitations}

## Additional Information

### Dataset Curators

Created by the DeepSynth team as part of multilingual visual summarization research.

### Contact

- **Repository**: [DeepSynth GitHub](https://github.com/bacoco/DeepSynth)
- **Issues**: [GitHub Issues](https://github.com/bacoco/DeepSynth/issues)

### Acknowledgments

- **DeepSeek-OCR**: Visual encoder from DeepSeek AI
- **Source Dataset**: {source_dataset}
- **HuggingFace**: Dataset hosting and infrastructure
"""

# Dataset-specific configurations
DATASET_CONFIGS = {
    'deepsynth-en-news': {
        'pretty_name': 'CNN/DailyMail News Summarization',
        'languages': '- en',
        'language_names': 'English',
        'size_categories': '- 100K<n<1M',
        'domain': 'News articles',
        'total_samples': '287,000',
        'avg_doc_length': '~800 tokens',
        'avg_summary_length': '~60 tokens',
        'source_dataset': 'cnn_dailymail',
        'description': '''A large-scale dataset of CNN and Daily Mail news articles paired with multi-sentence summaries.
This visual encoding version enables training DeepSeek-OCR models for news summarization with document layout awareness.''',
        'source_info': '''Based on the **CNN/DailyMail dataset** (version 3.0.0), containing news articles from CNN and Daily Mail newspapers.
- **Original Authors**: Hermann et al. (2015), See et al. (2017)
- **Paper**: [Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340)
- **License**: MIT License''',
        'source_citation': '''```bibtex
@article{hermann2015teaching,
    title={Teaching machines to read and comprehend},
    author={Hermann, Karl Moritz and Kocisky, Tomas and Grefenstette, Edward and Espeholt, Lasse and Kay, Will and Suleyman, Mustafa and Blunsom, Phil},
    journal={Advances in neural information processing systems},
    volume={28},
    year={2015}
}
```''',
        'example_text': 'London (CNN) -- A British woman accused of killing her...',
        'example_summary': 'British woman charged with murder in Dubai...',
        'limitations': '''- **Domain-specific**: Optimized for news articles; may not generalize to other domains
- **English-only**: Limited to English news articles
- **Temporal bias**: Articles from 2007-2015; may contain outdated information
- **Geographic bias**: Primarily US/UK news sources''',
        'license': 'mit',
        'license_text': 'MIT License - See source dataset for full license terms.'
    },
    'deepsynth-en-arxiv': {
        'pretty_name': 'arXiv Scientific Paper Summarization',
        'languages': '- en',
        'language_names': 'English',
        'size_categories': '- 10K<n<100K',
        'domain': 'Scientific papers (Computer Science, Physics, Mathematics)',
        'total_samples': '50,000',
        'avg_doc_length': '~5,000 tokens',
        'avg_summary_length': '~150 tokens',
        'source_dataset': 'ccdv/arxiv-summarization',
        'description': '''Scientific paper abstracts from arXiv, covering computer science, physics, and mathematics.
Visual encoding preserves mathematical notation and document structure critical for scientific summarization.''',
        'source_info': '''Based on the **arXiv dataset** from arXiv.org e-prints.
- **Papers**: Computer Science, Physics, Mathematics domains
- **Time Period**: 2007-2021
- **License**: arXiv Non-exclusive distribution license''',
        'source_citation': '''```bibtex
@article{cohan2018discourse,
    title={A discourse-aware attention model for abstractive summarization of long documents},
    author={Cohan, Arman and Dernoncourt, Franck and Kim, Doo Soon and Bui, Trung and Kim, Seokhwan and Chang, Walter and Goharian, Nazli},
    journal={arXiv preprint arXiv:1804.05685},
    year={2018}
}
```''',
        'example_text': 'We present a novel approach to neural machine translation...',
        'example_summary': 'This paper introduces an attention-based NMT model...',
        'limitations': '''- **Scientific jargon**: Heavy use of technical terminology
- **Mathematical notation**: LaTeX rendering may affect OCR accuracy
- **Domain-specific**: Optimized for CS/Physics/Math papers
- **Length**: Very long documents (up to 10,000+ tokens) may be truncated''',
        'license': 'cc-by-4.0',
        'license_text': 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
    },
    'deepsynth-en-xsum': {
        'pretty_name': 'XSum BBC News Summarization',
        'languages': '- en',
        'language_names': 'English',
        'size_categories': '- 10K<n<100K',
        'domain': 'BBC news articles',
        'total_samples': '50,000',
        'avg_doc_length': '~400 tokens',
        'avg_summary_length': '~20 tokens (single sentence)',
        'source_dataset': 'Rexhaif/xsum_reduced',
        'description': '''BBC news articles with single-sentence summaries. Focused on extreme summarization where the summary is
a single sentence capturing the essence of the article.''',
        'source_info': '''Based on the **XSum dataset** from BBC articles (2010-2017).
- **Original Authors**: Narayan et al. (2018)
- **Paper**: [Don't Give Me the Details, Just the Summary!](https://arxiv.org/abs/1808.08745)
- **License**: MIT License''',
        'source_citation': '''```bibtex
@inproceedings{narayan2018don,
    title={Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization},
    author={Narayan, Shashi and Cohen, Shay B and Lapata, Mirella},
    booktitle={Proceedings of EMNLP},
    year={2018}
}
```''',
        'example_text': 'The government has announced new measures to...',
        'example_summary': 'Government unveils climate change plan.',
        'limitations': '''- **Extreme summarization**: Single-sentence summaries may lose important details
- **UK-centric**: Primarily British news and perspectives
- **Short summaries**: Not suitable for multi-sentence summary training
- **Temporal bias**: Articles from 2010-2017''',
        'license': 'mit',
        'license_text': 'MIT License - See source dataset for full license terms.'
    },
    'deepsynth-fr': {
        'pretty_name': 'MLSUM French News Summarization',
        'languages': '- fr',
        'language_names': 'French',
        'size_categories': '- 100K<n<1M',
        'domain': 'French news articles',
        'total_samples': '392,000',
        'avg_doc_length': '~700 tokens',
        'avg_summary_length': '~50 tokens',
        'source_dataset': 'MLSUM (fr)',
        'description': '''Large-scale French news summarization dataset from major French newspapers.
Enables training multilingual DeepSeek-OCR models with proper Unicode/diacritics handling.''',
        'source_info': '''Based on **MLSUM (MultiLingual SUMmarization)** French subset.
- **Original Authors**: Scialom et al. (2020)
- **Paper**: [MLSUM: The Multilingual Summarization Corpus](https://arxiv.org/abs/2004.14900)
- **License**: CC BY-NC-SA 4.0''',
        'source_citation': '''```bibtex
@inproceedings{scialom2020mlsum,
    title={MLSUM: The Multilingual Summarization Corpus},
    author={Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
    booktitle={Proceedings of EMNLP},
    year={2020}
}
```''',
        'example_text': 'Le gouvernement français a annoncé de nouvelles mesures...',
        'example_summary': 'Nouvelles mesures gouvernementales contre le changement climatique.',
        'limitations': '''- **French-specific**: Requires French language models
- **Diacritics**: Proper handling of accents (é, è, ê, etc.) critical
- **Regional**: May contain France-specific cultural references
- **News domain**: Limited to journalistic style''',
        'license': 'cc-by-nc-sa-4.0',
        'license_text': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
    },
    'deepsynth-es': {
        'pretty_name': 'MLSUM Spanish News Summarization',
        'languages': '- es',
        'language_names': 'Spanish',
        'size_categories': '- 100K<n<1M',
        'domain': 'Spanish news articles',
        'total_samples': '266,000',
        'avg_doc_length': '~650 tokens',
        'avg_summary_length': '~45 tokens',
        'source_dataset': 'MLSUM (es)',
        'description': '''Large-scale Spanish news summarization dataset from major Spanish newspapers.
Covers European and Latin American Spanish with proper Unicode character support.''',
        'source_info': '''Based on **MLSUM (MultiLingual SUMmarization)** Spanish subset.
- **Original Authors**: Scialom et al. (2020)
- **Paper**: [MLSUM: The Multilingual Summarization Corpus](https://arxiv.org/abs/2004.14900)
- **License**: CC BY-NC-SA 4.0''',
        'source_citation': '''```bibtex
@inproceedings{scialom2020mlsum,
    title={MLSUM: The Multilingual Summarization Corpus},
    author={Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
    booktitle={Proceedings of EMNLP},
    year={2020}
}
```''',
        'example_text': 'El gobierno español ha anunciado nuevas medidas...',
        'example_summary': 'Nuevas medidas gubernamentales contra el cambio climático.',
        'limitations': '''- **Spanish-specific**: Requires Spanish language models
- **Diacritics**: Proper handling of tildes (ñ) and accents (á, é, í, ó, ú)
- **Regional variants**: Mix of European and Latin American Spanish
- **News domain**: Limited to journalistic style''',
        'license': 'cc-by-nc-sa-4.0',
        'license_text': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
    },
    'deepsynth-de': {
        'pretty_name': 'MLSUM German News Summarization',
        'languages': '- de',
        'language_names': 'German',
        'size_categories': '- 100K<n<1M',
        'domain': 'German news articles',
        'total_samples': '220,000',
        'avg_doc_length': '~600 tokens',
        'avg_summary_length': '~40 tokens',
        'source_dataset': 'MLSUM (de)',
        'description': '''Large-scale German news summarization dataset from major German newspapers.
Handles German-specific characters (umlauts, ß) through proper Unicode font rendering.''',
        'source_info': '''Based on **MLSUM (MultiLingual SUMmarization)** German subset.
- **Original Authors**: Scialom et al. (2020)
- **Paper**: [MLSUM: The Multilingual Summarization Corpus](https://arxiv.org/abs/2004.14900)
- **License**: CC BY-NC-SA 4.0''',
        'source_citation': '''```bibtex
@inproceedings{scialom2020mlsum,
    title={MLSUM: The Multilingual Summarization Corpus},
    author={Scialom, Thomas and Dray, Paul-Alexis and Lamprier, Sylvain and Piwowarski, Benjamin and Staiano, Jacopo},
    booktitle={Proceedings of EMNLP},
    year={2020}
}
```''',
        'example_text': 'Die deutsche Regierung hat neue Maßnahmen angekündigt...',
        'example_summary': 'Neue Regierungsmaßnahmen gegen den Klimawandel.',
        'limitations': '''- **German-specific**: Requires German language models
- **Umlauts**: Proper rendering of ä, ö, ü, ß critical for OCR
- **Compound words**: Very long German compound words may affect tokenization
- **News domain**: Limited to journalistic style''',
        'license': 'cc-by-nc-sa-4.0',
        'license_text': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
    },
    'deepsynth-en-legal': {
        'pretty_name': 'BillSum Legal Document Summarization',
        'languages': '- en',
        'language_names': 'English',
        'size_categories': '- 10K<n<100K',
        'domain': 'US Congressional bills',
        'total_samples': '22,000',
        'avg_doc_length': '~3,000 tokens',
        'avg_summary_length': '~200 tokens',
        'source_dataset': 'billsum',
        'description': '''US Congressional bills with human-written summaries. Specialized for legal document summarization
with complex structure, formal language, and legislative terminology.''',
        'source_info': '''Based on the **BillSum dataset** of US Congressional bills.
- **Original Authors**: Kornilova & Eidelman (2019)
- **Paper**: [BillSum: A Corpus for Automatic Summarization of US Legislation](https://arxiv.org/abs/1910.00523)
- **License**: CC0 1.0 Universal (Public Domain)''',
        'source_citation': '''```bibtex
@inproceedings{kornilova2019billsum,
    title={BillSum: A Corpus for Automatic Summarization of US Legislation},
    author={Kornilova, Anastassia and Eidelman, Vladimir},
    booktitle={Proceedings of the 2nd Workshop on New Frontiers in Summarization},
    year={2019}
}
```''',
        'example_text': 'A BILL to amend the Internal Revenue Code of 1986...',
        'example_summary': 'This bill amends the Internal Revenue Code to...',
        'limitations': '''- **Legal jargon**: Heavy use of legislative and legal terminology
- **Complex structure**: Bills have nested sections, subsections, clauses
- **US-specific**: United States federal legislation only
- **Formal language**: Very different from conversational or news text
- **Long documents**: Bills can be 10,000+ tokens''',
        'license': 'cc0-1.0',
        'license_text': 'CC0 1.0 Universal (Public Domain Dedication)'
    }
}


def generate_dataset_card(dataset_name: str, repo_id: str, num_examples: int = None) -> str:
    """
    Generate a comprehensive HuggingFace dataset card.

    Args:
        dataset_name: Name of the dataset (e.g., 'deepsynth-en-news')
        repo_id: Full HuggingFace repo ID (e.g., 'baconnier/deepsynth-en-news')
        num_examples: Actual number of examples in the dataset (optional)

    Returns:
        Complete README.md content as string
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_name]
    dataset_id = dataset_name.replace('deepsynth-', '')

    # Use actual count if provided, otherwise use estimated
    if num_examples is None:
        num_examples = config['total_samples'].replace(',', '')
        if num_examples.isdigit():
            num_examples = int(num_examples)
        else:
            num_examples = 0  # Placeholder

    # Prepare format kwargs
    format_kwargs = {
        'dataset_name': f"DeepSynth - {config['pretty_name']}",
        'dataset_id': dataset_id,
        'repo_id': repo_id,
        'repo_url': f"https://huggingface.co/datasets/{repo_id}",
        'num_examples': num_examples,
    }
    # Merge with config (config values take precedence for any conflicts)
    format_kwargs.update(config)

    return DATASET_CARD_TEMPLATE.format(**format_kwargs)
