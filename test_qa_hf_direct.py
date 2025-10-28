#!/usr/bin/env python3
"""Upload Q&A test DIRECT - sans transforms"""

import sys
sys.path.insert(0, "./src")

from datasets import Dataset
from deepsynth.config import Config

print("🚀 UPLOAD Q&A TEST DIRECT")
print("=" * 60)

config = Config.from_env()

# Import converters
from deepsynth.data.dataset_converters.natural_questions import (
    convert_natural_questions,
)
from deepsynth.data.dataset_converters.ms_marco import convert_ms_marco

# Collect samples - DIRECTLY without InstructionDataset wrapper
all_samples = []

print("\n📖 Natural Questions (5 samples)...")
from datasets import load_dataset as hf_load

nq_raw = hf_load("natural_questions", split="train", streaming=True)
from deepsynth.data.transforms.text_to_image import TextToImageConverter
from deepsynth.data.quality_calculator import calculate_quality, should_extract_context
from deepsynth.data.dataset_converters.natural_questions import (
    extract_contextual_window,
    extract_short_answer,
    extract_long_answer,
)

converter = TextToImageConverter(max_width=1600, max_height=10000)

count = 0
for sample in nq_raw.take(20):  # Take more to get 5 good ones
    try:
        if not sample.get("document") or not sample["document"].get("tokens"):
            continue
        tokens = sample["document"]["tokens"]
        if not isinstance(tokens, dict) or "token" not in tokens:
            continue
        text_tokens = tokens["token"]

        question = sample.get("question", {}).get("text", "")
        if not question:
            continue

        short_answer_text, short_start, short_end = extract_short_answer(sample, text_tokens)
        long_answer_text, long_start, long_end = extract_long_answer(sample, text_tokens)

        if not short_answer_text and not long_answer_text:
            continue

        if long_answer_text:
            primary_answer = long_answer_text
            answer_start = long_start
            answer_end = long_end
        else:
            primary_answer = short_answer_text
            answer_start = short_start
            answer_end = short_end

        full_document_text = " ".join(str(t) for t in text_tokens)
        full_document_token_count = len(text_tokens)

        should_extract, extraction_window = should_extract_context(
            full_document_token_count, "gundam"
        )

        if should_extract and answer_start is not None and answer_end is not None:
            document_for_image = extract_contextual_window(
                text_tokens, answer_start, answer_end,
                extraction_window, extraction_window
            )
            extracted_token_count = len(document_for_image.split())
        else:
            document_for_image = full_document_text
            extracted_token_count = full_document_token_count

        # Generate image - PIL format
        image = converter.convert(document_for_image)
        quality, quality_desc, estimated_height = calculate_quality(extracted_token_count)

        all_samples.append({
            'text': full_document_text[:1000],  # Truncate for upload
            'instruction': question,
            'answer': primary_answer[:500],  # Truncate
            'image': image,  # PIL Image
            'source': 'natural_questions',
            'quality': quality,
            'token_count': full_document_token_count,
        })

        count += 1
        print(f"  ✓ NQ sample {count}")
        if count >= 5:
            break
    except:
        continue

print(f"✅ {count} Natural Questions samples")

print("\n📚 MS MARCO (5 samples)...")
marco_raw = hf_load("ms_marco", "v2.1", split="train", streaming=True)

count2 = 0
for sample in marco_raw.take(10):
    try:
        query = sample.get("query", "")
        if not query:
            continue

        passages = sample.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        if not passage_texts:
            continue

        document = passage_texts[0]
        answers = sample.get("answers", [])
        if not answers:
            continue

        answer = answers[0]

        # Generate image
        image = converter.convert(document)
        token_count = len(document.split())
        quality, quality_desc, estimated_height = calculate_quality(token_count)

        all_samples.append({
            'text': document,
            'instruction': query,
            'answer': answer,
            'image': image,  # PIL Image
            'source': 'ms_marco',
            'quality': quality,
            'token_count': token_count,
        })

        count2 += 1
        print(f"  ✓ MARCO sample {count2}")
        if count2 >= 5:
            break
    except:
        continue

print(f"✅ {count2} MS MARCO samples")
print(f"\n📊 Total: {len(all_samples)} samples")

# Create dataset
print("\n📤 Creating & uploading dataset...")
hf_dataset = Dataset.from_list(all_samples)

REPO_ID = f"{config.hf_username}/deepsynth-qa-test"

hf_dataset.push_to_hub(
    REPO_ID,
    token=config.hf_token,
    private=False,
)

print(f"\n✅ SUCCESS!")
print(f"🔗 https://huggingface.co/datasets/{REPO_ID}")
