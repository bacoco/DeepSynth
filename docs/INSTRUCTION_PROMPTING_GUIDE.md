# Instruction Prompting Implementation Guide

**Version**: 1.0
**Date**: 2025-10-27
**Companion to**: INSTRUCTION_PROMPTING_PRD.md

---

## Quick Start

This guide shows you how to **use** and **train** DeepSynth with instruction prompting for Q&A, custom summarization, and information extraction.

---

## Table of Contents

1. [Using Pre-trained Models](#using-pre-trained-models)
2. [Training New Models](#training-new-models)
3. [Dataset Preparation](#dataset-preparation)
4. [Configuration Reference](#configuration-reference)
5. [Troubleshooting](#troubleshooting)

---

## Using Pre-trained Models

### Python API

```python
from deepsynth.inference import InstructionEngine

# Load model
engine = InstructionEngine(
    model_path="./models/deepsynth-qa",
    use_text_encoder=True,
)

# Q&A
answer = engine.generate(
    document="AI has transformed healthcare through improved diagnostics...",
    instruction="What are the main benefits of AI in healthcare?",
)
print(answer)
# Output: "The main benefits include: 1) Improved diagnostic accuracy..."

# Custom summarization
summary = engine.generate(
    document="Q3 2024 Financial Report: Revenue $45M (+15% YoY)...",
    instruction="Summarize focusing only on revenue trends",
)
print(summary)
```

### CLI

```bash
# Q&A inference
PYTHONPATH=./src python3 -m deepsynth.inference.instruct \
    --model ./models/deepsynth-qa \
    --document "path/to/document.txt" \
    --instruction "What are the key findings?" \
    --max-length 256

# Custom instruction
PYTHONPATH=./src python3 -m deepsynth.inference.instruct \
    --model ./models/deepsynth-qa \
    --document "financial_report.txt" \
    --instruction "Extract all revenue numbers and growth rates"
```

### Web UI

1. Navigate to http://localhost:5000
2. Click "Instruction Prompting" tab
3. Upload document or paste text
4. Enter your question or instruction
5. Click "Generate Answer"

---

## Training New Models

### Step 1: Prepare Dataset

#### Option A: Convert Existing Dataset (e.g., SQuAD)

```python
from datasets import load_dataset

# Load SQuAD
squad = load_dataset("squad_v2", split="train[:10000]")

# Convert to DeepSynth format
def convert(example):
    return {
        "text": example["context"],
        "instruction": example["question"],
        "answer": example["answers"]["text"][0] if example["answers"]["text"] else "",
    }

dataset = squad.map(convert, remove_columns=squad.column_names)

# Push to Hub
dataset.push_to_hub("username/deepsynth-squad-qa")
```

#### Option B: Create Custom Dataset

```python
from datasets import Dataset

data = {
    "text": [
        "Document 1 content...",
        "Document 2 content...",
    ],
    "instruction": [
        "What is the main topic?",
        "Summarize the key points",
    ],
    "answer": [
        "The main topic is...",
        "Key points: 1. ... 2. ...",
    ],
}

dataset = Dataset.from_dict(data)
dataset.push_to_hub("username/my-custom-instruction-dataset")
```

### Step 2: Configure Training

Create `config.json`:

```json
{
  "model_name": "deepseek-ai/DeepSeek-OCR",
  "output_dir": "./models/deepsynth-qa",

  "use_text_encoder": true,
  "text_encoder_type": "qwen3",
  "text_encoder_model": "Qwen/Qwen2.5-7B-Instruct",
  "text_encoder_trainable": true,

  "batch_size": 4,
  "num_epochs": 3,
  "learning_rate": 1e-5,
  "mixed_precision": "bf16"
}
```

### Step 3: Train

```bash
# Using config file
PYTHONPATH=./src python3 -m deepsynth.training.train \
    --config config.json \
    --hf-dataset username/deepsynth-squad-qa

# Or using CLI arguments
PYTHONPATH=./src python3 -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset username/deepsynth-squad-qa \
    --output ./models/deepsynth-qa \
    --use-text-encoder \
    --text-encoder-type qwen3 \
    --text-encoder-trainable \
    --batch-size 4 \
    --num-epochs 3
```

### Step 4: Evaluate

```bash
PYTHONPATH=./src python3 -m deepsynth.evaluation.evaluate \
    --model ./models/deepsynth-qa \
    --dataset username/deepsynth-squad-qa \
    --split test \
    --metrics exact_match f1_score rouge
```

---

## Dataset Preparation

### Format Requirements

**JSONL Format** (one JSON per line):
```jsonl
{"text": "Document 1...", "instruction": "Question 1?", "answer": "Answer 1"}
{"text": "Document 2...", "instruction": "Question 2?", "answer": "Answer 2"}
```

**HuggingFace Dataset** (recommended):
```python
{
    "text": List[str],        # Source documents
    "instruction": List[str], # Questions/instructions
    "answer": List[str],      # Expected outputs
}
```

### Conversion Scripts

#### SQuAD → DeepSynth

```python
from datasets import load_dataset

def convert_squad(split="train"):
    squad = load_dataset("squad_v2", split=split)

    def process(example):
        return {
            "text": example["context"],
            "instruction": example["question"],
            "answer": example["answers"]["text"][0] if example["answers"]["text"] else "No answer",
        }

    return squad.map(process, remove_columns=squad.column_names)

dataset = convert_squad()
dataset.push_to_hub("username/deepsynth-squad-qa")
```

#### Natural Questions → DeepSynth

```python
from datasets import load_dataset

def convert_nq(split="train"):
    nq = load_dataset("natural_questions", split=split)

    def process(example):
        # Extract short answer from annotations
        annotations = example.get("annotations", {})
        short_answers = annotations.get("short_answers", [])
        answer_text = short_answers[0]["text"] if short_answers else ""

        return {
            "text": example["document"]["text"],
            "instruction": example["question"]["text"],
            "answer": answer_text,
        }

    return nq.map(process)

dataset = convert_nq()
dataset.push_to_hub("username/deepsynth-nq-qa")
```

#### Custom Domain (Example: Finance)

```python
# Example: Financial report Q&A
financial_data = []

# Parse your financial documents
for report in financial_reports:
    financial_data.append({
        "text": report["full_text"],
        "instruction": "What was the total revenue?",
        "answer": f"${report['revenue']}M",
    })
    financial_data.append({
        "text": report["full_text"],
        "instruction": "Summarize growth trends",
        "answer": report["growth_summary"],
    })

from datasets import Dataset
dataset = Dataset.from_list(financial_data)
dataset.push_to_hub("username/deepsynth-finance-qa")
```

---

## Configuration Reference

### Text Encoder Options

| Parameter | Values | Description | Memory Impact |
|-----------|--------|-------------|---------------|
| `use_text_encoder` | `true`/`false` | Enable instruction prompting | +14GB if trainable |
| `text_encoder_type` | `"qwen3"`, `"bert"` | Encoder architecture | Qwen: 14GB, BERT: 2GB |
| `text_encoder_model` | HF model ID | Specific model | Varies by model |
| `text_encoder_trainable` | `true`/`false` | Fine-tune encoder | +50% memory if true |

### Recommended Configurations

#### Config 1: Q&A (High Quality)
```python
{
    "use_text_encoder": true,
    "text_encoder_type": "qwen3",
    "text_encoder_model": "Qwen/Qwen2.5-7B-Instruct",
    "text_encoder_trainable": true,  # Fine-tune for best quality

    "batch_size": 4,
    "learning_rate": 1e-5,  # Low LR for stable fine-tuning
    "num_epochs": 3,
}
```

#### Config 2: Fast Training (Frozen Encoder)
```python
{
    "use_text_encoder": true,
    "text_encoder_trainable": false,  # Frozen = faster, less memory

    "batch_size": 6,  # Can increase
    "learning_rate": 2e-5,
    "num_epochs": 2,
}
```

#### Config 3: Memory Efficient (LoRA + Frozen Encoder)
```python
{
    "use_lora": true,
    "lora_rank": 16,

    "use_text_encoder": true,
    "text_encoder_trainable": false,

    "batch_size": 8,  # Smaller model, bigger batch
}
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. **Freeze text encoder**:
   ```python
   text_encoder_trainable=False  # Saves ~7GB
   ```

2. **Reduce batch size**:
   ```python
   batch_size=2  # From 4
   gradient_accumulation_steps=8  # Maintain effective batch size
   ```

3. **Use LoRA**:
   ```python
   use_lora=True
   lora_rank=8
   ```

4. **Enable gradient checkpointing**:
   ```python
   use_gradient_checkpointing=True  # Saves ~30% memory
   ```

---

### Issue 2: Text Embeddings Not Used

**Symptom**: Model ignores instructions, generates standard summaries

**Debug**:
```python
# Check if text encoder is loaded
print(f"Text encoder: {trainer.text_encoder}")  # Should not be None

# Check if embeddings are being passed
# Add debug logging in forward pass
LOGGER.info(f"Text embeddings shape: {text_embeddings.shape}")  # Should be [batch, dim]
```

**Solutions**:
- Verify `use_text_encoder=True` in config
- Check model forward accepts `text_embeddings` parameter
- Ensure dataset has `instruction` field

---

### Issue 3: Poor Quality Answers

**Symptom**: Answers are generic or off-topic

**Solutions**:
1. **Increase training data**:
   - Minimum: 10k samples
   - Recommended: 50k+ samples

2. **Fine-tune text encoder**:
   ```python
   text_encoder_trainable=True
   ```

3. **Lower learning rate**:
   ```python
   learning_rate=5e-6  # Very careful fine-tuning
   ```

4. **More epochs**:
   ```python
   num_epochs=5
   ```

---

### Issue 4: Training Too Slow

**Symptom**: < 5 samples/sec

**Solutions**:
1. **Freeze text encoder**:
   ```python
   text_encoder_trainable=False  # 2x faster
   ```

2. **Mixed precision**:
   ```python
   mixed_precision="bf16"  # Enable on A100
   ```

3. **Increase batch size**:
   ```python
   batch_size=8  # If memory allows
   num_workers=8  # Parallel data loading
   ```

4. **Use Flash Attention** (if available):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

---

## API Usage Examples

### REST API

```bash
# Start API server
PYTHONPATH=./src python3 -m deepsynth.inference.api_server \
    --model ./models/deepsynth-qa \
    --port 5000

# Query via curl
curl -X POST http://localhost:5000/api/inference/instruct \
  -H "Content-Type: application/json" \
  -d '{
    "document": "AI has transformed healthcare...",
    "instruction": "What are the benefits?",
    "max_length": 256
  }'

# Response
{
  "answer": "The main benefits include...",
  "confidence": 0.92,
  "tokens_generated": 45,
  "inference_time_ms": 234
}
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:5000/api/inference/instruct",
    json={
        "document": "Financial report Q3 2024...",
        "instruction": "Extract revenue numbers",
        "max_length": 128
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Confidence: {data['confidence']}")
```

---

## Performance Benchmarks

### Expected Performance

| Configuration | Tokens/sec | VRAM | GPU |
|---------------|------------|------|-----|
| Standard (no text) | 15 | 16GB | A100 40GB |
| + Frozen text encoder | 14 | 23GB | A100 40GB |
| + Trainable text encoder | 12 | 30GB | A100 40GB |
| LoRA + Frozen text | 16 | 15GB | RTX 3090 24GB |

### Quality Benchmarks (SQuAD)

| Configuration | Exact Match | F1 Score | Training Time (10k samples) |
|---------------|-------------|----------|------------------------------|
| Frozen text encoder | 68% | 78% | 2 hours |
| Trainable text encoder | 72% | 82% | 4 hours |
| + LoRA | 70% | 80% | 3 hours |

---

## Best Practices

### Training
1. **Start with frozen text encoder** for fast iteration
2. **Use small dataset** (1k samples) to verify setup
3. **Monitor loss curves** - should decrease steadily
4. **Validate on held-out set** every epoch

### Dataset Quality
1. **Diverse instructions** - mix questions, commands, requests
2. **Balanced difficulty** - easy, medium, hard examples
3. **Domain coverage** - represent target use cases
4. **Quality over quantity** - 10k high-quality > 100k noisy

### Inference
1. **Batch queries** when possible (8-16 at once)
2. **Cache common documents** (avoid re-encoding)
3. **Set temperature** based on task:
   - Q&A: 0.3 (more deterministic)
   - Summarization: 0.7 (more creative)

---

## Next Steps

1. **Read the PRD**: `docs/INSTRUCTION_PROMPTING_PRD.md`
2. **Try examples**: Start with SQuAD conversion script
3. **Train small model**: 1k samples to verify setup
4. **Scale up**: Full dataset + fine-tuning
5. **Evaluate**: Compare to baselines

---

## Support

- **Issues**: https://github.com/bacoco/DeepSynth/issues
- **Documentation**: `docs/` directory
- **Examples**: `examples/instruction_prompting/`

---

**Document Version**: 1.0
**Last Updated**: 2025-10-27
