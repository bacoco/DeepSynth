# PRD: Instruction Prompting Feature for DeepSynth

**Version**: 1.0
**Date**: 2025-10-27
**Status**: Implementation Ready
**Author**: DeepSynth Team

---

## Executive Summary

This PRD defines the **Instruction Prompting** feature for DeepSynth, enabling the model to:
- Answer questions about documents (Q&A)
- Follow custom summarization instructions
- Extract specific information based on queries

The feature uses a **Qwen text encoder** (4096-dim) to encode instructions/queries separately, then concatenates them with vision tokens for processing by the MoE decoder.

**Key Benefit**: Optional text encoder allows both standard summarization AND instruction-following capabilities with minimal architectural changes.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture](#architecture)
3. [Use Cases & Examples](#use-cases--examples)
4. [Dataset Format](#dataset-format)
5. [Configuration](#configuration)
6. [Training Process](#training-process)
7. [Inference API](#inference-api)
8. [UI Integration](#ui-integration)
9. [Performance Considerations](#performance-considerations)
10. [Success Metrics](#success-metrics)

---

## Problem Statement

### Current Limitation

DeepSynth currently supports **only** standard summarization:
```
Input: Document (as image)
Output: Summary
```

Users cannot:
- Ask specific questions about documents
- Request summaries with custom focus (e.g., "financial aspects only")
- Extract targeted information (e.g., "list all dates and names")

### Proposed Solution

Add **optional instruction prompting** via a text encoder:
```
Input: Document (as image) + Instruction/Query (as text)
Output: Targeted response
```

**Examples**:
- "What are the key findings?" → "The study found that..."
- "Summarize financial trends" → "Revenue increased 15%..."
- "Extract action items" → "1. Follow up... 2. Schedule..."

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT PROCESSING                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Document Text → TextToImageConverter → Images (PNG)        │
│       ↓                                                       │
│  Vision Encoder (FROZEN, 380M params)                       │
│       ↓                                                       │
│  Visual Tokens (4096-dim, ~64-400 tokens)                   │
│       ↓                                                       │
│  ┌─────────────────────────────────────────┐                │
│  │  OPTIONAL: Instruction Prompting         │                │
│  │                                           │                │
│  │  Instruction/Query Text                   │                │
│  │       ↓                                   │                │
│  │  Qwen Text Encoder (4096-dim output)     │                │
│  │       ↓                                   │                │
│  │  Text Tokens (4096-dim, ~20-50 tokens)   │                │
│  └─────────────────────────────────────────┘                │
│       ↓                                                       │
│  CONCATENATION: [Visual Tokens | Text Tokens]               │
│       ↓                                                       │
│  Combined Sequence (4096-dim, ~84-450 tokens total)         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    DECODER PROCESSING                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  MoE Decoder (TRAINABLE, 570M active params)                │
│       ↓                                                       │
│  Attends to: [Visual context | Instruction context]         │
│       ↓                                                       │
│  Generates: Answer/Summary/Extraction                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. **Dimension Matching** (No Projection Needed)
- Vision tokens: **4096 dimensions**
- Qwen text encoder output: **4096 dimensions**
- ✅ **Perfect match** → Simple concatenation, no projection layer

#### 2. **Optional Text Encoder**
- **Mode 1**: Standard summarization (text encoder OFF)
  - Input: Visual tokens only
  - Output: Summary

- **Mode 2**: Instruction prompting (text encoder ON)
  - Input: Visual tokens + Text tokens
  - Output: Instruction-guided response

#### 3. **Concatenation Strategy**
```python
# Visual tokens: [batch, num_visual_tokens, 4096]
# Text tokens: [batch, num_text_tokens, 4096]

combined = torch.cat([visual_tokens, text_tokens], dim=1)
# Result: [batch, num_visual_tokens + num_text_tokens, 4096]

# Decoder processes combined sequence
outputs = decoder(combined, labels=target)
```

#### 4. **Text Encoder Model**
- **Model**: Qwen/Qwen2.5-7B-Instruct (or lighter variant)
- **Output**: 4096-dim embeddings (last hidden state, mean pooled)
- **Trainable**: Configurable (frozen or fine-tuned)
- **Memory**: ~14GB additional (bf16)

---

## Use Cases & Examples

### Use Case 1: Question Answering (Q&A)

**Input**:
```json
{
  "document": "Artificial intelligence has transformed healthcare...",
  "instruction": "What are the main benefits of AI in healthcare?"
}
```

**Output**:
```
The main benefits include: 1) Improved diagnostic accuracy through image analysis,
2) Personalized treatment recommendations, 3) Reduced administrative burden via automation.
```

**Training Data Format**:
```json
{
  "text": "Artificial intelligence has transformed healthcare...",
  "instruction": "What are the main benefits of AI in healthcare?",
  "answer": "The main benefits include..."
}
```

---

### Use Case 2: Custom Summarization

**Input**:
```json
{
  "document": "Q3 2024 Financial Report: Revenue $45M (+15% YoY)...",
  "instruction": "Summarize focusing only on revenue trends and growth"
}
```

**Output**:
```
Revenue reached $45M in Q3 2024, representing 15% year-over-year growth.
This marks the fourth consecutive quarter of double-digit growth.
```

**Training Data Format**:
```json
{
  "text": "Q3 2024 Financial Report...",
  "instruction": "Summarize focusing only on revenue trends",
  "answer": "Revenue reached $45M..."
}
```

---

### Use Case 3: Information Extraction

**Input**:
```json
{
  "document": "Meeting Notes - Jan 15, 2025. Attendees: John (CEO), Sarah (CTO)...",
  "instruction": "Extract all action items with assigned owners"
}
```

**Output**:
```
Action Items:
1. John: Finalize Q1 budget by Jan 20
2. Sarah: Deploy new API version by Jan 25
3. Marketing: Launch campaign by Feb 1
```

**Training Data Format**:
```json
{
  "text": "Meeting Notes - Jan 15, 2025...",
  "instruction": "Extract all action items",
  "answer": "Action Items: 1. John: Finalize..."
}
```

---

## Dataset Format

### Standard Format (JSON/JSONL)

```json
{
  "text": "Source document content (will be converted to image)",
  "instruction": "Question, query, or custom instruction",
  "answer": "Expected output (summary, answer, extraction)",
  "metadata": {
    "task_type": "qa|summarization|extraction",
    "domain": "general|finance|medical|legal",
    "difficulty": "easy|medium|hard"
  }
}
```

### HuggingFace Dataset Structure

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": [...],         # List of documents
    "instruction": [...],  # List of instructions/queries
    "answer": [...],       # List of expected outputs
    "task_type": [...],    # Optional metadata
})

# Save to HuggingFace Hub
dataset.push_to_hub("username/deepsynth-instruction-dataset")
```

### Recommended Datasets

| Dataset | URL | Task Type | Size | Description |
|---------|-----|-----------|------|-------------|
| SQuAD 2.0 | `squad_v2` | Q&A | 150k | Reading comprehension |
| Natural Questions | `natural_questions` | Q&A | 300k | Open-domain QA |
| MS MARCO | `ms_marco` | Q&A | 1M | Passage retrieval & QA |
| FiQA | `fiqa` | Finance Q&A | 6k | Financial domain |
| Contract Understanding | `contract_nli` | Legal | 600+ | Contract analysis |
| Custom Instructions | Manual | Custom | Variable | Domain-specific |

### Dataset Preparation Script

```python
# Example: Convert SQuAD to DeepSynth format
from datasets import load_dataset

squad = load_dataset("squad_v2", split="train")

def convert_to_deepsynth_format(example):
    return {
        "text": example["context"],
        "instruction": example["question"],
        "answer": example["answers"]["text"][0] if example["answers"]["text"] else "No answer",
        "task_type": "qa"
    }

deepsynth_dataset = squad.map(convert_to_deepsynth_format)
deepsynth_dataset.push_to_hub("username/deepsynth-squad-qa")
```

---

## Configuration

### Training Config Parameters

```python
# src/deepsynth/training/config.py (TrainerConfig class)

# Text encoder settings
use_text_encoder: bool = False              # Enable instruction prompting
text_encoder_type: str = "qwen3"            # "qwen3", "bert", or None
text_encoder_model: str = "Qwen/Qwen2.5-7B-Instruct"  # HF model ID
text_encoder_trainable: bool = True         # Fine-tune encoder or freeze
instruction_prompt: str = ""                # Optional prompt template

# No projection needed (dimensions match)
use_text_projection: bool = False           # Keep False for Qwen 4096-dim

# Standard training params apply
batch_size: int = 8
num_epochs: int = 3
learning_rate: float = 2e-5
```

### Example Configurations

#### Config 1: Q&A Training (Text Encoder ON, Trainable)
```python
config = TrainerConfig(
    model_name="deepseek-ai/DeepSeek-OCR",
    output_dir="./models/deepsynth-qa",

    # Enable instruction prompting
    use_text_encoder=True,
    text_encoder_type="qwen3",
    text_encoder_model="Qwen/Qwen2.5-7B-Instruct",
    text_encoder_trainable=True,  # Fine-tune Qwen

    # Training params
    batch_size=4,  # Reduced for memory (text encoder adds ~14GB)
    num_epochs=3,
    learning_rate=1e-5,  # Lower LR for text encoder fine-tuning
)
```

#### Config 2: Standard Summarization (Text Encoder OFF)
```python
config = TrainerConfig(
    model_name="deepseek-ai/DeepSeek-OCR",
    output_dir="./models/deepsynth-summarizer",

    # No instruction prompting
    use_text_encoder=False,

    # Standard training
    batch_size=8,
    num_epochs=3,
    learning_rate=2e-5,
)
```

#### Config 3: Q&A with Frozen Text Encoder
```python
config = TrainerConfig(
    # Enable text encoder but freeze it
    use_text_encoder=True,
    text_encoder_trainable=False,  # Frozen Qwen (faster, less memory)

    batch_size=6,  # Can increase slightly
    learning_rate=2e-5,
)
```

---

## Training Process

### Step 1: Prepare Dataset

```bash
# Create instruction-tuning dataset
PYTHONPATH=./src python3 scripts/prepare_instruction_dataset.py \
    --source squad_v2 \
    --output baconnier/deepsynth-squad-qa \
    --max-samples 50000
```

### Step 2: Train with Instruction Prompting

```bash
# Using CLI
PYTHONPATH=./src python3 -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset baconnier/deepsynth-squad-qa \
    --output ./models/deepsynth-qa \
    --use-text-encoder \
    --text-encoder-type qwen3 \
    --text-encoder-trainable \
    --batch-size 4 \
    --num-epochs 3

# Or using Python API
from deepsynth.training.production_trainer import UnifiedProductionTrainer
from deepsynth.training.config import TrainerConfig

config = TrainerConfig(
    use_text_encoder=True,
    text_encoder_type="qwen3",
    text_encoder_trainable=True,
)

trainer = UnifiedProductionTrainer(config)
metrics, checkpoints = trainer.train_from_hf_dataset("baconnier/deepsynth-squad-qa")
```

### Step 3: Evaluate

```bash
# Run evaluation on test set
PYTHONPATH=./src python3 -m deepsynth.evaluation.evaluate \
    --model ./models/deepsynth-qa \
    --dataset baconnier/deepsynth-squad-qa \
    --split test \
    --metrics exact_match f1_score rouge
```

---

## Inference API

### Python API

```python
from deepsynth.inference import InstructionEngine

# Load model with instruction prompting
engine = InstructionEngine(
    model_path="./models/deepsynth-qa",
    use_text_encoder=True,
)

# Q&A inference
answer = engine.generate(
    document="Artificial intelligence has transformed...",
    instruction="What are the main benefits?",
    max_length=256,
)
print(answer)
# Output: "The main benefits include..."

# Custom summarization
summary = engine.generate(
    document="Q3 Financial Report...",
    instruction="Summarize revenue trends only",
    max_length=128,
)
print(summary)
```

### REST API

#### Endpoint: `POST /api/inference/instruct`

**Request**:
```json
{
  "document": "Source document text or path",
  "instruction": "Question or custom instruction",
  "model_path": "./models/deepsynth-qa",
  "max_length": 256,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "answer": "The main benefits include...",
  "confidence": 0.92,
  "tokens_generated": 45,
  "inference_time_ms": 234
}
```

**Example Usage**:
```bash
curl -X POST http://localhost:5000/api/inference/instruct \
  -H "Content-Type: application/json" \
  -d '{
    "document": "AI has transformed healthcare...",
    "instruction": "What are the main benefits?",
    "model_path": "./models/deepsynth-qa"
  }'
```

---

## UI Integration

### New UI Components

#### 1. Instruction Prompting Tab

Add new tab to main UI (`index_improved.html`):

```html
<div class="tab-content" id="instruction-tab">
    <h2>Instruction Prompting (Q&A, Custom Instructions)</h2>

    <!-- Document Input -->
    <div class="form-section">
        <h3>Document Input</h3>
        <textarea id="document-text" rows="10"
                  placeholder="Paste document text or upload file..."></textarea>
        <input type="file" id="document-upload" accept=".txt,.pdf,.docx">
    </div>

    <!-- Instruction/Query Input -->
    <div class="form-section">
        <h3>Instruction or Question</h3>
        <input type="text" id="instruction-input"
               placeholder="What are the key findings?">

        <!-- Quick Templates -->
        <select id="instruction-template">
            <option value="">Custom instruction...</option>
            <option value="summarize">Summarize this document</option>
            <option value="key_points">What are the key points?</option>
            <option value="extract_facts">Extract important facts</option>
            <option value="financial">Summarize financial aspects</option>
        </select>
    </div>

    <!-- Model Selection -->
    <div class="form-section">
        <h3>Model Selection</h3>
        <select id="model-select">
            <option value="deepsynth-qa">Q&A Model</option>
            <option value="deepsynth-summarizer">Standard Summarizer</option>
            <option value="custom">Custom Model Path...</option>
        </select>
    </div>

    <!-- Generate Button -->
    <button id="generate-answer" class="btn-primary">Generate Answer</button>

    <!-- Output Display -->
    <div class="form-section" id="answer-output" style="display:none;">
        <h3>Answer</h3>
        <div id="answer-text" class="output-box"></div>
        <div class="metrics">
            <span>Confidence: <span id="confidence-score">--</span></span>
            <span>Time: <span id="inference-time">--</span>ms</span>
        </div>
    </div>
</div>
```

#### 2. JavaScript Handler

```javascript
document.getElementById('generate-answer').addEventListener('click', async () => {
    const document = document.getElementById('document-text').value;
    const instruction = document.getElementById('instruction-input').value;
    const model = document.getElementById('model-select').value;

    if (!document || !instruction) {
        alert('Please provide both document and instruction');
        return;
    }

    try {
        const response = await fetch('/api/inference/instruct', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({document, instruction, model_path: model})
        });

        const data = await response.json();

        // Display answer
        document.getElementById('answer-text').textContent = data.answer;
        document.getElementById('confidence-score').textContent = data.confidence.toFixed(2);
        document.getElementById('inference-time').textContent = data.inference_time_ms;
        document.getElementById('answer-output').style.display = 'block';

    } catch (error) {
        alert('Error: ' + error.message);
    }
});
```

---

## Performance Considerations

### Memory Requirements

| Configuration | Vision Model | Text Encoder | Total VRAM | GPU Required |
|---------------|--------------|--------------|------------|--------------|
| Standard (no text encoder) | 16GB | - | 16GB | A100 40GB |
| With frozen text encoder | 16GB | 7GB | 23GB | A100 40GB |
| With trainable text encoder | 16GB | 14GB | 30GB | A100 40GB |
| LoRA + frozen text | 8GB | 7GB | 15GB | RTX 3090 24GB |

### Speed Impact

| Mode | Tokens/sec | Overhead |
|------|------------|----------|
| Vision only | 15 | Baseline |
| + Frozen text encoder | 14 | +7% |
| + Trainable text encoder | 12 | +25% |

### Optimization Strategies

1. **Freeze Text Encoder**: 50% memory reduction, minimal quality loss
2. **Use LoRA for Vision Model**: Train decoder only with LoRA adapters
3. **Batch Size Tuning**: Reduce batch size when text encoder enabled
4. **Gradient Checkpointing**: Enable for text encoder to save memory

---

## Success Metrics

### Training Metrics

- **Loss Convergence**: Loss < 1.0 after 3 epochs
- **Gradient Flow**: Text encoder gradients present (if trainable)
- **Memory Stability**: No OOM errors during training

### Quality Metrics (Q&A)

- **Exact Match (EM)**: > 70% on SQuAD test set
- **F1 Score**: > 80% on SQuAD test set
- **ROUGE-L**: > 60% for summarization tasks

### Quality Metrics (Custom Instructions)

- **Human Evaluation**: > 80% responses follow instruction correctly
- **Relevance Score**: > 85% responses address the query
- **Factual Accuracy**: > 90% when verifiable

### System Metrics

- **Inference Latency**: < 500ms per query (A100)
- **Throughput**: > 10 queries/sec (batch=8)
- **API Uptime**: > 99.5%

---

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Update LoRA trainer to pass text_embeddings to model
- [ ] Update production trainer with text encoder support
- [ ] Add concatenation logic: [vision_tokens | text_tokens]
- [ ] Test end-to-end with sample data

### Phase 2: Dataset Support
- [ ] Create InstructionDataset class
- [ ] Add dataset conversion scripts (SQuAD, NQ)
- [ ] Test data loading pipeline

### Phase 3: API & Inference
- [ ] Implement POST /api/inference/instruct
- [ ] Create InstructionEngine class
- [ ] Add CLI inference command

### Phase 4: UI Integration
- [ ] Add Instruction Prompting tab to web UI
- [ ] Implement JavaScript handlers
- [ ] Add instruction templates

### Phase 5: Testing & Documentation
- [ ] Unit tests for text encoder integration
- [ ] Integration tests for full pipeline
- [ ] Create user documentation
- [ ] Create training examples

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Core | 2-3 hours | Text encoder module (exists) |
| Phase 2: Dataset | 1-2 hours | Phase 1 |
| Phase 3: API | 2-3 hours | Phase 1 |
| Phase 4: UI | 2-3 hours | Phase 3 |
| Phase 5: Testing | 1-2 hours | All phases |
| **Total** | **8-13 hours** | - |

---

## Risks & Mitigation

### Risk 1: Model API Incompatibility

**Risk**: DeepSeek-OCR may not support `text_embeddings` parameter

**Mitigation**:
- Test with small batch first
- If unsupported, modify model's forward method
- Document any model API changes

### Risk 2: Memory Constraints

**Risk**: Text encoder adds 14GB VRAM, may OOM on smaller GPUs

**Mitigation**:
- Always recommend frozen text encoder by default
- Provide LoRA option for vision model
- Document memory requirements clearly

### Risk 3: Quality Degradation

**Risk**: Text encoder may interfere with vision token processing

**Mitigation**:
- Validate on standard summarization benchmarks first
- A/B test with and without text encoder
- Monitor attention patterns (visual vs text tokens)

---

## References

- **DeepSeek-OCR Paper**: https://arxiv.org/abs/2510.18234
- **Qwen2.5 Models**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **SQuAD Dataset**: https://rajpurkar.github.io/SQuAD-explorer/
- **Instruction Tuning Survey**: https://arxiv.org/abs/2308.10792

---

## Approval & Sign-off

- [ ] Technical Review: Architecture validated
- [ ] Product Review: Use cases confirmed
- [ ] Engineering Review: Implementation feasible
- [ ] UX Review: UI design approved

**Document Version**: 1.0
**Status**: Ready for Implementation
**Next Step**: Proceed with Phase 1 implementation
