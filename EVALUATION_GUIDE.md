# 📊 Evaluation Guide: Benchmarking Your Summarization Model

This guide explains how to evaluate your trained models using industry-standard benchmarks and metrics.

## 🎯 Quick Start

```bash
# Evaluate on CNN/DailyMail
deepsynth-benchmark \
    --model ./deepsynth-ocr-summarizer \
    --benchmark cnn_dailymail \
    --max-samples 1000

# Output includes ROUGE scores and comparison to SOTA baselines
```

---

## 📊 Standard Benchmarks

### Available Datasets

| Benchmark | Size | Domain | Average Length | Best Model | ROUGE-1 |
|-----------|------|--------|----------------|------------|---------|
| **CNN/DailyMail** | 287k train | News | 766 words | BART | 44.16 |
| **XSum** | 204k train | News | 431 words | Pegasus | 47.21 |
| **arXiv** | ~200k | Science | 4938 words | Longformer | 46.23 |
| **PubMed** | ~119k | Medical | 3016 words | | 45.97 |
| **SAMSum** | 14.7k | Dialogue | 94 words | BART | 53.4 |

### Benchmark Characteristics

**CNN/DailyMail**
- News articles with bullet-point highlights
- Multi-sentence summaries
- Extractive + abstractive nature
- Most common benchmark for comparison

**XSum**
- BBC articles with one-sentence summaries
- Highly abstractive
- Tests extreme compression
- Harder than CNN/DM

**arXiv**
- Scientific papers with abstracts
- Technical language
- Long documents
- Tests domain adaptation

**PubMed**
- Medical research papers
- Specialized terminology
- Similar to arXiv

**SAMSum**
- Conversation dialogues
- Informal language
- Short documents
- Tests conversational understanding

---

## 📏 Evaluation Metrics

### ROUGE Scores (Primary)

**ROUGE-1**: Unigram overlap
- Measures word-level similarity
- Range: 0-100 (higher is better)
- Typical: 40-47 for news summarization

**ROUGE-2**: Bigram overlap
- Measures phrase-level similarity
- More strict than ROUGE-1
- Typical: 18-28 for news

**ROUGE-L**: Longest Common Subsequence
- Measures sentence-level structure
- Captures fluency
- Typical: 37-49

**Formula:**
```
ROUGE = (overlap / total_in_reference) × 100
```

**Interpretation:**
- 40+: State-of-the-art performance
- 35-40: Strong performance
- 30-35: Competitive performance
- <30: Needs improvement

### BERTScore (Semantic)

Measures semantic similarity using contextual embeddings:
- **Precision**: How much of prediction is relevant
- **Recall**: How much of reference is covered
- **F1**: Harmonic mean of P and R

**Advantages:**
- Robust to paraphrasing
- Captures meaning, not just words
- Correlates better with human judgment

**Typical Scores:**
- 85-92: Excellent semantic match
- 80-85: Good semantic match
- <80: Poor semantic match

### Additional Metrics

**Compression Ratio**
```
ratio = len(original) / len(summary)
```
- Typical: 3-10x for news
- Higher = more aggressive compression

**Length Statistics**
- Average prediction length
- Average reference length
- Distribution analysis

---

## 🏃 Running Benchmarks

### Basic Usage

```bash
deepsynth-benchmark \
    --model ./your-model \
    --benchmark cnn_dailymail \
    --max-samples 1000
```

### Advanced Options

```bash
deepsynth-benchmark \
    --model username/hf-model \      # HuggingFace model
    --benchmark xsum \                # Different dataset
    --split test \                    # test/validation/train
    --max-samples 5000 \              # More samples
    --max-length 128 \                # Summary length
    --no-bertscore \                  # Skip BERTScore (faster)
    --output results.json             # Save results
```

### Evaluate Multiple Benchmarks

```bash
#!/bin/bash
# benchmark_all.sh

MODEL="./deepsynth-ocr-summarizer"

for BENCHMARK in cnn_dailymail xsum arxiv samsum; do
    echo "Evaluating on $BENCHMARK..."
    deepsynth-benchmark \
        --model $MODEL \
        --benchmark $BENCHMARK \
        --max-samples 1000 \
        --output results_${BENCHMARK}.json
done
```

---

## 📈 Interpreting Results

### Sample Output

```
======================================================================
BENCHMARK: CNN/DailyMail
======================================================================

ROUGE Scores:
  ROUGE-1: 42.35 (P: 44.12, R: 41.23)
  ROUGE-2: 19.87 (P: 21.45, R: 18.76)
  ROUGE-L: 39.12 (P: 40.89, R: 37.98)

BERTScore:
  F1: 87.23 (P: 88.12, R: 86.45)

Summary Statistics:
  Avg prediction length: 58.3 words
  Avg reference length: 56.2 words
  Compression ratio: 13.15x

Comparison to SOTA:
  ROUGE-1: Your 42.35 vs SOTA 44.16
  📊 Your model is competitive with SOTA (within 5 points)
```

### What This Means

**ROUGE-1: 42.35**
- Your model captures 42% of important words
- Within 2 points of BART baseline
- Competitive performance

**ROUGE-2: 19.87**
- Strong bigram overlap
- Captures key phrases well
- Slightly below SOTA (21.28)

**ROUGE-L: 39.12**
- Good sentence structure
- Fluent summaries
- Room for improvement

**BERTScore: 87.23**
- Excellent semantic similarity
- Good at paraphrasing
- Captures meaning effectively

---

## 🎯 Performance Targets

### CNN/DailyMail

| Metric | Poor | Fair | Good | Excellent | SOTA |
|--------|------|------|------|-----------|------|
| ROUGE-1 | <35 | 35-40 | 40-43 | 43-45 | 44.16 |
| ROUGE-2 | <15 | 15-18 | 18-21 | 21-23 | 21.28 |
| ROUGE-L | <32 | 32-37 | 37-40 | 40-42 | 40.90 |

### XSum

| Metric | Poor | Fair | Good | Excellent | SOTA |
|--------|------|------|------|-----------|------|
| ROUGE-1 | <40 | 40-44 | 44-46 | 46-48 | 47.21 |
| ROUGE-2 | <18 | 18-22 | 22-24 | 24-26 | 24.56 |
| ROUGE-L | <35 | 35-38 | 38-40 | 40-42 | 39.25 |

---

## 🔬 Deep Analysis

### Compare Against Baselines

```python
from evaluation.benchmarks import BENCHMARKS

# Get baseline scores
baseline = BENCHMARKS["cnn_dailymail"].typical_scores
print(f"BART baseline: {baseline}")

# Compare your results
your_scores = {
    "rouge1": 42.35,
    "rouge2": 19.87,
    "rougeL": 39.12,
}

for metric, score in your_scores.items():
    diff = score - baseline[metric]
    print(f"{metric}: {score:.2f} vs {baseline[metric]:.2f} ({diff:+.2f})")
```

### Analyze Errors

```python
# Generate summaries for analysis
predictions, references = model_evaluator.evaluate_dataset(dataset)

# Find worst examples
from evaluation.benchmarks import SummarizationEvaluator
evaluator = SummarizationEvaluator()

errors = []
for pred, ref in zip(predictions, references):
    metrics = evaluator.evaluate([pred], [ref])
    if metrics.rouge1_f < 0.3:  # Poor ROUGE-1
        errors.append({
            "prediction": pred,
            "reference": ref,
            "rouge1": metrics.rouge1_f
        })

# Analyze error patterns
for error in errors[:5]:
    print(f"ROUGE-1: {error['rouge1']:.2f}")
    print(f"Reference: {error['reference']}")
    print(f"Prediction: {error['prediction']}")
    print("-" * 50)
```

---

## 📊 Statistical Significance

### Multiple Runs

For robust evaluation, run multiple times:

```bash
# Run 5 times with different random seeds
for i in {1..5}; do
    deepsynth-benchmark \
        --model ./model \
        --benchmark cnn_dailymail \
        --max-samples 1000 \
        --output results_run${i}.json
done

# Compute mean and std
python -c "
import json
import numpy as np

scores = []
for i in range(1, 6):
    with open(f'results_run{i}.json') as f:
        data = json.load(f)
        scores.append(data['metrics']['rouge1_f'])

print(f'ROUGE-1: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
"
```

---

## 🎨 Visualization

### Plot Results

```python
import matplotlib.pyplot as plt
import json

# Load results
benchmarks = ["cnn_dailymail", "xsum", "arxiv", "samsum"]
rouge1_scores = []

for benchmark in benchmarks:
    with open(f"results_{benchmark}.json") as f:
        data = json.load(f)
        rouge1_scores.append(data["metrics"]["rouge1_f"] * 100)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(benchmarks, rouge1_scores)
plt.axhline(y=44.16, color='r', linestyle='--', label='BART baseline (CNN/DM)')
plt.ylabel("ROUGE-1 Score")
plt.title("Model Performance Across Benchmarks")
plt.legend()
plt.savefig("benchmark_results.png")
```

---

## 🚀 Best Practices

### 1. **Use Test Set**
```bash
# Always evaluate on test set, not training data
deepsynth-benchmark --benchmark cnn_dailymail --split test
```

### 2. **Sufficient Sample Size**
```bash
# Use at least 1000 samples for reliable estimates
deepsynth-benchmark --max-samples 1000
```

### 3. **Multiple Metrics**
```bash
# Don't rely on ROUGE alone
deepsynth-benchmark --benchmark cnn_dailymail  # Includes BERTScore
```

### 4. **Cross-Dataset Evaluation**
```bash
# Test generalization
deepsynth-benchmark --model ./model --benchmark xsum
deepsynth-benchmark --model ./model --benchmark arxiv
```

### 5. **Human Evaluation**
```python
# Sample predictions for human review
import random
predictions, references = model_evaluator.evaluate_dataset(dataset)

samples = random.sample(list(zip(predictions, references)), 10)
for i, (pred, ref) in enumerate(samples):
    print(f"\nExample {i+1}:")
    print(f"Reference: {ref}")
    print(f"Prediction: {pred}")
    print(f"Quality (1-5): _____")
```

---

## 📚 Advanced Topics

### Domain Adaptation

Test how well your model transfers:

```bash
# Train on CNN/DM
python run_complete_pipeline.py  # Uses CNN/DM by default

# Evaluate on different domains
deepsynth-benchmark --model ./model --benchmark arxiv
deepsynth-benchmark --model ./model --benchmark pubmed
deepsynth-benchmark --model ./model --benchmark samsum
```

### Few-Shot Learning

Evaluate with minimal training data:

```bash
# Train on 100 examples
echo "MAX_SAMPLES_PER_SPLIT=100" >> .env
python run_complete_pipeline.py

# Evaluate performance
deepsynth-benchmark --model ./model --benchmark cnn_dailymail
```

### Zero-Shot Evaluation

Test base model without fine-tuning:

```bash
deepsynth-benchmark \
    --model deepseek-ai/DeepSeek-OCR \
    --benchmark cnn_dailymail \
    --max-samples 100
```

---

## 🔍 Troubleshooting

### Low ROUGE Scores

**Possible causes:**
1. Insufficient training data
2. Learning rate too high/low
3. Wrong hyperparameters
4. Model not converged

**Solutions:**
```bash
# Increase training data
MAX_SAMPLES_PER_SPLIT=10000

# Adjust learning rate
LEARNING_RATE=1e-5

# More epochs
NUM_EPOCHS=5
```

### High ROUGE, Low BERTScore

**Meaning:** Model copies words but misses meaning

**Solution:** Encourage more abstractive summaries
```bash
# Adjust generation parameters
--temperature 0.9 \
--top_p 0.95 \
--no_repeat_ngram_size 3
```

### Slow Evaluation

**Speed up evaluation:**
```bash
# Skip BERTScore
deepsynth-benchmark --no-bertscore

# Reduce samples
deepsynth-benchmark --max-samples 500

# Use smaller model
deepsynth-benchmark --model facebook/bart-base
```

---

## 📖 References

**ROUGE:**
- [Lin, 2004](https://aclanthology.org/W04-1013.pdf) - Original ROUGE paper

**BERTScore:**
- [Zhang et al., 2019](https://arxiv.org/abs/1904.09675) - BERTScore paper

**Benchmarks:**
- [See et al., 2017](https://arxiv.org/abs/1704.04368) - CNN/DailyMail
- [Narayan et al., 2018](https://arxiv.org/abs/1808.08745) - XSum

**SOTA Models:**
- [Lewis et al., 2019](https://arxiv.org/abs/1910.13461) - BART
- [Zhang et al., 2019](https://arxiv.org/abs/1912.08777) - Pegasus

---

## 🎯 Summary

**Key Takeaways:**
1. Use standard benchmarks (CNN/DM, XSum) for comparison
2. Report ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore
3. Evaluate on test set with 1000+ samples
4. Compare against published baselines
5. Analyze errors and edge cases
6. Consider domain transfer and generalization

**Quick Command:**
```bash
deepsynth-benchmark \
    --model ./your-model \
    --benchmark cnn_dailymail \
    --max-samples 1000 \
    --output results.json
```

---

**Ready to benchmark? Start evaluating!** 🚀
