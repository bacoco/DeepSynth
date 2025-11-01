"""
Benchmark Runner for evaluating trained models.
Handles reproducible benchmark creation and metric computation.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .state_manager import JobStatus, StateManager

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Handles model benchmarking with reproducible test sets."""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.hf_api = HfApi()

    def run_benchmark(self, job_id: str):
        """
        Run benchmark evaluation on a model using pre-defined test split.

        Args:
            job_id: Job identifier for this benchmark run
        """
        job = self.state_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = JobStatus.IN_PROGRESS.value
        self.state_manager.update_job(job)

        try:
            config = job.config

            # 1. Load benchmark dataset (ONLY benchmark indices)
            logger.info("Loading benchmark dataset...")
            benchmark_set = self._load_benchmark_set(
                split_id=config["split_id"],
                dataset_repos=config["dataset_repos"]
            )

            # 2. Load model
            logger.info(f"Loading model from {config['model_path']}...")
            model, tokenizer = self._load_model(config["model_path"])

            # 3. Run inference
            logger.info("Running inference...")
            predictions, references, latencies = self._run_inference(
                model=model,
                tokenizer=tokenizer,
                benchmark_set=benchmark_set
            )

            # 4. Compute metrics
            logger.info("Computing metrics...")
            results = self._compute_metrics(
                predictions=predictions,
                references=references,
                latencies=latencies,
                config=config
            )

            # 5. Push results to HuggingFace model card
            if config.get("hub_model_id"):
                logger.info("Pushing benchmark results to HuggingFace...")
                self._update_model_card_with_benchmark(
                    hub_model_id=config["hub_model_id"],
                    results=results,
                    checkpoint_name=config.get("checkpoint_name", "final")
                )

            # Update job with results
            job.config["benchmark_results"] = results
            job.status = JobStatus.COMPLETED.value
            self.state_manager.update_job(job)

            logger.info("Benchmark completed successfully!")
            return results

        except Exception as e:
            logger.exception("Benchmark failed")
            job.status = JobStatus.FAILED.value
            job.last_error = str(e)
            self.state_manager.update_job(job)
            raise

    def _load_benchmark_set(
        self, split_id: str, dataset_repos: List[str]
    ) -> Dataset:
        """Load and concatenate benchmark indices from multiple datasets."""
        benchmark_indices = self.state_manager.get_split_indices(split_id, "benchmark")

        if not benchmark_indices:
            raise ValueError(f"No benchmark indices found for split_id: {split_id}")

        benchmark_datasets = []
        for repo in dataset_repos:
            logger.info(f"Loading benchmark samples from {repo}...")
            ds = load_dataset(repo, split="train")

            # Filter to benchmark indices only
            repo_indices = benchmark_indices.get(repo, [])
            if repo_indices:
                ds_filtered = ds.select(repo_indices)
                benchmark_datasets.append(ds_filtered)
                logger.info(f"  Loaded {len(ds_filtered)} benchmark samples from {repo}")

        if not benchmark_datasets:
            raise ValueError("No benchmark samples loaded!")

        return concatenate_datasets(benchmark_datasets)

    def _load_model(self, model_path: str):
        """Load model and tokenizer from HuggingFace or local path."""
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")

            model.eval()
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _run_inference(
        self, model, tokenizer, benchmark_set: Dataset
    ) -> tuple:
        """Run inference on benchmark set and collect predictions."""
        predictions = []
        references = []
        latencies = []

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for sample in tqdm(benchmark_set, desc="Running inference"):
            try:
                # Get input image
                image = sample.get("image")
                if image is None:
                    logger.warning("Sample missing 'image' field, skipping")
                    continue

                # Prepare input (for vision models like DeepSeek-OCR)
                start_time = time.time()

                # TODO: Adapt this based on your model's input format
                # For DeepSeek-OCR, input is the rendered image
                # For text-only models, use sample["text"]

                # Placeholder for image-based generation
                # You may need to adjust based on DeepSeek-OCR's API
                with torch.no_grad():
                    # Example for text-based models:
                    # inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=512)
                    # inputs = inputs.to(device)
                    # outputs = model.generate(**inputs, max_length=128)
                    # prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Placeholder - adjust for actual model
                    prediction = "PLACEHOLDER_SUMMARY"

                latency = time.time() - start_time
                latencies.append(latency)

                predictions.append(prediction)
                references.append(sample["summary"])

            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue

        return predictions, references, latencies

    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        latencies: List[float],
        config: Dict
    ) -> Dict:
        """Compute ROUGE, BERTScore, and other metrics."""
        from deepsynth.evaluation.metrics import compute_rouge, compute_bertscore

        # ROUGE scores
        rouge_scores = compute_rouge(predictions, references)

        # BERTScore
        bert_scores = compute_bertscore(predictions, references)

        # Latency statistics
        avg_latency = np.mean(latencies) if latencies else 0
        p50_latency = np.percentile(latencies, 50) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0

        # Compression ratio
        compression_ratios = [
            len(ref.split()) / len(pred.split()) if pred else 0
            for ref, pred in zip(references, predictions)
        ]
        avg_compression = np.mean(compression_ratios) if compression_ratios else 0

        results = {
            "rouge1": float(rouge_scores.get("rouge1", 0)),
            "rouge2": float(rouge_scores.get("rouge2", 0)),
            "rougeL": float(rouge_scores.get("rougeL", 0)),
            "bertscore_precision": float(bert_scores.get("precision", 0)),
            "bertscore_recall": float(bert_scores.get("recall", 0)),
            "bertscore_f1": float(bert_scores.get("f1", 0)),
            "avg_latency_ms": float(avg_latency * 1000),
            "p50_latency_ms": float(p50_latency * 1000),
            "p95_latency_ms": float(p95_latency * 1000),
            "avg_compression_ratio": float(avg_compression),
            "total_samples": len(predictions),
            "seed": config.get("seed", "unknown"),
            "benchmark_percentage": config.get("benchmark_percentage", "unknown"),
            "evaluated_at": datetime.utcnow().isoformat()
        }

        return results

    def _update_model_card_with_benchmark(
        self, hub_model_id: str, results: Dict, checkpoint_name: str
    ):
        """Update model card on HuggingFace with benchmark results."""
        try:
            # Download existing README
            try:
                readme_path = hf_hub_download(
                    repo_id=hub_model_id,
                    filename="README.md",
                    repo_type="model",
                    token=os.environ.get("HF_TOKEN")
                )
                with open(readme_path, 'r') as f:
                    existing_content = f.read()
            except Exception:
                existing_content = f"# {hub_model_id}\n\nModel card\n"

            # Create benchmark section
            benchmark_section = self._format_benchmark_section(results, checkpoint_name)

            # Replace or append benchmark section
            import re
            if "## Benchmark Results" in existing_content:
                # Replace existing section
                updated_content = re.sub(
                    r"## Benchmark Results.*?(?=\n##|\Z)",
                    benchmark_section,
                    existing_content,
                    flags=re.DOTALL
                )
            else:
                # Append new section
                updated_content = existing_content.rstrip() + "\n\n" + benchmark_section

            # Upload to HuggingFace
            self.hf_api.upload_file(
                path_or_fileobj=updated_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=hub_model_id,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
                commit_message=f"Update benchmark results for {checkpoint_name}"
            )

            logger.info(f"✅ Benchmark results pushed to {hub_model_id}")

        except Exception as e:
            logger.error(f"Failed to update model card: {e}")
            # Don't fail the whole benchmark just because of model card update
            raise

    def _format_benchmark_section(self, results: Dict, checkpoint_name: str) -> str:
        """Format benchmark results as markdown section."""
        return f"""## Benchmark Results

**Checkpoint:** {checkpoint_name}
**Evaluation Date:** {results['evaluated_at']}
**Benchmark Size:** {results['total_samples']} samples
**Seed:** {results['seed']}
**Benchmark Split:** {results['benchmark_percentage'] * 100:.1f}% of total data

### Metrics

| Metric | Score |
|--------|-------|
| **ROUGE-1** | {results['rouge1']:.4f} |
| **ROUGE-2** | {results['rouge2']:.4f} |
| **ROUGE-L** | {results['rougeL']:.4f} |
| **BERTScore Precision** | {results['bertscore_precision']:.4f} |
| **BERTScore Recall** | {results['bertscore_recall']:.4f} |
| **BERTScore F1** | {results['bertscore_f1']:.4f} |
| **Avg Latency** | {results['avg_latency_ms']:.2f} ms |
| **P50 Latency** | {results['p50_latency_ms']:.2f} ms |
| **P95 Latency** | {results['p95_latency_ms']:.2f} ms |
| **Avg Compression Ratio** | {results['avg_compression_ratio']:.2f}x |
"""
