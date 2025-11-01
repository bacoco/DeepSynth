"""Flask web application for dataset generation and training jobs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Thread
from typing import Optional

from flask import Flask, jsonify, render_template, request

from apps.web.config import WebConfig
from deepsynth.training.optimal_configs import (
    PRESET_CONFIGS,
    get_optimal_config,
    list_benchmark_datasets,
)

from deepsynth.data.transforms.text_to_image import DEEPSEEK_OCR_RESOLUTIONS

from .dataset_generator_improved import IncrementalDatasetGenerator, ModelTrainer
from .state_manager import JobStatus, StateManager
from .benchmark_runner import BenchmarkRunner

logger = logging.getLogger(__name__)


def create_app(config: Optional[type] = None) -> Flask:
    """Application factory used by the CLI and WSGI servers."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app_config = config or WebConfig
    app.config.from_object(app_config)

    job_manager_cfg = app.config["JOB_MANAGER"]
    job_manager_cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (job_manager_cfg.state_dir / "hashes").mkdir(exist_ok=True)

    state_manager = StateManager(job_manager_cfg.state_dir)
    dataset_generator = IncrementalDatasetGenerator(state_manager)
    model_trainer = ModelTrainer(state_manager)
    benchmark_runner = BenchmarkRunner(state_manager)

    _register_routes(app, state_manager, dataset_generator, model_trainer, benchmark_runner)

    app.state_manager = state_manager  # type: ignore[attr-defined]
    app.dataset_generator = dataset_generator  # type: ignore[attr-defined]
    app.model_trainer = model_trainer  # type: ignore[attr-defined]
    app.benchmark_runner = benchmark_runner  # type: ignore[attr-defined]

    return app


def _register_routes(
    app: Flask,
    state_manager: StateManager,
    dataset_generator: IncrementalDatasetGenerator,
    model_trainer: ModelTrainer,
    benchmark_runner: BenchmarkRunner,
) -> None:
    """Bind HTTP routes to the Flask application."""

    @app.route("/")
    def index():
        """Render the main UI."""

        return render_template("index_improved.html")

    @app.route("/qa")
    def qa_generator():
        """Render the Q&A dataset generator UI."""

        return render_template("qa_generator.html")

    @app.route("/api/jobs", methods=["GET"])
    def list_jobs():
        """List all registered jobs."""

        job_type = request.args.get("type")
        jobs = state_manager.list_jobs(job_type)

        return jsonify(
            {
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "status": job.status,
                        "created_at": job.created_at,
                        "updated_at": job.updated_at,
                        "progress": state_manager.get_progress(job.job_id),
                        "metrics": job.config.get("metrics") if job.config else None,
                        "hf_dataset_repo": job.hf_dataset_repo,
                        "model_output_path": job.model_output_path,
                    }
                    for job in jobs
                ]
            }
        )

    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def get_job(job_id: str):
        """Get details for a given job."""

        job = state_manager.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        progress = state_manager.get_progress(job_id)

        return jsonify(
            {
                "job": {
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "status": job.status,
                    "config": job.config,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                    "hf_dataset_repo": job.hf_dataset_repo,
                    "model_output_path": job.model_output_path,
                },
                "progress": progress,
            }
        )

    @app.route("/api/jobs/<job_id>/progress", methods=["GET"])
    def get_job_progress(job_id: str):
        """Get progress information for a job."""

        progress = state_manager.get_progress(job_id)
        if not progress:
            return jsonify({"error": "Job not found"}), 404

        return jsonify(progress)

    @app.route("/api/jobs/<job_id>/pause", methods=["POST"])
    def pause_job(job_id: str):
        """Pause a running job."""

        job = state_manager.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        if job.status != JobStatus.IN_PROGRESS.value:
            return jsonify({"error": "Job is not running"}), 400

        job.status = JobStatus.PAUSED.value
        state_manager.update_job(job)

        return jsonify({
            "job_id": job_id,
            "status": "paused",
            "message": "Job paused successfully"
        })

    @app.route("/api/jobs/<job_id>/resume", methods=["POST"])
    def resume_job(job_id: str):
        """Resume a paused job."""

        job = state_manager.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        if job.status != JobStatus.PAUSED.value:
            return jsonify({"error": "Job is not paused"}), 400

        job.status = JobStatus.IN_PROGRESS.value
        state_manager.update_job(job)

        # Restart the background thread based on job type
        if job.job_type == "dataset_generation":
            thread = Thread(
                target=_run_dataset_generation,
                args=(dataset_generator, state_manager, job_id),
                daemon=True,
            )
            thread.start()
        elif job.job_type == "model_training":
            thread = Thread(
                target=_run_model_training,
                args=(model_trainer, state_manager, job_id),
                daemon=True,
            )
            thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "in_progress",
            "message": "Job resumed successfully"
        })

    @app.route("/api/jobs/<job_id>", methods=["DELETE"])
    def delete_job(job_id: str):
        """Delete a job and its associated data."""

        job = state_manager.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        if job.status == JobStatus.IN_PROGRESS.value:
            return jsonify({"error": "Cannot delete running job. Pause it first."}), 400

        # Delete job files and metadata
        state_manager.delete_job(job_id)

        return jsonify({
            "job_id": job_id,
            "message": "Job deleted successfully"
        })

    @app.route("/api/dataset/generate", methods=["POST"])
    def generate_dataset():
        """Start a dataset generation job."""

        try:
            data = request.json or {}

            required_fields = [
                "source_dataset",
                "text_field",
                "summary_field",
                "output_dir",
            ]
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            config = {
                "source_dataset": data["source_dataset"],
                "source_subset": data.get("source_subset"),
                "source_split": data.get("source_split", "train"),
                "text_field": data["text_field"],
                "summary_field": data["summary_field"],
                "output_dir": data["output_dir"],
                "max_samples": data.get("max_samples"),
                "dataset_name": data.get("dataset_name", "generated-dataset"),
                "hf_username": data.get(
                    "hf_username", os.environ.get("HF_USERNAME")
                ),
                "hf_dataset_repo": data.get("hf_dataset_repo"),
                "private_dataset": data.get("private_dataset", False),
                "hf_token": data.get("hf_token", os.environ.get("HF_TOKEN")),
                "multi_resolution": data.get("multi_resolution", False),
                "resolution_sizes": data.get("resolution_sizes", None),
                # Instruction prompting for LoRA training
                "instruction_prompt": data.get("instruction_prompt", ""),
            }

            job_id = state_manager.create_job("dataset_generation", config)

            thread = Thread(
                target=_run_dataset_generation,
                args=(dataset_generator, state_manager, job_id),
                daemon=True,
            )
            thread.start()

            return jsonify(
                {
                    "job_id": job_id,
                    "message": "Dataset generation started",
                    "status": "in_progress",
                }
            )

        except Exception as exc:  # noqa: BLE001 - bubble useful message to UI
            logger.exception("Error starting dataset generation")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/dataset/generate-qa", methods=["POST"])
    def generate_qa_dataset():
        """Start a Q&A dataset generation job."""

        try:
            data = request.json or {}

            config = {
                "qa_sources": data.get("qa_sources", ["natural_questions", "ms_marco"]),
                "target_resolution": data.get("target_resolution", "gundam"),
                "max_nq_samples": data.get("max_nq_samples"),
                "max_marco_samples": data.get("max_marco_samples"),
                "dataset_name": data.get("dataset_name", "deepsynth-qa"),
                "batch_size": data.get("batch_size", 5000),
                "hf_username": data.get("hf_username", os.environ.get("HF_USERNAME")),
                "hf_token": data.get("hf_token", os.environ.get("HF_TOKEN")),
                "private_dataset": data.get("private_dataset", False),
            }

            job_id = state_manager.create_job("qa_dataset_generation", config)

            thread = Thread(
                target=_run_qa_generation,
                args=(dataset_generator, state_manager, job_id),
                daemon=True,
            )
            thread.start()

            return jsonify(
                {
                    "job_id": job_id,
                    "message": "Q&A dataset generation started",
                    "status": "in_progress",
                }
            )

        except Exception as exc:  # noqa: BLE001 - bubble useful message to UI
            logger.exception("Error starting Q&A dataset generation")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/model/train", methods=["POST"])
    def train_model():
        """Start a model training job."""

        try:
            data = request.json or {}

            required_fields = [
                "model_name",
                "output_dir",
                "dataset_path",
                "trainer_type",
            ]
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            config = {
                "model_name": data["model_name"],
                "output_dir": data["output_dir"],
                "dataset_path": data["dataset_path"],
                "trainer_type": data["trainer_type"],
                "hf_username": data.get(
                    "hf_username", os.environ.get("HF_USERNAME")
                ),
                "hf_model_repo": data.get("hf_model_repo"),
                "hf_token": data.get("hf_token", os.environ.get("HF_TOKEN")),
                "training_config": data.get("training_config"),
            }

            job_id = state_manager.create_job("model_training", config)

            thread = Thread(
                target=_run_model_training,
                args=(model_trainer, state_manager, job_id),
                daemon=True,
            )
            thread.start()

            return jsonify(
                {
                    "job_id": job_id,
                    "message": "Model training started",
                    "status": "in_progress",
                }
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error starting model training")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint for Docker and monitoring."""

        return jsonify({"status": "ok"})

    @app.route("/api/lora/presets", methods=["GET"])
    def get_lora_presets():
        """Return available LoRA preset configurations."""
        from deepsynth.training.lora_config import LORA_PRESETS

        presets = {}
        for name, config in LORA_PRESETS.items():
            presets[name] = config.to_dict()

        return jsonify({"presets": presets})

    @app.route("/api/lora/estimate", methods=["POST"])
    def estimate_lora_resources():
        """Estimate memory and training time for LoRA configuration."""
        try:
            data = request.json or {}

            lora_rank = data.get("lora_rank", 16)
            use_qlora = data.get("use_qlora", False)
            qlora_bits = data.get("qlora_bits", 4)
            use_text_encoder = data.get("use_text_encoder", False)
            batch_size = data.get("batch_size", 8)

            # Base memory estimation
            base_memory = 16.0  # GB for full model

            # LoRA reduction
            if use_qlora:
                if qlora_bits == 4:
                    memory = base_memory * 0.25  # 75% reduction
                elif qlora_bits == 8:
                    memory = base_memory * 0.5  # 50% reduction
                else:
                    memory = base_memory
            else:
                memory = base_memory * 0.5  # Standard LoRA ~50% reduction

            # Add text encoder overhead
            if use_text_encoder:
                memory += 4.0  # Additional 4GB for text encoder

            # Adjust for batch size
            memory += (batch_size - 8) * 0.5  # ~500MB per additional batch item

            # Estimate trainable parameters
            if lora_rank <= 8:
                trainable_params_m = 2.0
            elif lora_rank <= 16:
                trainable_params_m = 4.0
            elif lora_rank <= 32:
                trainable_params_m = 8.0
            else:
                trainable_params_m = 16.0

            if use_text_encoder:
                trainable_params_m += 8.0  # Text encoder params

            # GPU compatibility
            gpu_fit = {
                "T4 (16GB)": memory <= 16.0,
                "RTX 3090 (24GB)": memory <= 24.0,
                "A100 (40GB)": memory <= 40.0,
                "A100 (80GB)": memory <= 80.0,
            }

            # Speed estimate (relative to full fine-tuning)
            if use_qlora and qlora_bits == 4:
                speed_multiplier = 1.0  # Same speed as full
            elif use_qlora and qlora_bits == 8:
                speed_multiplier = 1.2
            else:
                speed_multiplier = 1.5

            return jsonify({
                "estimated_vram_gb": round(memory, 1),
                "trainable_params_millions": round(trainable_params_m, 1),
                "gpu_fit": gpu_fit,
                "speed_multiplier": round(speed_multiplier, 2),
                "configuration": {
                    "lora_rank": lora_rank,
                    "use_qlora": use_qlora,
                    "qlora_bits": qlora_bits if use_qlora else None,
                    "use_text_encoder": use_text_encoder,
                }
            })

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error estimating LoRA resources")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/datasets/benchmarks", methods=["GET"])
    def list_benchmarks():
        """Return available benchmark datasets."""

        return jsonify({"benchmarks": list_benchmark_datasets()})

    @app.route("/api/datasets/presets", methods=["GET"])
    def list_presets():
        """Return available preset configurations."""

        return jsonify({"presets": PRESET_CONFIGS})

    @app.route("/api/training/presets", methods=["GET"])
    def list_training_presets():
        """Return available training configuration presets."""

        # Training presets with recommended hyperparameters
        training_presets = {
            "quick_test": {
                "name": "Quick Test",
                "description": "Fast training for testing (1 epoch, small batch)",
                "batch_size": 4,
                "num_epochs": 1,
                "learning_rate": 3e-5,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 100,
                "save_steps": 1000,
            },
            "standard": {
                "name": "Standard Training",
                "description": "Balanced training (3 epochs, moderate batch)",
                "batch_size": 8,
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 500,
                "save_steps": 1000,
            },
            "high_quality": {
                "name": "High Quality",
                "description": "Best results (5 epochs, careful tuning)",
                "batch_size": 4,
                "num_epochs": 5,
                "learning_rate": 1e-5,
                "gradient_accumulation_steps": 8,
                "warmup_steps": 1000,
                "save_steps": 500,
            },
            "memory_efficient": {
                "name": "Memory Efficient",
                "description": "Low VRAM (batch 2, grad accumulation 16)",
                "batch_size": 2,
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "gradient_accumulation_steps": 16,
                "warmup_steps": 500,
                "save_steps": 1000,
            },
        }

        return jsonify({"presets": training_presets})

    @app.route("/api/datasets/generated", methods=["GET"])
    def list_generated_datasets():
        """List all generated deepsynth datasets (alias for /api/datasets/deepsynth)."""

        # Reuse the existing deepsynth datasets endpoint
        return list_deepsynth_datasets()

    @app.route("/api/training/checkpoints", methods=["GET"])
    def list_training_checkpoints():
        """List available model checkpoints from completed training jobs."""

        try:
            # Get all completed training jobs
            all_jobs = state_manager.list_jobs("model_training")
            completed_jobs = [
                job for job in all_jobs
                if job.status == JobStatus.COMPLETED.value
            ]

            checkpoints = []
            for job in completed_jobs:
                if job.model_output_path:
                    # Check if checkpoint directory exists
                    checkpoint_path = Path(job.model_output_path)
                    if checkpoint_path.exists():
                        # List checkpoint subdirectories
                        checkpoint_dirs = []
                        for item in checkpoint_path.iterdir():
                            if item.is_dir() and (item.name.startswith("checkpoint-") or item.name == "final"):
                                checkpoint_dirs.append({
                                    "name": item.name,
                                    "path": str(item),
                                    "size_mb": sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
                                })

                        if checkpoint_dirs:
                            checkpoints.append({
                                "job_id": job.job_id,
                                "model_name": job.config.get("model_name", "unknown"),
                                "output_path": job.model_output_path,
                                "created_at": job.created_at,
                                "checkpoints": checkpoint_dirs,
                                "hf_model_repo": job.config.get("hf_model_repo"),
                            })

            return jsonify({"checkpoints": checkpoints})

        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list checkpoints")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/datasets/optimal_config", methods=["POST"])
    def optimal_config():
        """Return the optimal configuration for a dataset."""

        payload = request.json or {}
        dataset_name = payload.get("dataset_name")
        if not dataset_name:
            return jsonify({"error": "dataset_name is required"}), 400

        optimal = get_optimal_config(dataset_name)
        if not optimal:
            return jsonify({"error": "Dataset configuration not found"}), 404

        return jsonify({"config": optimal})

    @app.route("/api/datasets/deepsynth", methods=["GET"])
    def list_deepsynth_datasets():
        """List all deepsynth-* datasets from HuggingFace for the user."""
        from huggingface_hub import HfApi, list_datasets

        try:
            hf_token = os.environ.get("HF_TOKEN")
            api = HfApi(token=hf_token)

            # Get username from environment (default: baconnier)
            username = os.environ.get("HF_USERNAME", "baconnier")

            # List all datasets for this user
            all_datasets = list(api.list_datasets(author=username))

            # Filter for deepsynth-* datasets
            deepsynth_datasets = []
            for ds in all_datasets:
                if ds.id.split('/')[-1].startswith('deepsynth-'):
                    try:
                        # Get dataset info
                        ds_info = api.dataset_info(ds.id)

                        # Extract language from name (e.g., deepsynth-fr -> fr)
                        name = ds.id.split('/')[-1]
                        lang_code = name.replace('deepsynth-', '')

                        # Map language codes to flags
                        lang_flags = {
                            'fr': '🇫🇷',
                            'es': '🇪🇸',
                            'de': '🇩🇪',
                            'en-news': '📰',
                            'en-arxiv': '📚',
                            'en-xsum': '📺',
                            'en-legal': '⚖️'
                        }

                        # Get train split size if available
                        train_samples = 0
                        if hasattr(ds_info, 'splits') and 'train' in ds_info.splits:
                            train_samples = ds_info.splits['train'].num_examples

                        deepsynth_datasets.append({
                            "repo": ds.id,
                            "name": name,
                            "language": lang_code,
                            "flag": lang_flags.get(lang_code, '🌍'),
                            "last_modified": ds.lastModified.isoformat() if hasattr(ds, 'lastModified') else None,
                            "downloads": getattr(ds, 'downloads', 0),
                            "train_samples": train_samples
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get info for {ds.id}: {e}")
                        continue

            # Sort by name
            deepsynth_datasets.sort(key=lambda x: x['name'])

            return jsonify({"datasets": deepsynth_datasets})

        except Exception as e:
            logger.exception("Failed to list datasets")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/datasets/prepare-split", methods=["POST"])
    def prepare_train_benchmark_split():
        """Create and save a reproducible train/benchmark split."""
        import random
        from datasets import load_dataset

        try:
            data = request.json or {}

            required = ["dataset_repos", "seed", "benchmark_percentage"]
            for field in required:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            dataset_repos = data["dataset_repos"]
            seed = int(data["seed"])
            benchmark_pct = float(data["benchmark_percentage"])
            stratify = data.get("stratify", True)

            if not (0 < benchmark_pct < 1):
                return jsonify({"error": "benchmark_percentage must be between 0 and 1"}), 400

            # Set random seed for reproducibility
            random.seed(seed)
            import numpy as np
            np.random.seed(seed)

            train_indices = {}
            benchmark_indices = {}
            train_total = 0
            benchmark_total = 0

            # Create split for each dataset
            # Note: dataset sizes should be passed from client (already loaded in UI)
            dataset_sizes = data.get("dataset_sizes", {})

            for repo in dataset_repos:
                logger.info(f"Creating split for {repo}...")
                # Use pre-loaded dataset size from client
                total_samples = dataset_sizes.get(repo, 0)
                if total_samples == 0:
                    logger.warning(f"No size info for {repo}, skipping")
                    continue

                # Calculate split sizes
                benchmark_size = int(total_samples * benchmark_pct)
                train_size = total_samples - benchmark_size

                # Generate random indices
                all_indices = list(range(total_samples))
                random.shuffle(all_indices)

                # Split indices
                benchmark_indices[repo] = sorted(all_indices[:benchmark_size])
                train_indices[repo] = sorted(all_indices[benchmark_size:])

                train_total += train_size
                benchmark_total += benchmark_size

                logger.info(f"  {repo}: {train_size} train, {benchmark_size} benchmark")

            # Save split to state manager
            metadata = {
                "seed": seed,
                "benchmark_percentage": benchmark_pct,
                "stratify": stratify,
                "total_train_samples": train_total,
                "total_benchmark_samples": benchmark_total
            }

            split_id = state_manager.create_split(
                dataset_repos=dataset_repos,
                train_indices=train_indices,
                benchmark_indices=benchmark_indices,
                metadata=metadata
            )

            logger.info(f"✅ Split created: {split_id}")

            return jsonify({
                "split_id": split_id,
                "train_samples": train_total,
                "benchmark_samples": benchmark_total,
                "metadata": metadata
            })

        except Exception as e:
            logger.exception("Failed to create split")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/benchmark/run", methods=["POST"])
    def run_benchmark_evaluation():
        """Start a benchmark evaluation job."""
        try:
            data = request.json or {}

            required = ["split_id", "dataset_repos", "model_path", "hub_model_id"]
            for field in required:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Get split metadata to include seed/percentage
            split_data = state_manager.get_split(data["split_id"])
            if not split_data:
                return jsonify({"error": f"Split {data['split_id']} not found"}), 404

            config = {
                "split_id": data["split_id"],
                "dataset_repos": data["dataset_repos"],
                "model_path": data["model_path"],
                "hub_model_id": data["hub_model_id"],
                "checkpoint_name": data.get("checkpoint_name", "final"),
                "seed": split_data["metadata"].get("seed"),
                "benchmark_percentage": split_data["metadata"].get("benchmark_percentage")
            }

            job_id = state_manager.create_job("benchmark", config)

            # Run benchmark in background
            thread = Thread(
                target=_run_benchmark,
                args=(benchmark_runner, state_manager, job_id),
                daemon=True
            )
            thread.start()

            return jsonify({
                "job_id": job_id,
                "message": "Benchmark started",
                "status": "in_progress"
            })

        except Exception as e:
            logger.exception("Failed to start benchmark")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/inference/instruct", methods=["POST"])
    def inference_instruct():
        """Run instruction-based inference (Q&A, custom instructions)."""

        try:
            data = request.get_json()

            # Validate required fields
            if not data.get("document"):
                return jsonify({"error": "document field is required"}), 400
            if not data.get("instruction"):
                return jsonify({"error": "instruction field is required"}), 400

            document = data["document"]
            instruction = data["instruction"]
            model_path = data.get("model_path", "./models/deepsynth-qa")

            # Generation parameters
            max_length = data.get("max_length", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            num_beams = data.get("num_beams", 4)

            # Use cached model for 10x speedup (3s → 50ms)
            from deepsynth.inference.model_cache import get_cached_model
            from deepsynth.inference.instruction_engine import GenerationParams

            logger.info(f"Getting cached model: {model_path}")
            engine = get_cached_model(model_path=model_path)

            # Create generation params
            params = GenerationParams(
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
            )

            # Generate answer
            logger.info(f"Generating answer for instruction: {instruction[:50]}...")
            result = engine.generate(document, instruction, params)

            return jsonify({
                "answer": result.answer,
                "tokens_generated": result.tokens_generated,
                "inference_time_ms": round(result.inference_time_ms, 2),
                "confidence": result.confidence,
                "metadata": result.metadata,
            })

        except Exception as e:
            logger.exception("Inference failed")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/inference/cache/stats", methods=["GET"])
    def inference_cache_stats():
        """Get model cache statistics."""

        try:
            from deepsynth.inference.model_cache import ModelCache

            cache = ModelCache.get_instance()
            stats = cache.get_stats()
            models = cache.get_loaded_models()

            return jsonify({
                "stats": stats,
                "loaded_models": models,
            })

        except Exception as e:
            logger.exception("Failed to get cache stats")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/inference/cache/clear", methods=["POST"])
    def inference_cache_clear():
        """Clear model cache."""

        try:
            data = request.get_json() or {}
            model_path = data.get("model_path")

            from deepsynth.inference.model_cache import ModelCache

            cache = ModelCache.get_instance()
            cache.clear_cache(model_path)

            return jsonify({
                "message": "Cache cleared successfully",
                "model_path": model_path or "all",
            })

        except Exception as e:
            logger.exception("Failed to clear cache")
            return jsonify({"error": str(e)}), 500


def _run_dataset_generation(
    generator: IncrementalDatasetGenerator,
    state_manager: StateManager,
    job_id: str,
) -> None:
    """Background worker for dataset generation."""

    try:
        generator.generate_dataset(job_id)
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED.value
            state_manager.update_job(job)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Dataset generation failed")
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED.value
            job.last_error = str(exc)
            state_manager.update_job(job)


def _run_qa_generation(
    generator: IncrementalDatasetGenerator,
    state_manager: StateManager,
    job_id: str,
) -> None:
    """Background worker for Q&A dataset generation."""

    try:
        generator.generate_qa_dataset(job_id)
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED.value
            state_manager.update_job(job)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Q&A dataset generation failed")
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED.value
            job.last_error = str(exc)
            state_manager.update_job(job)


def _run_model_training(
    trainer: ModelTrainer,
    state_manager: StateManager,
    job_id: str,
) -> None:
    """Background worker for model training."""

    try:
        trainer.train_model(job_id)
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED.value
            state_manager.update_job(job)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model training failed")
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED.value
            job.last_error = str(exc)
            state_manager.update_job(job)


def _run_benchmark(
    runner: BenchmarkRunner,
    state_manager: StateManager,
    job_id: str,
) -> None:
    """Background worker for benchmark evaluation."""

    try:
        runner.run_benchmark(job_id)
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED.value
            state_manager.update_job(job)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Benchmark evaluation failed")
        job = state_manager.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED.value
            job.last_error = str(exc)
            state_manager.update_job(job)


app = create_app()


__all__ = ["app", "create_app"]
