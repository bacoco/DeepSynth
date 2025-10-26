"""Flask web application for dataset generation and training jobs."""

from __future__ import annotations

import logging
import os
from threading import Thread
from typing import Optional

from flask import Flask, jsonify, render_template, request

from apps.web.config import WebConfig
from deepsynth.training.optimal_configs import (
    PRESET_CONFIGS,
    get_optimal_config,
    list_benchmark_datasets,
)

from .dataset_generator_improved import IncrementalDatasetGenerator, ModelTrainer
from .state_manager import JobStatus, StateManager

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

    _register_routes(app, state_manager, dataset_generator, model_trainer)

    app.state_manager = state_manager  # type: ignore[attr-defined]
    app.dataset_generator = dataset_generator  # type: ignore[attr-defined]
    app.model_trainer = model_trainer  # type: ignore[attr-defined]

    return app


def _register_routes(
    app: Flask,
    state_manager: StateManager,
    dataset_generator: IncrementalDatasetGenerator,
    model_trainer: ModelTrainer,
) -> None:
    """Bind HTTP routes to the Flask application."""

    @app.route("/")
    def index():
        """Render the main UI."""

        return render_template("index_improved.html")

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

    @app.route("/api/datasets/benchmarks", methods=["GET"])
    def list_benchmarks():
        """Return available benchmark datasets."""

        return jsonify({"benchmarks": list_benchmark_datasets()})

    @app.route("/api/datasets/presets", methods=["GET"])
    def list_presets():
        """Return available preset configurations."""

        return jsonify({"presets": PRESET_CONFIGS})

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


app = create_app()


__all__ = ["app", "create_app"]
