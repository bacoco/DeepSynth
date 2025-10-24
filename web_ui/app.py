"""
Flask web application for dataset generation and model training.
Provides a minimalist UI with real-time monitoring and job management.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from threading import Thread
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_ui.state_manager import StateManager, JobStatus
from web_ui.dataset_generator import IncrementalDatasetGenerator, ModelTrainer
from training.optimal_configs import (
    list_benchmark_datasets,
    get_optimal_config,
    PRESET_CONFIGS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize state manager
state_manager = StateManager()
dataset_generator = IncrementalDatasetGenerator(state_manager)
model_trainer = ModelTrainer(state_manager)


@app.route('/')
def index():
    """Render main UI."""
    return render_template('index.html')


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs."""
    job_type = request.args.get('type')
    jobs = state_manager.list_jobs(job_type)

    return jsonify({
        'jobs': [
            {
                'job_id': job.job_id,
                'job_type': job.job_type,
                'status': job.status,
                'created_at': job.created_at,
                'updated_at': job.updated_at,
                'progress': state_manager.get_progress(job.job_id)
            }
            for job in jobs
        ]
    })


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get job details."""
    job = state_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    progress = state_manager.get_progress(job_id)

    return jsonify({
        'job': {
            'job_id': job.job_id,
            'job_type': job.job_type,
            'status': job.status,
            'config': job.config,
            'created_at': job.created_at,
            'updated_at': job.updated_at,
            'hf_dataset_repo': job.hf_dataset_repo,
            'model_output_path': job.model_output_path,
        },
        'progress': progress
    })


@app.route('/api/jobs/<job_id>/progress', methods=['GET'])
def get_job_progress(job_id):
    """Get job progress."""
    progress = state_manager.get_progress(job_id)
    if not progress:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(progress)


@app.route('/api/dataset/generate', methods=['POST'])
def generate_dataset():
    """Start dataset generation job."""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['source_dataset', 'text_field', 'summary_field', 'output_dir']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Create job configuration
        config = {
            'source_dataset': data['source_dataset'],
            'source_subset': data.get('source_subset'),
            'source_split': data.get('source_split', 'train'),
            'text_field': data['text_field'],
            'summary_field': data['summary_field'],
            'output_dir': data['output_dir'],
            'max_samples': data.get('max_samples'),
            'dataset_name': data.get('dataset_name', 'generated-dataset'),
            'hf_username': data.get('hf_username', os.environ.get('HF_USERNAME')),
            'hf_dataset_repo': data.get('hf_dataset_repo'),
            'private_dataset': data.get('private_dataset', False)
        }

        # Create job
        job_id = state_manager.create_job('dataset_generation', config)

        # Start generation in background
        thread = Thread(
            target=_run_dataset_generation,
            args=(job_id,),
            daemon=True
        )
        thread.start()

        return jsonify({
            'job_id': job_id,
            'message': 'Dataset generation started',
            'status': 'in_progress'
        })

    except Exception as e:
        logger.error(f"Error starting dataset generation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Start model training job."""
    try:
        data = request.json

        # Validate required fields
        if 'dataset_repo' not in data:
            return jsonify({'error': 'Missing required field: dataset_repo'}), 400

        # Create job configuration
        config = {
            'dataset_repo': data['dataset_repo'],
            'model_name': data.get('model_name', 'deepseek-ai/DeepSeek-OCR'),
            'output_dir': data.get('output_dir', './trained_model'),
            'batch_size': int(data.get('batch_size', 2)),
            'num_epochs': int(data.get('num_epochs', 1)),
            'learning_rate': data.get('learning_rate', '2e-5'),
            'max_length': int(data.get('max_length', 512)),
            'mixed_precision': data.get('mixed_precision', 'bf16'),
            'gradient_accumulation_steps': int(data.get('gradient_accumulation_steps', 4)),
            'push_to_hub': data.get('push_to_hub', False),
            'hub_model_id': data.get('hub_model_id'),
            'hf_token': data.get('hf_token', os.environ.get('HF_TOKEN'))
        }

        # Create job
        job_id = state_manager.create_job('model_training', config)

        # Start training in background
        thread = Thread(
            target=_run_model_training,
            args=(job_id,),
            daemon=True
        )
        thread.start()

        return jsonify({
            'job_id': job_id,
            'message': 'Model training started',
            'status': 'in_progress'
        })

    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs/<job_id>/resume', methods=['POST'])
def resume_job(job_id):
    """Resume a paused or failed job."""
    job = state_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.status not in [JobStatus.PAUSED.value, JobStatus.FAILED.value]:
        return jsonify({'error': 'Job cannot be resumed'}), 400

    # Resume based on job type
    if job.job_type == 'dataset_generation':
        thread = Thread(
            target=_run_dataset_generation,
            args=(job_id,),
            daemon=True
        )
        thread.start()
    elif job.job_type == 'model_training':
        thread = Thread(
            target=_run_model_training,
            args=(job_id,),
            daemon=True
        )
        thread.start()
    else:
        return jsonify({'error': 'Unknown job type'}), 400

    return jsonify({
        'job_id': job_id,
        'message': 'Job resumed',
        'status': 'in_progress'
    })


@app.route('/api/jobs/<job_id>/pause', methods=['POST'])
def pause_job(job_id):
    """Pause a running job."""
    job = state_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.status != JobStatus.IN_PROGRESS.value:
        return jsonify({'error': 'Job is not running'}), 400

    job.status = JobStatus.PAUSED.value
    state_manager.update_job(job)

    return jsonify({
        'job_id': job_id,
        'message': 'Job paused',
        'status': 'paused'
    })


@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job."""
    job = state_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    state_manager.delete_job(job_id)

    return jsonify({
        'job_id': job_id,
        'message': 'Job deleted'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'jobs_count': len(state_manager.list_jobs())
    })


@app.route('/api/datasets/presets', methods=['GET'])
def get_dataset_presets():
    """Get available dataset presets for benchmarking."""
    datasets = list_benchmark_datasets()
    return jsonify({'datasets': datasets})


@app.route('/api/training/presets', methods=['GET'])
def get_training_presets():
    """Get optimal training configuration presets."""
    presets = {
        name: config.to_dict()
        for name, config in PRESET_CONFIGS.items()
    }
    return jsonify({'presets': presets})


@app.route('/api/training/optimal-config/<preset>', methods=['GET'])
def get_optimal_training_config(preset):
    """Get optimal configuration for a specific preset."""
    config = get_optimal_config(preset)
    if config:
        return jsonify({'config': config.to_dict()})
    return jsonify({'error': 'Preset not found'}), 404


@app.route('/api/benchmark/create', methods=['POST'])
def create_benchmark_dataset():
    """Create a benchmark dataset for evaluation."""
    try:
        data = request.json

        # Validate required fields
        if 'benchmark_name' not in data:
            return jsonify({'error': 'Missing required field: benchmark_name'}), 400

        benchmark_info = list_benchmark_datasets().get(data['benchmark_name'])
        if not benchmark_info:
            return jsonify({'error': 'Unknown benchmark dataset'}), 400

        # Create job configuration for benchmark dataset
        config = {
            'source_dataset': benchmark_info['name'],
            'source_subset': benchmark_info['subset'],
            'source_split': data.get('split', 'train'),
            'text_field': benchmark_info['text_field'],
            'summary_field': benchmark_info['summary_field'],
            'output_dir': data.get('output_dir', f'./benchmarks/{data["benchmark_name"]}'),
            'max_samples': data.get('max_samples'),
            'dataset_name': f'{data["benchmark_name"]}-benchmark',
            'hf_username': data.get('hf_username', os.environ.get('HF_USERNAME')),
            'hf_dataset_repo': data.get('hf_dataset_repo'),
            'private_dataset': data.get('private_dataset', False),
            'is_benchmark': True
        }

        # Create job
        job_id = state_manager.create_job('dataset_generation', config)

        # Start generation in background
        thread = Thread(
            target=_run_dataset_generation,
            args=(job_id,),
            daemon=True
        )
        thread.start()

        return jsonify({
            'job_id': job_id,
            'message': 'Benchmark dataset generation started',
            'status': 'in_progress',
            'benchmark_info': benchmark_info
        })

    except Exception as e:
        logger.error(f"Error creating benchmark dataset: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/<job_id>', methods=['GET'])
def get_job_metrics(job_id):
    """Get metrics for a completed training job."""
    job = state_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.job_type != 'model_training':
        return jsonify({'error': 'Metrics only available for training jobs'}), 400

    # Get metrics from job state
    metrics = job.config.get('metrics', {})

    return jsonify({
        'job_id': job_id,
        'metrics': metrics,
        'model_output_path': job.model_output_path
    })


def _run_dataset_generation(job_id: str):
    """Background worker for dataset generation."""
    try:
        logger.info(f"Starting dataset generation for job {job_id}")
        dataset_generator.generate_dataset(job_id)
        logger.info(f"Dataset generation completed for job {job_id}")
    except Exception as e:
        logger.error(f"Dataset generation failed for job {job_id}: {e}")


def _run_model_training(job_id: str):
    """Background worker for model training."""
    try:
        logger.info(f"Starting model training for job {job_id}")
        model_trainer.train_model(job_id)
        logger.info(f"Model training completed for job {job_id}")
    except Exception as e:
        logger.error(f"Model training failed for job {job_id}: {e}")


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=True, threaded=True)
