"""
Incremental dataset generator with deduplication support.
Handles resumable dataset generation and HuggingFace Hub uploads.

IMPROVED VERSION: Added multi-trainer support with MoE dropout configuration.
"""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional

from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from datasets.features import Image as HFImage
from huggingface_hub import HfApi, create_repo

from deepsynth.data.text_to_image import TextToImageConverter
from deepsynth.training.config import OptimizerConfig, TrainerConfig
from .state_manager import JobStatus, StateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncrementalDatasetGenerator:
    """Generates datasets incrementally with deduplication."""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.converter = TextToImageConverter()
        self.hf_api = HfApi()

    def generate_dataset(
        self,
        job_id: str,
        progress_callback: Optional[Callable] = None
    ):
        """
        Generate dataset incrementally with resumption support.

        Args:
            job_id: Job identifier
            progress_callback: Optional callback for progress updates
        """
        job = self.state_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Update status
        job.status = JobStatus.IN_PROGRESS.value
        self.state_manager.update_job(job)

        try:
            config = job.config

            # Load source dataset
            logger.info(f"Loading source dataset: {config['source_dataset']}")
            source_ds = self._load_source_dataset(
                config['source_dataset'],
                config.get('source_subset'),
                config.get('source_split', 'train')
            )

            # Get existing processed hashes
            processed_hashes = self.state_manager.get_processed_hashes(job_id)
            logger.info(f"Already processed: {len(processed_hashes)} samples")

            # Set total samples
            max_samples = config.get('max_samples')
            total_samples = len(source_ds) if not max_samples else min(max_samples, len(source_ds))
            job.total_samples = total_samples
            self.state_manager.update_job(job)

            # Process samples incrementally
            processed_data = []
            text_field = config.get('text_field', 'article')
            summary_field = config.get('summary_field', 'highlights')

            # Check if we need to load existing dataset
            existing_data = []
            if job.hf_dataset_repo:
                existing_data = self._load_existing_dataset(job.hf_dataset_repo)
                logger.info(f"Loaded {len(existing_data)} existing samples from HF")

            for idx, sample in enumerate(source_ds):
                # Check if we've reached max samples
                if max_samples and job.processed_samples >= max_samples:
                    break

                try:
                    # Get text content
                    text_content = sample.get(text_field, '')
                    if not text_content:
                        continue

                    # Check for duplicates
                    if self.state_manager.is_duplicate(job_id, text_content):
                        logger.debug(f"Skipping duplicate sample {idx}")
                        continue

                    # Generate image
                    image_filename = f"image_{job.processed_samples:06d}.png"
                    image_path = Path(config['output_dir']) / image_filename

                    # Ensure output directory exists
                    image_path.parent.mkdir(parents=True, exist_ok=True)

                    # Convert text to image
                    self.converter.text_to_image(text_content, str(image_path))

                    # Create processed sample
                    processed_sample = {
                        'text': text_content,
                        'summary': sample.get(summary_field, ''),
                        'image_path': str(image_path),
                        'image': str(image_path),
                        'source_index': idx
                    }
                    processed_data.append(processed_sample)

                    # Mark as processed
                    self.state_manager.mark_processed(job_id, text_content)

                    # Update progress
                    job.processed_samples += 1
                    self.state_manager.update_job(job)

                    # Progress callback
                    if progress_callback:
                        progress_callback(job.processed_samples, total_samples)

                    # Periodic save to HF (every 100 samples)
                    if len(processed_data) >= 100:
                        self._save_to_hf(
                            job,
                            existing_data + processed_data,
                            config
                        )
                        existing_data.extend(processed_data)
                        processed_data = []

                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {e}")
                    job.failed_samples += 1
                    job.error_count += 1
                    job.last_error = str(e)
                    self.state_manager.update_job(job)
                    continue

            # Final save to HF
            if processed_data:
                self._save_to_hf(
                    job,
                    existing_data + processed_data,
                    config
                )

            # Mark job as completed
            job.status = JobStatus.COMPLETED.value
            self.state_manager.update_job(job)

            logger.info(f"Dataset generation completed. Processed: {job.processed_samples}")

        except Exception as e:
            logger.error(f"Job failed: {e}")
            job.status = JobStatus.FAILED.value
            job.last_error = str(e)
            self.state_manager.update_job(job)
            raise

    def _load_source_dataset(self, dataset_name: str, subset: Optional[str], split: str):
        """Load source dataset from HuggingFace."""
        if subset:
            return load_dataset(dataset_name, subset, split=split)
        return load_dataset(dataset_name, split=split)

    def _load_existing_dataset(self, repo_id: str):
        """Load existing dataset from HuggingFace Hub."""
        try:
            ds = load_dataset(repo_id, split='train', token=os.environ.get('HF_TOKEN'))
            return list(ds)
        except Exception as e:
            logger.warning(f"Could not load existing dataset: {e}")
            return []

    def _save_to_hf(self, job, data: list, config: Dict):
        """Save dataset to HuggingFace Hub."""
        if not data:
            return

        try:
            # Create dataset with explicit image feature so files are uploaded to HF
            features = Features(
                {
                    'text': Value('string'),
                    'summary': Value('string'),
                    'image_path': Value('string'),
                    'image': HFImage(),
                    'source_index': Value('int32'),
                }
            )

            dataset = Dataset.from_list(data, features=features)

            # Create or update repo
            repo_id = config.get('hf_dataset_repo')
            if not repo_id:
                hf_username = config.get('hf_username', os.environ.get('HF_USERNAME'))
                dataset_name = config.get('dataset_name', 'generated-dataset')
                repo_id = f"{hf_username}/{dataset_name}"

            # Create repo if it doesn't exist
            token = config.get('hf_token', os.environ.get('HF_TOKEN'))

            try:
                create_repo(
                    repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                    token=token,
                    private=config.get('private_dataset', False)
                )
            except Exception as e:
                logger.warning(f"Repo might already exist: {e}")

            # Push to hub
            dataset.push_to_hub(
                repo_id,
                private=config.get('private_dataset', False),
                token=token
            )

            # Update job with repo info
            job.hf_dataset_repo = repo_id
            self.state_manager.update_job(job)

            logger.info(f"Saved {len(data)} samples to {repo_id}")

        except Exception as e:
            logger.error(f"Error saving to HuggingFace Hub: {e}")
            raise


class ModelTrainer:
    """Handles model training with resumption support and multi-trainer selection."""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def train_model(
        self,
        job_id: str,
        progress_callback: Optional[Callable] = None
    ):
        """
        Train model with checkpoint resumption.

        Args:
            job_id: Job identifier
            progress_callback: Optional callback for progress updates
        """
        job = self.state_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Update status
        job.status = JobStatus.IN_PROGRESS.value
        self.state_manager.update_job(job)

        try:
            config = job.config

            # Determine trainer type
            trainer_type = config.get('trainer_type', 'production')
            logger.info(f"Using trainer type: {trainer_type}")

            # Create trainer config
            optimizer_config = OptimizerConfig(
                learning_rate=float(config.get('learning_rate', 2e-5)),
                weight_decay=float(config.get('weight_decay', 0.01)),
                warmup_steps=int(config.get('warmup_steps', 0)),
            )

            trainer_config = TrainerConfig(
                model_name=config.get('model_name', 'deepseek-ai/DeepSeek-OCR'),
                output_dir=config.get('output_dir', './trained_model'),
                batch_size=int(config.get('batch_size', 2)),
                num_epochs=int(config.get('num_epochs', 1)),
                max_length=int(config.get('max_length', 512)),
                mixed_precision=config.get('mixed_precision', 'bf16'),
                gradient_accumulation_steps=int(config.get('gradient_accumulation_steps', 4)),
                optimizer=optimizer_config,
                push_to_hub=config.get('push_to_hub', False),
                hub_model_id=config.get('hub_model_id'),
                hub_private=bool(config.get('hub_private', False)),
                hub_token=config.get('hf_token', os.environ.get('HF_TOKEN')),
                evaluation_split=config.get('evaluation_split'),
                save_checkpoints_to_hub=bool(config.get('save_checkpoints_to_hub', True)),
                resume_from_checkpoint=config.get('resume_from_checkpoint'),
                save_metrics_to_hub=bool(config.get('save_metrics_to_hub', True)),
                # MoE dropout parameters (NEW)
                expert_dropout_rate=float(config.get('expert_dropout_rate', 0.0)),
                expert_dropout_min_keep=int(config.get('expert_dropout_min_keep', 1)),
                gate_dropout_rate=float(config.get('gate_dropout_rate', 0.0)),
                bidrop_passes=int(config.get('bidrop_passes', 1)),
            )

            # Load dataset
            dataset_repo = config.get('dataset_repo')
            if not dataset_repo:
                raise ValueError("dataset_repo is required for training")

            logger.info(f"Loading training dataset from {dataset_repo}")

            # Create trainer based on type
            if trainer_type == 'production':
                from deepsynth.training.deepsynth_trainer_v2 import ProductionDeepSynthTrainer
                trainer = ProductionDeepSynthTrainer(trainer_config)
                logger.info(f"Using ProductionDeepSynthTrainer with MoE dropout support")
                logger.info(f"  - Expert dropout: {trainer_config.expert_dropout_rate}")
                logger.info(f"  - Gate dropout: {trainer_config.gate_dropout_rate}")
                logger.info(f"  - Bi-Drop passes: {trainer_config.bidrop_passes}")
            elif trainer_type == 'deepsynth':
                from deepsynth.training.deepsynth_trainer import DeepSynthOCRTrainer
                trainer = DeepSynthOCRTrainer(trainer_config)
                logger.info("Using DeepSynthOCRTrainer (basic frozen encoder)")
            elif trainer_type == 'generic':
                from deepsynth.training.trainer import SummarizationTrainer
                trainer = SummarizationTrainer(trainer_config)
                logger.info("Using SummarizationTrainer (generic seq2seq)")
            else:
                raise ValueError(f"Unknown trainer type: {trainer_type}")

            # Train
            logger.info("Starting training...")
            def progress_callback_wrapper(processed: int, total: int):
                job_state = self.state_manager.get_job(job_id)
                if not job_state:
                    return
                job_state.total_samples = total or job_state.total_samples
                job_state.processed_samples = processed
                self.state_manager.update_job(job_state)

            # Use appropriate training method based on trainer type
            if trainer_type == 'production':
                metrics, checkpoints = trainer.train_from_hf_dataset(
                    dataset_repo,
                    progress_callback=progress_callback_wrapper
                )
            else:
                # For other trainers, we need to load the dataset and call train()
                dataset = load_dataset(dataset_repo, split='train')
                trainer.train(dataset)

                # Create compatible return values
                metrics = {}
                checkpoints = {'last_checkpoint': trainer_config.output_dir}

            # Update job
            job.status = JobStatus.COMPLETED.value
            job.model_output_path = trainer_config.output_dir
            job.training_checkpoint = checkpoints.get('last_checkpoint')
            job.config['metrics'] = metrics
            job.config['checkpoints'] = checkpoints
            job.config['metrics_path'] = str(Path(trainer_config.output_dir) / 'metrics.json')
            job.config['trainer_type'] = trainer_type
            if trainer_config.push_to_hub:
                job.config['hub_model_id'] = trainer_config.hub_model_id
            self.state_manager.update_job(job)

            logger.info(f"Training completed successfully with {trainer_type} trainer")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            job.status = JobStatus.FAILED.value
            job.last_error = str(e)
            self.state_manager.update_job(job)
            raise
