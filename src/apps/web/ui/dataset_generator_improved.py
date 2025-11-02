"""
Incremental dataset generator with deduplication support.
Handles resumable dataset generation and HuggingFace Hub uploads.

IMPROVED VERSION: Added multi-trainer support with MoE dropout configuration.
"""

import os
from pathlib import Path
from typing import Callable, Dict, Optional
import logging
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from datasets.features import Image as HFImage
from huggingface_hub import HfApi, create_repo

from deepsynth.data.transforms.text_to_image import (
    DEEPSEEK_OCR_RESOLUTIONS,
    TextToImageConverter,
)
from deepsynth.data.dataset_converters import convert_natural_questions, convert_ms_marco
from deepsynth.data.hub.shards import HubShardManager
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
            job.status_message = f"Loaded source dataset ({total_samples} samples). Starting generation…"
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

                    # Ensure output directory exists
                    output_dir = Path(config['output_dir'])
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Apply instruction prompting if configured
                    instruction_prompt = config.get('instruction_prompt', '')
                    if instruction_prompt:
                        # Prepend instruction to text for image generation
                        display_text = f"{instruction_prompt}\n\n{text_content}"
                        logger.debug(f"Applied instruction prompt: {instruction_prompt[:50]}...")
                    else:
                        display_text = text_content

                    # Check if multi-resolution is enabled
                    multi_resolution = config.get('multi_resolution', False)
                    resolution_sizes = config.get('resolution_sizes', None)

                    # Generate images
                    if multi_resolution:
                        filtered_sizes = []
                        for size in resolution_sizes or []:
                            if size in DEEPSEEK_OCR_RESOLUTIONS and size not in filtered_sizes:
                                filtered_sizes.append(size)
                        selected_sizes = filtered_sizes or list(DEEPSEEK_OCR_RESOLUTIONS.keys())
                        config['resolution_sizes'] = selected_sizes

                        sizes_dict = {
                            name: DEEPSEEK_OCR_RESOLUTIONS[name]
                            for name in selected_sizes
                        }

                        multi_res_images = self.converter.convert_multi_resolution(
                            display_text,
                            sizes=sizes_dict
                        )

                        # Create processed sample with multi-resolution images
                        base_image = multi_res_images.get('original')
                        if base_image is None:
                            base_image = self.converter.convert(display_text)

                        processed_sample = {
                            'text': text_content,  # Store original text
                            'display_text': display_text,  # Store text with instruction
                            'instruction_prompt': instruction_prompt,  # Store instruction separately
                            'summary': sample.get(summary_field, ''),
                            'image': base_image,
                            'source_index': idx
                        }

                        # Add resolution images
                        for res_name in selected_sizes:
                            if res_name in multi_res_images:
                                processed_sample[f'image_{res_name}'] = multi_res_images[res_name]

                    else:
                        # Single resolution mode
                        image = self.converter.convert(display_text)

                        # Create processed sample
                        processed_sample = {
                            'text': text_content,  # Store original text
                            'display_text': display_text,  # Store text with instruction
                            'instruction_prompt': instruction_prompt,  # Store instruction separately
                            'summary': sample.get(summary_field, ''),
                            'image': image,
                            'source_index': idx
                        }
                        config['resolution_sizes'] = None

                    processed_data.append(processed_sample)

                    # Mark as processed
                    self.state_manager.mark_processed(job_id, text_content)

                    # Update progress
                    job.processed_samples += 1
                    job.status_message = f"Generating dataset: {job.processed_samples}/{total_samples} processed"
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
            job.status_message = "Dataset generation completed"
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
            # Build features dict based on multi-resolution setting
            multi_resolution = config.get('multi_resolution', False)

            features_dict = {
                'text': Value('string'),
                'summary': Value('string'),
                'image': HFImage(),
                'source_index': Value('int32'),
            }

            # Add multi-resolution image features if enabled
            if multi_resolution:
                filtered_sizes = []
                for size in config.get('resolution_sizes') or []:
                    if size in DEEPSEEK_OCR_RESOLUTIONS and size not in filtered_sizes:
                        filtered_sizes.append(size)
                selected_sizes = filtered_sizes or list(DEEPSEEK_OCR_RESOLUTIONS.keys())
                for res_name in selected_sizes:
                    features_dict[f'image_{res_name}'] = HFImage()

            features = Features(features_dict)

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

    def generate_qa_dataset(
        self,
        job_id: str,
        progress_callback: Optional[Callable] = None
    ):
        """Generate Q&A dataset (Natural Questions + MS MARCO) incrementally."""
        job = self.state_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = JobStatus.IN_PROGRESS.value
        self.state_manager.update_job(job)

        manager: Optional[HubShardManager] = None

        try:
            config = job.config or {}

            qa_sources = config.get('qa_sources', ['natural_questions', 'ms_marco'])
            target_resolution = config.get('target_resolution', 'gundam')
            max_nq_samples = config.get('max_nq_samples')
            max_marco_samples = config.get('max_marco_samples')
            batch_size = int(config.get('batch_size', 5000))

            logger.info(f"Generating Q&A dataset with sources: {qa_sources}")
            job.status_message = "Preparing Q&A dataset generation…"
            self.state_manager.update_job(job)
            logger.info(f"Target resolution: {target_resolution}")
            logger.info(f"Batch size: {batch_size}")

            default_estimates = {
                'natural_questions': 300_000,
                'ms_marco': 1_000_000,
            }
            estimated_total = 0
            if 'natural_questions' in qa_sources:
                estimated_total += max_nq_samples if max_nq_samples else default_estimates['natural_questions']
            if 'ms_marco' in qa_sources:
                estimated_total += max_marco_samples if max_marco_samples else default_estimates['ms_marco']
            if estimated_total == 0:
                estimated_total = sum(default_estimates.get(source, 0) for source in qa_sources) or sum(default_estimates.values())
            job.total_samples = estimated_total
            job.status_message = f"Q&A generation started; target ~{estimated_total} samples"
            self.state_manager.update_job(job)

            hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')
            if not hf_token:
                raise ValueError("HF_TOKEN is required to upload the Q&A dataset")

            repo_id = config.get('repo_id')
            dataset_name = config.get('dataset_name', 'deepsynth-qa')
            if not repo_id:
                if '/' in dataset_name:
                    repo_id = dataset_name
                else:
                    username = config.get('hf_username') or os.environ.get('HF_USERNAME')
                    if not username:
                        raise ValueError(
                            "Either provide 'repo_id', a dataset name with owner (e.g. 'org/dataset'),"
                            " or set HF_USERNAME to build the repository id."
                        )
                    repo_id = f"{username}/{dataset_name}"

            manager = HubShardManager(repo_id=repo_id, token=hf_token)

            total_uploaded = 0

            def flush_batch(batch: list, source_name: str) -> None:
                nonlocal total_uploaded
                if not batch:
                    return

                shard_id = manager.next_shard_id()
                commit_message = f"Add {source_name} shard {shard_id} ({len(batch)} samples)"
                result = manager.upload_samples_as_shard(batch, shard_id, commit_message=commit_message)
                logger.info(
                    "Uploaded shard %s: %s new samples (%s duplicates)",
                    shard_id,
                    result.uploaded_samples,
                    result.skipped_duplicates,
                )

                if result.uploaded_samples:
                    total_uploaded += result.uploaded_samples
                    job.processed_samples = total_uploaded
                    if not job.hf_dataset_repo:
                        job.hf_dataset_repo = repo_id
                    job.status_message = f"Uploading Q&A shards: {total_uploaded}/{job.total_samples} samples uploaded"
                    self.state_manager.update_job(job)
                    if progress_callback:
                        total_target = job.total_samples or total_uploaded
                        progress_callback(job.processed_samples, total_target)

                if result.index_updated:
                    try:
                        manager.save_index(f"Update shard index after {shard_id}")
                    except Exception as index_err:  # pragma: no cover - defensive logging
                        logger.warning("Failed to persist shard index for %s: %s", shard_id, index_err)

                batch.clear()

            # Process Natural Questions first if selected
            if 'natural_questions' in qa_sources:
                logger.info("Processing Natural Questions...")
                nq_dataset = convert_natural_questions(
                    split="train",
                    max_samples=max_nq_samples,
                    streaming=True,
                    target_resolution=target_resolution,
                )

                nq_batch: list = []
                for sample in nq_dataset:
                    nq_batch.append(sample)
                    if len(nq_batch) >= batch_size:
                        flush_batch(nq_batch, 'natural_questions')

                # Upload remaining
                flush_batch(nq_batch, 'natural_questions')

            # Process MS MARCO second for faster startup
            if 'ms_marco' in qa_sources:
                logger.info("Processing MS MARCO...")
                marco_dataset = convert_ms_marco(
                    config="v2.1",
                    split="train",
                    max_samples=max_marco_samples,
                    streaming=True,
                    target_resolution=target_resolution,
                )

                marco_batch: list = []
                for sample in marco_dataset:
                    marco_batch.append(sample)
                    if len(marco_batch) >= batch_size:
                        flush_batch(marco_batch, 'ms_marco')

                flush_batch(marco_batch, 'ms_marco')

            job.status = JobStatus.COMPLETED.value
            job.processed_samples = total_uploaded
            self.state_manager.update_job(job)

            logger.info(f"Q&A dataset generation completed. Total uploaded: {total_uploaded}")

        except Exception as e:
            logger.error(f"Q&A generation failed: {e}")
            job.status = JobStatus.FAILED.value
            job.last_error = str(e)
            self.state_manager.update_job(job)
            raise
        finally:
            if manager is not None:
                try:
                    manager.save_index("Persist shard index after QA generation")
                except Exception as index_err:  # pragma: no cover - defensive logging
                    logger.warning("Failed to persist shard index: %s", index_err)


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

            # Apply upload backend environment controls (affect production_trainer behavior)
            try:
                ds_push_backend = str(config.get('ds_push_backend') or '').strip().lower()
                if ds_push_backend in ('git', 'http'):
                    os.environ['DS_PUSH_BACKEND'] = ds_push_backend
                ds_upload_intermediate = bool(config.get('ds_upload_intermediate', False))
                os.environ['DS_UPLOAD_INTERMEDIATE'] = '1' if ds_upload_intermediate else '0'
            except Exception as _env_exc:
                logger.warning(f"Could not apply upload backend env toggles: {_env_exc}")

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
                # Quick test parameters (for streaming mode optimization)
                max_train_samples=int(config['max_train_samples']) if config.get('max_train_samples') else None,
                max_eval_samples=int(config['max_eval_samples']) if config.get('max_eval_samples') else None,
                # MoE dropout parameters (NEW)
                expert_dropout_rate=float(config.get('expert_dropout_rate', 0.0)),
                expert_dropout_min_keep=int(config.get('expert_dropout_min_keep', 1)),
                gate_dropout_rate=float(config.get('gate_dropout_rate', 0.0)),
                bidrop_passes=int(config.get('bidrop_passes', 1)),
            )

            # Load datasets (can be multiple)
            dataset_repos = config.get('dataset_repos')
            if not dataset_repos:
                # Fallback to single dataset_repo for backward compatibility
                dataset_repo = config.get('dataset_repo')
                if dataset_repo:
                    dataset_repos = [dataset_repo]
                else:
                    raise ValueError("dataset_repos is required for training")

            split_id = config.get('split_id')
            if split_id:
                # Use train split only (exclude benchmark)
                logger.info(f"Using split {split_id} for training (benchmark excluded)")
                train_indices = self.state_manager.get_split_indices(split_id, "train")

                if not train_indices:
                    raise ValueError(f"No train indices found for split_id: {split_id}")

                # Load and filter each dataset
                from datasets import concatenate_datasets, load_dataset

                train_datasets = []
                for repo in dataset_repos:
                    logger.info(f"Loading training samples from {repo}...")

                    # Get indices for this repo
                    repo_indices = train_indices.get(repo, [])
                    if not repo_indices:
                        continue

                    # OPTIMIZATION: For quick tests, limit the number of indices to load
                    max_train_samples = trainer_config.max_train_samples
                    if max_train_samples and max_train_samples < len(repo_indices):
                        logger.info(f"  Quick test mode: Limiting to first {max_train_samples} of {len(repo_indices)} indices")
                        repo_indices = repo_indices[:max_train_samples]

                    logger.info(f"  Loading {len(repo_indices)} samples from {repo}...")
                    ds = load_dataset(repo, split="train")
                    ds_filtered = ds.select(repo_indices)
                    logger.info(f"  ✅ Loaded {len(ds_filtered)} training samples from {repo}")

                    train_datasets.append(ds_filtered)

                if not train_datasets:
                    raise ValueError("No training samples loaded!")

                # Combine all datasets
                combined_dataset = concatenate_datasets(train_datasets)
                logger.info(f"Combined dataset: {len(combined_dataset)} total training samples")

            else:
                # No split_id: load entire datasets (backward compatibility)
                logger.warning("No split_id provided - using entire datasets (no benchmark split)")

                from datasets import concatenate_datasets, load_dataset

                train_datasets = []
                for repo in dataset_repos:
                    logger.info(f"Loading entire dataset from {repo}...")
                    ds = load_dataset(repo, split="train")
                    train_datasets.append(ds)
                    logger.info(f"  Loaded {len(ds)} samples from {repo}")

                combined_dataset = concatenate_datasets(train_datasets)
                logger.info(f"Combined dataset: {len(combined_dataset)} total samples")

            # Create trainer based on type
            if trainer_type == 'production':
                from deepsynth.training.production_trainer import UnifiedProductionTrainer
                trainer = UnifiedProductionTrainer(trainer_config)
                logger.info(f"Using UnifiedProductionTrainer with vision-to-text flow")
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
                job_state.status_message = f"Training in progress: {processed}/{total} steps"
                self.state_manager.update_job(job_state)

            # Train on combined dataset
            logger.info(f"Starting training on {len(combined_dataset)} samples...")

            # Initialize job progress totals early so UI doesn't flag job as stuck
            try:
                job_state = self.state_manager.get_job(job_id)
                if job_state:
                    job_state.total_samples = int(len(combined_dataset)) * int(trainer_config.num_epochs)
                    job_state.processed_samples = 0
                    job_state.status_message = "Initializing trainer and starting training…"
                    self.state_manager.update_job(job_state)
            except Exception as _init_exc:  # defensive: never block training due to progress init
                logger.warning(f"Could not initialize job progress totals: {_init_exc}")

            if trainer_type == 'production':
                # Production trainer has train() method that accepts dataset directly
                metrics, checkpoints = trainer.train(
                    combined_dataset,
                    progress_callback=progress_callback_wrapper
                )
            else:
                # Other trainers also use train() method
                trainer.train(combined_dataset)

                # Create compatible return values
                metrics = {}
                checkpoints = {'last_checkpoint': trainer_config.output_dir}

            # Push final model to HuggingFace if configured
            if trainer_config.push_to_hub and trainer_config.hub_model_id:
                logger.info(f"Pushing final model to {trainer_config.hub_model_id}...")
                # The trainers should handle this, but we can add it here for safety
                # Create model card with dataset info
                self._create_model_card(
                    hub_model_id=trainer_config.hub_model_id,
                    datasets=dataset_repos,
                    trainer_config=trainer_config,
                    metrics=metrics
                )

            # Update job
            job.status = JobStatus.COMPLETED.value
            job.status_message = "Training completed"
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

    def _create_model_card(
        self,
        hub_model_id: str,
        datasets: list,
        trainer_config,
        metrics: dict
    ):
        """Create initial model card on HuggingFace."""
        from huggingface_hub import HfApi

        # Determine languages from datasets
        languages = set()
        for ds in datasets:
            name = ds.split('/')[-1]
            if 'fr' in name:
                languages.add('fr')
            elif 'es' in name:
                languages.add('es')
            elif 'de' in name:
                languages.add('de')
            elif 'en' in name:
                languages.add('en')

        lang_list = '\n'.join(f'- {lang}' for lang in sorted(languages))
        datasets_list = '\n'.join(f'- {ds}' for ds in datasets)

        card_content = f"""---
language:
{lang_list}
license: apache-2.0
tags:
- deepseek-ocr
- summarization
- multilingual
- vision-language-model
datasets:
{datasets_list}
metrics:
- rouge
- bertscore
---

# {hub_model_id}

## Model Description

Fine-tuned DeepSeek-OCR model for multilingual document summarization.

This model uses a frozen visual encoder (380M parameters) and fine-tunes the MoE decoder (570M active parameters) for improved summarization across multiple languages.

## Training Data

{datasets_list}

Total training samples: {metrics.get('total_samples', 'N/A')}

## Training Configuration

- **Base Model**: {trainer_config.model_name}
- **Trainer Type**: Production (frozen encoder + fine-tuned decoder)
- **Batch Size**: {trainer_config.batch_size}
- **Epochs**: {trainer_config.num_epochs}
- **Learning Rate**: {trainer_config.optimizer.learning_rate}
- **Mixed Precision**: {trainer_config.mixed_precision}
- **Gradient Accumulation Steps**: {trainer_config.gradient_accumulation_steps}

## MoE Dropout Configuration

- **Expert Dropout Rate**: {trainer_config.expert_dropout_rate}
- **Gate Dropout Rate**: {trainer_config.gate_dropout_rate}
- **Bi-Drop Passes**: {trainer_config.bidrop_passes}

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("{hub_model_id}")
tokenizer = AutoTokenizer.from_pretrained("{hub_model_id}")

# For vision-based input (recommended for DeepSeek-OCR)
# See DeepSeek-OCR documentation for image processing

# For text-based input (fallback)
text = "Your document text here..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=128)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Benchmark Results

(To be updated after evaluation)

## Citation

```bibtex
@misc{{deepsynth2025,
  title={{DeepSynth: Multilingual Document Summarization}},
  author={{DeepSynth Team}},
  year={{2025}},
  url={{https://huggingface.co/{hub_model_id}}}
}}
```
"""

        try:
            api = HfApi(token=trainer_config.hub_token)
            api.upload_file(
                path_or_fileobj=card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=hub_model_id,
                repo_type="model",
                token=trainer_config.hub_token,
                commit_message="Initialize model card with training info"
            )
            logger.info(f"✅ Model card created for {hub_model_id}")
        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")
            # Don't fail training just because of model card
