"""
State Manager for resumable dataset generation and training jobs.
Handles job persistence, recovery, and deduplication tracking.
"""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class JobState:
    """State of a dataset generation or training job."""
    job_id: str
    job_type: str  # "dataset_generation" or "model_training"
    status: str
    created_at: str
    updated_at: str

    # Configuration
    config: Dict

    # Progress tracking
    total_samples: int = 0
    processed_samples: int = 0
    failed_samples: int = 0

    # Deduplication tracking
    processed_hashes: List[str] = None

    # Dataset info
    hf_dataset_repo: Optional[str] = None
    local_dataset_path: Optional[str] = None

    # Model training info
    model_output_path: Optional[str] = None
    training_checkpoint: Optional[str] = None

    # Error tracking
    last_error: Optional[str] = None
    error_count: int = 0
    # Human-readable status message for UI
    status_message: Optional[str] = None

    def __post_init__(self):
        if self.processed_hashes is None:
            self.processed_hashes = []


class StateManager:
    """Manages job state persistence and recovery."""

    def __init__(self, state_dir: Union[str, os.PathLike] = None):
        default_state_dir = Path(__file__).resolve().parent / "state"
        self.state_dir = Path(state_dir) if state_dir else default_state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = self.state_dir / "jobs.json"
        self.hashes_dir = self.state_dir / "hashes"
        self.hashes_dir.mkdir(exist_ok=True)
        self.splits_dir = self.state_dir / "splits"
        self.splits_dir.mkdir(exist_ok=True)

    def create_job(self, job_type: str, config: Dict) -> str:
        """Create a new job with unique ID."""
        job_id = self._generate_job_id(job_type)
        timestamp = datetime.utcnow().isoformat()

        job_state = JobState(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING.value,
            created_at=timestamp,
            updated_at=timestamp,
            config=config,
            processed_hashes=[]
        )

        self._save_job(job_state)
        return job_id

    def get_job(self, job_id: str) -> Optional[JobState]:
        """Retrieve job state by ID."""
        jobs = self._load_all_jobs()
        job_data = jobs.get(job_id)
        if job_data:
            return JobState(**job_data)
        return None

    def update_job(self, job_state: JobState):
        """Update job state."""
        job_state.updated_at = datetime.utcnow().isoformat()
        self._save_job(job_state)

    def list_jobs(self, job_type: Optional[str] = None) -> List[JobState]:
        """List all jobs, optionally filtered by type."""
        jobs = self._load_all_jobs()
        job_states = [JobState(**job_data) for job_data in jobs.values()]

        if job_type:
            job_states = [j for j in job_states if j.job_type == job_type]

        # Sort by updated_at descending
        job_states.sort(key=lambda j: j.updated_at, reverse=True)
        return job_states

    def delete_job(self, job_id: str):
        """Delete a job and its associated data."""
        jobs = self._load_all_jobs()
        if job_id in jobs:
            del jobs[job_id]
            self._save_all_jobs(jobs)

            # Delete hash file
            hash_file = self.hashes_dir / f"{job_id}.txt"
            if hash_file.exists():
                hash_file.unlink()

    def add_processed_hash(self, job_id: str, content_hash: str):
        """Add a hash to the processed set for deduplication."""
        hash_file = self.hashes_dir / f"{job_id}.txt"
        with open(hash_file, 'a') as f:
            f.write(f"{content_hash}\n")

    def get_processed_hashes(self, job_id: str) -> Set[str]:
        """Get all processed hashes for a job."""
        hash_file = self.hashes_dir / f"{job_id}.txt"
        if not hash_file.exists():
            return set()

        with open(hash_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())

    def is_duplicate(self, job_id: str, content: str) -> bool:
        """Check if content has already been processed."""
        content_hash = self._hash_content(content)
        processed_hashes = self.get_processed_hashes(job_id)
        return content_hash in processed_hashes

    def mark_processed(self, job_id: str, content: str):
        """Mark content as processed."""
        content_hash = self._hash_content(content)
        self.add_processed_hash(job_id, content_hash)

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _generate_job_id(self, job_type: str) -> str:
        """Generate unique job ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(4).hex()
        return f"{job_type}_{timestamp}_{random_suffix}"

    def _save_job(self, job_state: JobState):
        """Save single job to storage."""
        jobs = self._load_all_jobs()
        jobs[job_state.job_id] = asdict(job_state)
        self._save_all_jobs(jobs)

    def _load_all_jobs(self) -> Dict:
        """Load all jobs from storage."""
        if not self.jobs_file.exists():
            return {}

        with open(self.jobs_file, 'r') as f:
            return json.load(f)

    def _save_all_jobs(self, jobs: Dict):
        """Save all jobs to storage."""
        with open(self.jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2)

    def get_progress(self, job_id: str) -> Dict:
        """Get job progress summary."""
        job = self.get_job(job_id)
        if not job:
            return {}

        progress_pct = 0
        if job.total_samples > 0:
            progress_pct = (job.processed_samples / job.total_samples) * 100

        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_samples": job.total_samples,
            "processed_samples": job.processed_samples,
            "failed_samples": job.failed_samples,
            "progress_percentage": round(progress_pct, 2),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "unique_samples": len(self.get_processed_hashes(job_id)),
            "error_count": job.error_count,
            "last_error": job.last_error,
            "status_message": job.status_message,
            "hf_dataset_repo": job.hf_dataset_repo,
            "dataset_url": f"https://huggingface.co/datasets/{job.hf_dataset_repo}" if job.hf_dataset_repo else None,
        }

    def create_split(
        self,
        dataset_repos: List[str],
        train_indices: Dict[str, List[int]],
        benchmark_indices: Dict[str, List[int]],
        metadata: Dict
    ) -> str:
        """
        Create and save a train/benchmark split.

        Args:
            dataset_repos: List of dataset repository names
            train_indices: Dict mapping repo_name -> list of training indices
            benchmark_indices: Dict mapping repo_name -> list of benchmark indices
            metadata: Additional metadata (seed, percentage, etc.)

        Returns:
            split_id: Unique identifier for this split
        """
        split_id = self._generate_split_id()
        split_data = {
            "split_id": split_id,
            "created_at": datetime.utcnow().isoformat(),
            "dataset_repos": dataset_repos,
            "train_indices": train_indices,
            "benchmark_indices": benchmark_indices,
            "metadata": metadata
        }

        split_file = self.splits_dir / f"{split_id}.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)

        return split_id

    def get_split(self, split_id: str) -> Optional[Dict]:
        """Retrieve a saved split by ID."""
        split_file = self.splits_dir / f"{split_id}.json"
        if not split_file.exists():
            return None

        with open(split_file, 'r') as f:
            return json.load(f)

    def get_split_indices(self, split_id: str, split_type: str) -> Dict[str, List[int]]:
        """
        Get indices for a specific split type (train or benchmark).

        Args:
            split_id: The split identifier
            split_type: Either "train" or "benchmark"

        Returns:
            Dict mapping dataset repo names to lists of indices
        """
        split_data = self.get_split(split_id)
        if not split_data:
            return {}

        if split_type == "train":
            return split_data.get("train_indices", {})
        elif split_type == "benchmark":
            return split_data.get("benchmark_indices", {})
        else:
            raise ValueError(f"Invalid split_type: {split_type}. Must be 'train' or 'benchmark'")

    def _generate_split_id(self) -> str:
        """Generate unique split ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(4).hex()
        return f"split_{timestamp}_{random_suffix}"
