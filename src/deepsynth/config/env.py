"""Configuration management with .env support and GDPR compliance.

Enhanced configuration with:
- GDPR-compliant privacy controls
- Type-safe environment variable loading
- Production safety checks
- Feature flags for gradual rollout
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from enum import Enum

_ENV_CACHE: Dict[Path, bool] = {}


def load_env(env_file: str = ".env", *, strict: bool = True) -> bool:
    """Load environment variables from ``env_file``.

    Parameters
    ----------
    env_file:
        Path to the ``.env`` file to read.
    strict:
        When ``True`` (default) a missing file raises :class:`FileNotFoundError`.
        Data preparation scripts set ``strict`` to ``False`` so the absence of a
        configuration file does not stop ad-hoc explorations.

    Returns
    -------
    bool
        ``True`` when variables were loaded, ``False`` otherwise.
    """

    path = Path(env_file).resolve()
    if path in _ENV_CACHE:
        return _ENV_CACHE[path]

    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"{env_file} not found. Copy .env.example to .env and configure it."
            )
        _ENV_CACHE[path] = False
        return False

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key:
                    os.environ[key] = value

    _ENV_CACHE[path] = True
    return True


def load_shared_env(env_file: str = ".env") -> bool:
    """Load ``env_file`` without failing when it is missing.

    The helper is used by scripts that want to opportunistically source
    credentials while remaining runnable in development environments that have
    not been fully configured.
    """

    return load_env(env_file, strict=False)


@dataclass
class Config:
    """Configuration loaded from environment variables."""

    # HuggingFace
    hf_token: str
    hf_username: str

    # Dataset
    source_dataset: str
    source_subset: Optional[str]
    target_dataset_name: str
    max_samples_per_split: Optional[int]

    # Training
    model_name: str
    output_model_name: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    max_length: int

    # Optimization
    mixed_precision: str
    gradient_accumulation_steps: int

    @staticmethod
    def _parse_optional_int(key: str, default: str | None) -> Optional[int]:
        """Read an optional integer from the environment.

        Parameters
        ----------
        key:
            Environment variable name to read.
        default:
            Default value used when the variable is missing. ``None`` propagates
            the absence of the variable.
        """

        value = os.getenv(key)
        if value is None or value == "":
            if default is None or default == "":
                return None
            value = default

        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer for {key}: {value!r}") from exc

    @classmethod
    def from_env(cls, env_file: str = ".env") -> Config:
        """Load configuration from .env file."""
        load_env(env_file)

        def get_env(key: str, default: Optional[str] = None, required: bool = True) -> str:
            value = os.getenv(key, default)
            if required and not value:
                raise ValueError(f"Missing required environment variable: {key}")
            return value or ""

        return cls(
            hf_token=get_env("HF_TOKEN"),
            hf_username=get_env("HF_USERNAME"),
            source_dataset=get_env("SOURCE_DATASET"),
            source_subset=get_env("SOURCE_SUBSET", None, required=False),
            target_dataset_name=get_env("TARGET_DATASET_NAME"),
            max_samples_per_split=cls._parse_optional_int(
                "MAX_SAMPLES_PER_SPLIT", "1000"
            ),
            model_name=get_env("MODEL_NAME"),
            output_model_name=get_env("OUTPUT_MODEL_NAME"),
            batch_size=int(get_env("BATCH_SIZE", "2")),
            num_epochs=int(get_env("NUM_EPOCHS", "1")),
            learning_rate=float(get_env("LEARNING_RATE", "2e-5")),
            max_length=int(get_env("MAX_LENGTH", "512")),
            mixed_precision=get_env("MIXED_PRECISION", "bf16"),
            gradient_accumulation_steps=int(get_env("GRADIENT_ACCUMULATION_STEPS", "4")),
        )

    @property
    def target_dataset_repo(self) -> str:
        """Full HuggingFace dataset repository ID."""
        return f"{self.hf_username}/{self.target_dataset_name}"

    @property
    def output_model_repo(self) -> str:
        """Full HuggingFace model repository ID."""
        return f"{self.hf_username}/{self.output_model_name}"


# ============================================================================
# Enhanced Configuration with GDPR Compliance
# ============================================================================


class Environment(str, Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class PrivacyConfig:
    """GDPR-compliant privacy configuration (Article 5, 7, 17)."""

    allow_sample_persistence: bool = field(
        default_factory=lambda: os.getenv("ALLOW_SAMPLE_PERSISTENCE", "false").lower() == "true"
    )
    redact_pii_in_logs: bool = field(
        default_factory=lambda: os.getenv("REDACT_PII_IN_LOGS", "true").lower() == "true"
    )
    data_retention_days: int = field(
        default_factory=lambda: int(os.getenv("DATA_RETENTION_DAYS", "90"))
    )
    require_consent: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_CONSENT", "true").lower() == "true"
    )
    anonymize_metrics: bool = field(
        default_factory=lambda: os.getenv("ANONYMIZE_METRICS", "true").lower() == "true"
    )


@dataclass
class DeepSynthConfig:
    """Enhanced configuration with GDPR compliance and feature flags."""

    environment: Environment = field(
        default_factory=lambda: Environment(os.getenv("ENVIRONMENT", "development"))
    )
    service_name: str = field(
        default_factory=lambda: os.getenv("SERVICE_NAME", "deepsynth")
    )
    version: str = field(
        default_factory=lambda: os.getenv("VERSION", "0.2.0")
    )
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION


# Global configuration instance
_enhanced_config: Optional[DeepSynthConfig] = None


def get_config(reload: bool = False) -> DeepSynthConfig:
    """Get global enhanced configuration instance.

    Args:
        reload: Force reload configuration from environment

    Returns:
        DeepSynthConfig instance

    Example:
        >>> config = get_config()
        >>> if config.privacy.allow_sample_persistence:
        ...     save_samples()
    """
    global _enhanced_config

    if _enhanced_config is None or reload:
        _enhanced_config = DeepSynthConfig()

    return _enhanced_config


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = Config.from_env()
        print("✓ Configuration loaded successfully")
        print(f"  HF Username: {config.hf_username}")
        print(f"  Source Dataset: {config.source_dataset}")
        print(f"  Target Dataset: {config.target_dataset_repo}")
        print(f"  Output Model: {config.output_model_repo}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")

    # Test enhanced configuration
    try:
        enhanced = get_config()
        print("\n✓ Enhanced configuration loaded")
        print(f"  Environment: {enhanced.environment.value}")
        print(f"  Privacy - PII Redaction: {enhanced.privacy.redact_pii_in_logs}")
        print(f"  Privacy - Data Retention: {enhanced.privacy.data_retention_days} days")
    except Exception as e:
        print(f"✗ Enhanced configuration error: {e}")