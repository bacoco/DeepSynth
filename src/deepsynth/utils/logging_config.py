"""
Centralized logging configuration for DeepSynth.

Provides consistent logging across all modules with proper formatting,
file output, and level management.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[35m", # Magenta
    "RESET": "\033[0m",     # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""

    def __init__(self, *args, use_colors: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors if enabled."""
        if self.use_colors:
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Setup a logger with consistent configuration.

    Args:
        name: Logger name. If None, returns root logger.
        level: Logging level (default: INFO).
        log_file: Optional file path for logging output.
        format_string: Custom format string.
        use_colors: Whether to use colors in terminal output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name or "deepsynth")

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        format_string,
        use_colors=use_colors,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            format_string,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def setup_global_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = "logs/deepsynth.log",
) -> None:
    """
    Setup global logging configuration for the entire application.

    Args:
        level: Global logging level.
        log_file: Path to log file.
    """
    # Configure root logger
    root_logger = setup_logger(
        name=None,
        level=level,
        log_file=log_file,
    )

    # Configure specific loggers
    loggers_config = {
        "deepsynth.training": logging.INFO,
        "deepsynth.pipelines": logging.INFO,
        "deepsynth.data": logging.WARNING,
        "deepsynth.inference": logging.INFO,
        "transformers": logging.WARNING,
        "datasets": logging.WARNING,
        "huggingface_hub": logging.WARNING,
    }

    for logger_name, logger_level in loggers_config.items():
        specific_logger = logging.getLogger(logger_name)
        specific_logger.setLevel(logger_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the standard configuration.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return setup_logger(name)


# Convenience functions for module-level logging
def log_info(message: str, *args, **kwargs) -> None:
    """Log an info message."""
    logging.getLogger("deepsynth").info(message, *args, **kwargs)


def log_warning(message: str, *args, **kwargs) -> None:
    """Log a warning message."""
    logging.getLogger("deepsynth").warning(message, *args, **kwargs)


def log_error(message: str, *args, **kwargs) -> None:
    """Log an error message."""
    logging.getLogger("deepsynth").error(message, *args, **kwargs)


def log_debug(message: str, *args, **kwargs) -> None:
    """Log a debug message."""
    logging.getLogger("deepsynth").debug(message, *args, **kwargs)