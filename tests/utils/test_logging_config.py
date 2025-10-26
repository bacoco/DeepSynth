"""
Tests for logging configuration.

Tests logger setup, formatting, and global configuration.
"""

import logging
import tempfile
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from deepsynth.utils.logging_config import (
    setup_logger,
    setup_global_logging,
    get_logger,
    ColoredFormatter,
)


class TestColoredFormatter:
    """Test the colored formatter."""

    def test_formatter_creation(self):
        """Test formatter can be created."""
        formatter = ColoredFormatter(
            "%(levelname)s - %(message)s",
            use_colors=True,
        )
        assert formatter is not None

    def test_formatter_without_colors(self):
        """Test formatter without colors."""
        formatter = ColoredFormatter(
            "%(levelname)s - %(message)s",
            use_colors=False,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "Test message" in output
        assert "\033[" not in output  # No color codes

    def test_formatter_with_colors(self):
        """Test formatter preserves color functionality."""
        formatter = ColoredFormatter(
            "%(levelname)s - %(message)s",
            use_colors=True,
        )
        assert formatter.use_colors is not None


class TestSetupLogger:
    """Test logger setup."""

    def test_basic_setup(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_custom_level(self):
        """Test logger with custom level."""
        logger = setup_logger("test_logger_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_with_log_file(self, tmp_path):
        """Test logger with file output."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_file_logger", log_file=str(log_file))

        logger.info("Test message")

        # Check file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_no_duplicate_handlers(self):
        """Test that calling setup multiple times doesn't duplicate handlers."""
        logger_name = "test_no_duplicate"
        logger1 = setup_logger(logger_name)
        handler_count_1 = len(logger1.handlers)

        logger2 = setup_logger(logger_name)
        handler_count_2 = len(logger2.handlers)

        assert handler_count_1 == handler_count_2
        assert logger1 is logger2

    def test_custom_format(self):
        """Test logger with custom format string."""
        logger = setup_logger(
            "test_format",
            format_string="%(name)s | %(message)s",
        )
        assert len(logger.handlers) > 0

    def test_logger_propagation(self):
        """Test logger doesn't propagate to parent."""
        logger = setup_logger("test_propagation")
        assert logger.propagate is False


class TestSetupGlobalLogging:
    """Test global logging setup."""

    def test_global_setup(self, tmp_path):
        """Test global logging configuration."""
        log_file = tmp_path / "global.log"
        setup_global_logging(level=logging.INFO, log_file=str(log_file))

        # Get a logger and test it
        logger = logging.getLogger("deepsynth.training")
        logger.info("Global test message")

        assert log_file.exists()

    def test_global_setup_configures_subloggers(self):
        """Test global setup configures specific loggers."""
        setup_global_logging()

        # Check specific loggers are configured
        training_logger = logging.getLogger("deepsynth.training")
        assert training_logger.level == logging.INFO

        transformers_logger = logging.getLogger("transformers")
        assert transformers_logger.level == logging.WARNING


class TestGetLogger:
    """Test get_logger convenience function."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test_get_logger")
        assert logger.name == "test_get_logger"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_returns_same_instance(self):
        """Test getting the same logger returns same instance."""
        logger1 = get_logger("test_same_logger")
        logger2 = get_logger("test_same_logger")
        assert logger1 is logger2


class TestLoggingLevels:
    """Test different logging levels."""

    def test_debug_level(self, tmp_path):
        """Test debug level logging."""
        log_file = tmp_path / "debug.log"
        logger = setup_logger("test_debug", level=logging.DEBUG, log_file=str(log_file))

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        content = log_file.read_text()
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content

    def test_warning_level_filters(self, tmp_path):
        """Test warning level filters lower levels."""
        log_file = tmp_path / "warning.log"
        logger = setup_logger("test_warning", level=logging.WARNING, log_file=str(log_file))

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        content = log_file.read_text()
        assert "Debug message" not in content
        assert "Info message" not in content
        assert "Warning message" in content
        assert "Error message" in content


class TestLoggerHierarchy:
    """Test logger hierarchy and naming."""

    def test_module_logger_naming(self):
        """Test logger naming follows module convention."""
        logger = get_logger("deepsynth.training.optimized_trainer")
        assert "deepsynth.training.optimized_trainer" in logger.name

    def test_nested_loggers(self):
        """Test nested logger configuration."""
        parent = get_logger("deepsynth")
        child = get_logger("deepsynth.training")

        # Child should inherit from parent but not propagate
        assert child.propagate is False


class TestLoggerOutput:
    """Test actual logger output."""

    def test_logger_writes_to_file(self, tmp_path):
        """Test logger actually writes to file."""
        log_file = tmp_path / "output.log"
        logger = setup_logger("test_output", log_file=str(log_file))

        test_messages = [
            "Message 1",
            "Message 2",
            "Message 3 with %s formatting" % "string",
        ]

        for msg in test_messages:
            logger.info(msg)

        content = log_file.read_text()
        for msg in test_messages:
            assert msg in content

    def test_logger_respects_level(self, tmp_path):
        """Test logger respects set level."""
        log_file = tmp_path / "level.log"
        logger = setup_logger("test_level", level=logging.ERROR, log_file=str(log_file))

        logger.debug("Should not appear")
        logger.info("Should not appear")
        logger.warning("Should not appear")
        logger.error("Should appear")
        logger.critical("Should appear")

        content = log_file.read_text()
        assert "Should not appear" not in content
        assert content.count("Should appear") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])