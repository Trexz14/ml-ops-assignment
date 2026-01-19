"""
Centralized logging configuration using loguru.

This module provides a consistent logging setup across the entire application.
It configures loguru to write logs to both console and rotating log files.

Log levels:
    - DEBUG: Detailed information for debugging
    - INFO: General information about application progress
    - WARNING: Warning messages for potentially harmful situations
    - ERROR: Error messages for serious problems
    - CRITICAL: Critical messages for very serious errors

Log files are stored in the logs/ directory with automatic rotation:
    - app.log: General application logs (rotated at 10 MB, kept for 7 days)
    - errors.log: Error and critical messages only

Environment variable LOG_LEVEL can be set to control logging verbosity.
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

# Get log level from environment variable (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Console handler - colored output with simple format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True,
)

# File handler - detailed logs with rotation
logger.add(
    LOG_DIR / "app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress old logs
    enqueue=True,  # Thread-safe logging
)

# Error file handler - errors and critical only
logger.add(
    LOG_DIR / "errors.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
    level="ERROR",
    rotation="5 MB",
    retention="14 days",
    compression="zip",
    enqueue=True,
    backtrace=True,  # Include full traceback for errors
    diagnose=True,  # Include detailed diagnostic information
)


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name.

    Args:
        name: The name for the logger (typically __name__ of the calling module)

    Returns:
        Logger instance configured with the application settings

    Example:
        >>> from ml_ops_assignment.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting data processing")
    """
    return logger.bind(name=name)
