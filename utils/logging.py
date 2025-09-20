"""
Centralized logging utilities for Reddit mod collection pipeline.
Simple unified logger that works for both main process and multiprocessing.
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path


def setup_stage_logger(stage_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Set up a unified logger for a pipeline stage that works with multiprocessing.
    Uses thread-safe file writing and process-safe naming.

    Args:
        stage_name: Name of the stage (e.g., "stage1_collect_mod_comments")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance that works in both main and worker processes
    """
    from config import PATHS

    # Create logs directory if it doesn't exist
    log_dir = Path(PATHS['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{stage_name}_{timestamp}.log"

    # Create logger with process-safe name
    process_id = os.getpid()
    logger_name = f"{stage_name}_{process_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters with process info
    file_formatter = logging.Formatter(
        '%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        'PID:%(process)d - %(levelname)s - %(message)s'
    )

    # Thread-safe file handler (multiple processes can write to same file)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(file_formatter)

    # Add thread lock for file writing safety
    file_handler.lock = threading.Lock()
    logger.addHandler(file_handler)

    # Console handler (for real-time feedback)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log the initialization
    logger.info(f"Logger initialized for {stage_name} (PID: {process_id})")
    logger.info(f"Log file: {log_file}")

    return logger


def get_stage_logger(stage_num: int, stage_description: str = None) -> logging.Logger:
    """
    Get a logger for a specific stage number.

    Args:
        stage_num: Stage number (0-9)
        stage_description: Optional description for the stage

    Returns:
        Configured logger instance
    """
    if stage_description:
        stage_name = f"stage{stage_num}_{stage_description}"
    else:
        stage_name = f"stage{stage_num}"

    return setup_stage_logger(stage_name)


def log_stage_start(logger: logging.Logger, stage_num: int, stage_name: str):
    """Log the start of a pipeline stage."""
    logger.info("=" * 60)
    logger.info(f"🚀 Stage {stage_num}: {stage_name}")
    logger.info("=" * 60)


def log_stage_end(logger: logging.Logger, stage_num: int, success: bool = True, elapsed_time: float = None):
    """Log the end of a pipeline stage."""
    status = "✅ COMPLETED" if success else "❌ FAILED"
    time_str = f" in {elapsed_time:.1f}s" if elapsed_time else ""
    logger.info(f"{status}: Stage {stage_num}{time_str}")
    logger.info("=" * 60)


def log_progress(logger: logging.Logger, current: int, total: int, item_name: str = "items"):
    """Log progress with percentage."""
    percentage = (current / total) * 100 if total > 0 else 0
    logger.info(f"📊 Progress: {current:,}/{total:,} {item_name} ({percentage:.1f}%)")


def log_stats(logger: logging.Logger, stats_dict: dict, title: str = "Statistics"):
    """Log statistics in a formatted way."""
    logger.info(f"📈 {title}:")
    for key, value in stats_dict.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:,}")
        else:
            logger.info(f"  {key}: {value}")


def log_error_and_continue(logger: logging.Logger, error: Exception, context: str = ""):
    """Log an error but continue processing."""
    context_str = f" in {context}" if context else ""
    logger.error(f"❌ Error{context_str}: {str(error)}")
    logger.debug(f"Full traceback{context_str}:", exc_info=True)


def log_file_operation(logger: logging.Logger, operation: str, file_path: str, success: bool = True):
    """Log file operations (read, write, etc.)."""
    status = "✅" if success else "❌"
    logger.debug(f"{status} {operation}: {file_path}")