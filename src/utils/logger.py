"""
Centralized logging configuration for Benchmark 360.

Configures Loguru with consistent formatting, log levels,
and file output for the entire pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """Configure Loguru logger for the Benchmark 360 pipeline.

    Sets up console output with colors and optional file output
    with rotation. Should be called once at application startup.

    Args:
        log_level: Minimum log level to capture. One of:
            DEBUG, INFO, WARNING, ERROR, CRITICAL. Defaults to env var LOG_LEVEL.
        log_file: Optional path for file-based log output.
            If None, attempts to use env var LOG_FILE.
    """
    # Remove default handler
    logger.remove()

    # Resolve from environment variables if not explicitly provided
    resolved_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    
    # Try to resolve log file from env var if not passed
    env_log_file = os.getenv("LOG_FILE")
    resolved_log_file = log_file or (Path(env_log_file) if env_log_file else None)

    # Console handler — colored, human-readable
    logger.add(
        sys.stdout,
        level=resolved_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,    # Thread-safe y non-blocking para concurrencia (Playwright)
        backtrace=True,  # Extiende el trazado de errores
        diagnose=True,   # Muestra el valor de las variables que causaron la excepción
    )

    # File handler — if path provided
    if resolved_log_file:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(resolved_log_file),
            level=resolved_level,
            rotation="50 MB",
            retention="7 days",
            compression="zip",
            encoding="utf-8",
            serialize=True,  # Estructura los logs en formato JSON para máquinas
            enqueue=True,    # Thread-safe y non-blocking
        )

    logger.info("✅ Logger initialized at level: {} (JSON File Logging: {})", 
                resolved_level, 
                "ENABLED" if resolved_log_file else "DISABLED")
