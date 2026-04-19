# src/utils/logger.py
"""Configuración de logging estructurado y persistente para Benchmark 360.

Configura Loguru con dos handlers:
  1. Consola (stderr): Formato legible para desarrollo.
  2. Archivo rotativo: JSON estructurado para análisis post-mortem.

Uso en módulos:
    from src.utils.logger import logger
    log = logger.bind(isp="netlife", phase="scraping")
    log.info("Iniciando extracción...")
    log.error("Timeout al cargar página")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """Configura Loguru con handlers de consola y archivo JSON.

    Args:
        log_level: Nivel mínimo para el handler de consola.
    """
    # Importar settings aquí para evitar circular imports en el boot
    from src.config.settings import get_settings
    settings = get_settings()
    log_path = str(Path(settings.logs_dir) / "pipeline_{time:YYYY-MM-DD}.log")

    # Limpiar todos los handlers previos
    logger.remove()

    # 1. Consola: Formato legible para desarrollo
    #    Dos variantes: con y sin contexto ISP
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[isp]: >10}</cyan> | "
            "<yellow>{extra[phase]:<12}</yellow> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        filter=lambda r: "isp" in r["extra"],
        colorize=True,
    )
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        filter=lambda r: "isp" not in r["extra"],
        colorize=True,
    )

    # 2. Archivo: formato estructurado para Datadog/Splunk
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra} | {message}",
        rotation="1 day",
        retention="7 days",
        compression="gz",
        level="DEBUG",
    )


# Inicializar al importar el módulo
setup_logger()
