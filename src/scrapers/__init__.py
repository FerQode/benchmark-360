# src/scrapers/__init__.py
"""
Fábrica de scrapers dinámicos para todos los ISPs registrados.

Con el nuevo BaseISPScraper dinámico, no hay subclases individuales.
Todos los ISPs usan la misma clase con diferente ISPStrategy.

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

from pathlib import Path

from src.scrapers.base_scraper import BaseISPScraper
from src.scrapers.isp_url_strategy import ISP_STRATEGIES
from src.utils.robots_checker import RobotsChecker

# Mapa de URLs para el RobotsChecker (se mantiene para compatibilidad)
ALL_ISP_URLS: dict[str, str] = {
    key: strategy.plan_pages[0].url
    for key, strategy in ISP_STRATEGIES.items()
    if strategy.plan_pages
}


def build_all_scrapers(
    robots_checker: RobotsChecker,
    data_raw_path: Path = Path("data/raw"),
) -> dict[str, BaseISPScraper]:
    """Construye un scraper dinámico para cada ISP registrado.

    Args:
        robots_checker: Verificador de robots.txt compartido.
        data_raw_path: Directorio raíz para datos crudos.

    Returns:
        Diccionario {isp_key: BaseISPScraper} para todos los ISPs.
    """
    return {
        isp_key: BaseISPScraper(
            isp_key=isp_key,
            data_raw_path=data_raw_path,
            robots_checker=robots_checker,
        )
        for isp_key in ISP_STRATEGIES
    }
