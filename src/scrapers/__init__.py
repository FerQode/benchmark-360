"""
Scraping factory and central registry for Benchmark 360.

This module centralizes scraper instantiation using the Factory Pattern.
It implements the Open/Closed Principle: to add a new ISP, simply
add its scraper to the ALL_ISP_URLS registry; the rest of the pipeline
automatically discovers and utilizes it.
"""

from __future__ import annotations

from pathlib import Path

from src.scrapers.base_scraper import BaseISPScraper
from src.scrapers.alfanet_scraper import AlfanetScraper
from src.scrapers.claro_scraper import ClaroScraper
from src.scrapers.cnt_scraper import CNTScraper
from src.scrapers.ecuanet_scraper import EcuanetScraper
from src.scrapers.fibramax_scraper import FibramaxScraper
from src.scrapers.netlife_scraper import NetlifeScraper
from src.scrapers.puntonet_scraper import PuntonetScraper
from src.scrapers.xtrim_scraper import XtrimScraper
from src.utils.robots_checker import RobotsChecker

# Central registry mapping ISP keys to their base URLs
ALL_ISP_URLS: dict[str, str] = {
    "netlife": "https://www.netlife.ec",
    "cnt": "https://www.cnt.com.ec",
    "claro": "https://www.claro.com.ec",
    "xtrim": "https://www.xtrim.com.ec",
    "ecuanet": "https://www.ecuanet.ec",
    "puntonet": "https://www.puntonet.ec",
    "alfanet": "https://www.alfanet.ec",
    "fibramax": "https://www.fibramax.ec",
}


def build_all_scrapers(
    robots_checker: RobotsChecker,
    data_raw_path: Path = Path("data/raw"),
) -> dict[str, BaseISPScraper]:
    """Factory method to instantiate all registered ISP scrapers.

    Args:
        robots_checker: Shared instance of the compliance checker.
            Must be passed to all scrapers to ensure unified polite
            delays and caching.
        data_raw_path: Directory where raw HTML/screenshots will
            be saved by the BaseScraper.

    Returns:
        Dictionary mapping isp_key -> initialized Scraper instance.
    """
    scrapers = {
        "netlife": NetlifeScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
        "cnt": CNTScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
        "claro": ClaroScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
        "xtrim": XtrimScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
        "ecuanet": EcuanetScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
        "puntonet": PuntonetScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
        "alfanet": AlfanetScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
        "fibramax": FibramaxScraper(robots_checker=robots_checker, data_raw_path=data_raw_path),
    }

    # Verify that all URLs in the registry have an instantiated scraper
    missing = set(ALL_ISP_URLS.keys()) - set(scrapers.keys())
    if missing:
        raise NotImplementedError(
            f"Factory error: missing scraper implementation for ISPs: {missing}"
        )

    return scrapers
