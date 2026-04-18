# src/scrapers/__init__.py
"""
ISP scraper registry — Factory Pattern implementation.

Centralizes all scraper instantiation so the orchestrator
never needs to know about specific scraper classes.

Adding a new ISP requires only:
    1. Create new_isp_scraper.py inheriting BaseISPScraper
    2. Add entry to ALL_SCRAPERS dict below

No other file needs modification (Open/Closed Principle).

Typical usage example:
    >>> from src.scrapers import build_all_scrapers
    >>> scrapers = build_all_scrapers(robots_checker=checker)
    >>> for key, scraper in scrapers.items():
    ...     page = await scraper.scrape()
"""

from __future__ import annotations

from src.scrapers.alfanet_scraper import AlfanetScraper
from src.scrapers.claro_scraper import ClaroScraper
from src.scrapers.cnt_scraper import CNTScraper
from src.scrapers.ecuanet_scraper import EcuanetScraper
from src.scrapers.fibramax_scraper import FibramaxScraper
from src.scrapers.netlife_scraper import NetlifeScraper
from src.scrapers.puntonet_scraper import PuntonetScraper
from src.scrapers.xtrim_scraper import XtrimScraper
from src.scrapers.base_scraper import BaseISPScraper
from src.utils.robots_checker import RobotsChecker

# Registry: isp_key → (ScraperClass, base_url)
# This is the SINGLE place to register new ISPs
_SCRAPER_REGISTRY: dict[str, tuple[type[BaseISPScraper], str]] = {
    "netlife":  (NetlifeScraper,  "https://netlife.ec"),
    "claro":    (ClaroScraper,    "https://www.claro.com.ec"),
    "cnt":      (CNTScraper,      "https://www.cnt.com.ec"),
    "xtrim":    (XtrimScraper,    "https://www.xtrim.com.ec"),
    "ecuanet":  (EcuanetScraper,  "https://ecuanet.ec"),
    "puntonet": (PuntonetScraper, "https://www.celerity.ec"),
    "alfanet":  (AlfanetScraper,  "https://www.alfanet.ec"),
    "fibramax": (FibramaxScraper, "https://fibramax.ec"),
}

# Public constants for pipeline use
ALL_ISP_URLS: dict[str, str] = {
    key: url for key, (_, url) in _SCRAPER_REGISTRY.items()
}


def build_all_scrapers(
    robots_checker: RobotsChecker,
    delay_range: tuple[float, float] = (2.0, 5.0),
) -> dict[str, BaseISPScraper]:
    """Instantiate all registered ISP scrapers (Factory Pattern).

    Creates one scraper instance per ISP, all sharing the same
    robots_checker instance for compliance coordination.

    Args:
        robots_checker: Pre-initialized and pre-loaded checker.
        delay_range: (min_sec, max_sec) for polite delays.

    Returns:
        Dict mapping isp_key → initialized scraper instance.

    Example:
        >>> checker = RobotsChecker()
        >>> await checker.analyze_all_isps(ALL_ISP_URLS)
        >>> scrapers = build_all_scrapers(checker)
        >>> len(scrapers)
        8
    """
    return {
        key: ScraperClass(
            isp_key=key,
            base_url=url,
            robots_checker=robots_checker,
            delay_range=delay_range,
        )
        for key, (ScraperClass, url) in _SCRAPER_REGISTRY.items()
    }
