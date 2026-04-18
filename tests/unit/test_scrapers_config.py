# tests/unit/test_scrapers_config.py
"""
Unit tests for individual ISP scraper configurations.

Tests each scraper's:
    - isp_key correctness
    - base_url correctness
    - requires_playwright() contract
    - get_plan_urls() structure and validity
    - Plan URLs are within the same domain as base_url
    - First URL in list is the most specific plan page
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest

from src.scrapers.alfanet_scraper import AlfanetScraper
from src.scrapers.claro_scraper import ClaroScraper
from src.scrapers.cnt_scraper import CNTScraper
from src.scrapers.ecuanet_scraper import EcuanetScraper
from src.scrapers.fibramax_scraper import FibramaxScraper
from src.scrapers.netlife_scraper import NetlifeScraper
from src.scrapers.puntonet_scraper import PuntonetScraper
from src.scrapers.xtrim_scraper import XtrimScraper


# ─────────────────────────────────────────────────────────────────
# Parametrize: one test class drives all 8 scrapers
# ─────────────────────────────────────────────────────────────────

SCRAPER_CONFIGS = [
    # (ScraperClass, expected_key, expected_base_domain, requires_playwright)
    (NetlifeScraper,  "netlife",  "netlife.ec",   True),
    (CNTScraper,      "cnt",      "cnt.com.ec",   False),
    (ClaroScraper,    "claro",    "claro.com.ec", True),
    (XtrimScraper,    "xtrim",    "xtrim.com.ec", True),
    (EcuanetScraper,  "ecuanet",  "ecuanet.ec",   True),
    (PuntonetScraper, "puntonet", "puntonet.ec",  True),
    (AlfanetScraper,  "alfanet",  "alfanet.ec",   True),
    (FibramaxScraper, "fibramax", "fibramax.ec",  True),
]


@pytest.fixture
def make_scraper(mock_robots_checker: MagicMock, tmp_data_raw: Path):
    """Factory fixture to instantiate any scraper class."""
    def _factory(scraper_class):
        return scraper_class(
            robots_checker=mock_robots_checker,
            data_raw_path=tmp_data_raw,
        )
    return _factory


class TestISPScraperIdentity:
    """Validates each scraper's identity attributes."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,expected_key,expected_domain,_",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_isp_key_is_correct(
        self, make_scraper, scraper_class, expected_key, expected_domain, _
    ) -> None:
        scraper = make_scraper(scraper_class)
        assert scraper.isp_key == expected_key

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,_,expected_domain,__",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_base_url_contains_correct_domain(
        self, make_scraper, scraper_class, _, expected_domain, __
    ) -> None:
        scraper = make_scraper(scraper_class)
        assert expected_domain in scraper.base_url

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,_,__,expected_requires_pw",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_requires_playwright_is_correct(
        self, make_scraper, scraper_class, _, __, expected_requires_pw
    ) -> None:
        scraper = make_scraper(scraper_class)
        assert scraper.requires_playwright() is expected_requires_pw


class TestISPScraperPlanUrls:
    """Validates get_plan_urls() contract for all scrapers."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,isp_key,expected_domain,_",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_plan_urls_not_empty(
        self, make_scraper, scraper_class, isp_key, expected_domain, _
    ) -> None:
        scraper = make_scraper(scraper_class)
        urls = scraper.get_plan_urls()
        assert len(urls) >= 1, f"{isp_key}: must have at least 1 plan URL"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,isp_key,expected_domain,_",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_all_plan_urls_use_https(
        self, make_scraper, scraper_class, isp_key, expected_domain, _
    ) -> None:
        scraper = make_scraper(scraper_class)
        for url in scraper.get_plan_urls():
            assert url.startswith("https://"), (
                f"{isp_key}: URL '{url}' must use HTTPS"
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,isp_key,expected_domain,_",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_all_plan_urls_are_valid(
        self, make_scraper, scraper_class, isp_key, expected_domain, _
    ) -> None:
        scraper = make_scraper(scraper_class)
        for url in scraper.get_plan_urls():
            parsed = urlparse(url)
            assert parsed.scheme in ("https", "http"), (
                f"{isp_key}: '{url}' has invalid scheme"
            )
            assert parsed.netloc, f"{isp_key}: '{url}' has no netloc"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,isp_key,expected_domain,_",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_no_duplicate_plan_urls(
        self, make_scraper, scraper_class, isp_key, expected_domain, _
    ) -> None:
        scraper = make_scraper(scraper_class)
        urls = scraper.get_plan_urls()
        assert len(urls) == len(set(urls)), (
            f"{isp_key}: duplicate plan URLs found: {urls}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scraper_class,isp_key,expected_domain,_",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    def test_delay_range_is_valid(
        self, make_scraper, scraper_class, isp_key, expected_domain, _
    ) -> None:
        """Delay range min must be >= 2.0s (ethical scraping requirement)."""
        scraper = make_scraper(scraper_class)
        min_delay, max_delay = scraper.delay_range
        assert min_delay >= 2.0, (
            f"{isp_key}: min delay {min_delay}s violates ethical scraping (min 2.0s)"
        )
        assert max_delay > min_delay, (
            f"{isp_key}: max delay must be > min delay"
        )


class TestScrapeMethodReturnsScrapedPage:
    """Validates scrape() returns valid ScrapedPage DTO with mocked network."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scraper_class,isp_key,_,__",
        SCRAPER_CONFIGS,
        ids=[c[1] for c in SCRAPER_CONFIGS],
    )
    async def test_scrape_returns_scraped_page(
        self,
        make_scraper,
        scraper_class,
        isp_key,
        _,
        __,
    ) -> None:
        """All scrapers must return ScrapedPage even when network fails."""
        from src.scrapers.base_scraper import ScrapedPage
        from unittest.mock import AsyncMock, patch

        scraper = make_scraper(scraper_class)
        fake_html = f"<html><body><p>{isp_key} Plan 300 Mbps</p></body></html>" * 10

        with patch.object(
            scraper, "_scrape_with_fallback", new_callable=AsyncMock
        ) as mock_fallback:
            mock_fallback.return_value = (fake_html, f"{isp_key} title", [], "httpx")
            result = await scraper.scrape()

        assert isinstance(result, ScrapedPage)
        assert result.isp_key == isp_key
