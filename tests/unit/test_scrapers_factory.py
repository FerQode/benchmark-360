# tests/unit/test_scrapers_factory.py
"""
Unit tests for the scraper factory and ISP registry.

Tests:
    - build_all_scrapers() instantiates all 8 ISPs
    - All scrapers are subclasses of BaseISPScraper
    - Registry completeness: ISP keys match scraper keys
    - Missing scraper raises NotImplementedError
    - Each scraper has valid isp_key and base_url
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scrapers import ALL_ISP_URLS, build_all_scrapers
from src.scrapers.base_scraper import BaseISPScraper
from src.utils.robots_checker import RobotsChecker


# ─────────────────────────────────────────────────────────────────
# Registry completeness
# ─────────────────────────────────────────────────────────────────


class TestAllISPUrlsRegistry:
    """Validates the central ISP URL registry is complete and correct."""

    EXPECTED_ISP_KEYS = {
        "netlife", "cnt", "claro", "xtrim",
        "ecuanet", "puntonet", "alfanet", "fibramax",
    }

    @pytest.mark.unit
    def test_registry_has_all_8_isps(self) -> None:
        assert set(ALL_ISP_URLS.keys()) == self.EXPECTED_ISP_KEYS

    @pytest.mark.unit
    def test_all_urls_start_with_https(self) -> None:
        for isp_key, url in ALL_ISP_URLS.items():
            assert url.startswith("https://"), (
                f"ISP '{isp_key}' URL must use HTTPS: {url}"
            )

    @pytest.mark.unit
    def test_all_urls_have_no_trailing_slash_issue(self) -> None:
        """URLs should be base domains without path (no trailing slashes)."""
        for isp_key, url in ALL_ISP_URLS.items():
            # Base URLs should not have paths like /planes/
            parts = url.replace("https://", "").split("/")
            assert len(parts) == 1, (
                f"ISP '{isp_key}' base URL should be domain only, got: {url}"
            )

    @pytest.mark.unit
    def test_no_duplicate_urls(self) -> None:
        urls = list(ALL_ISP_URLS.values())
        assert len(urls) == len(set(urls)), "Duplicate base URLs found in registry"

    @pytest.mark.unit
    @pytest.mark.parametrize("isp_key,expected_domain", [
        ("netlife", "netlife.ec"),
        ("cnt", "cnt.com.ec"),
        ("claro", "claro.com.ec"),
        ("xtrim", "xtrim.com.ec"),
        ("ecuanet", "ecuanet.ec"),
        ("alfanet", "alfanet.ec"),
        ("fibramax", "fibramax.ec"),
    ])
    def test_correct_domain_per_isp(
        self, isp_key: str, expected_domain: str
    ) -> None:
        assert expected_domain in ALL_ISP_URLS[isp_key]


# ─────────────────────────────────────────────────────────────────
# build_all_scrapers factory
# ─────────────────────────────────────────────────────────────────


class TestBuildAllScrapers:
    """Validates the scraper factory function."""

    @pytest.mark.unit
    def test_returns_8_scrapers(
        self,
        mock_robots_checker: MagicMock,
        tmp_data_raw: Path,
    ) -> None:
        scrapers = build_all_scrapers(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        assert len(scrapers) == 8

    @pytest.mark.unit
    def test_all_scrapers_are_base_subclasses(
        self,
        mock_robots_checker: MagicMock,
        tmp_data_raw: Path,
    ) -> None:
        scrapers = build_all_scrapers(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        for isp_key, scraper in scrapers.items():
            assert isinstance(scraper, BaseISPScraper), (
                f"Scraper for '{isp_key}' must subclass BaseISPScraper"
            )

    @pytest.mark.unit
    def test_scraper_keys_match_registry(
        self,
        mock_robots_checker: MagicMock,
        tmp_data_raw: Path,
    ) -> None:
        scrapers = build_all_scrapers(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        assert set(scrapers.keys()) == set(ALL_ISP_URLS.keys())

    @pytest.mark.unit
    def test_each_scraper_isp_key_matches_dict_key(
        self,
        mock_robots_checker: MagicMock,
        tmp_data_raw: Path,
    ) -> None:
        scrapers = build_all_scrapers(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        for isp_key, scraper in scrapers.items():
            assert scraper.isp_key == isp_key, (
                f"scraper.isp_key='{scraper.isp_key}' "
                f"must match dict key='{isp_key}'"
            )

    @pytest.mark.unit
    def test_each_scraper_has_valid_base_url(
        self,
        mock_robots_checker: MagicMock,
        tmp_data_raw: Path,
    ) -> None:
        scrapers = build_all_scrapers(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        for isp_key, scraper in scrapers.items():
            assert scraper.base_url.startswith("https://"), (
                f"Scraper '{isp_key}' base_url must use HTTPS"
            )
            assert len(scraper.base_url) > 10

    @pytest.mark.unit
    def test_data_raw_path_propagated_to_all_scrapers(
        self,
        mock_robots_checker: MagicMock,
        tmp_data_raw: Path,
    ) -> None:
        scrapers = build_all_scrapers(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        for scraper in scrapers.values():
            assert scraper.data_raw_path == tmp_data_raw

    @pytest.mark.unit
    def test_factory_raises_if_scraper_missing(
        self,
        mock_robots_checker: MagicMock,
        tmp_data_raw: Path,
    ) -> None:
        """If ALL_ISP_URLS has an entry without a scraper, must raise."""
        from unittest.mock import patch
        from src.scrapers import ALL_ISP_URLS
        with patch("src.scrapers.ALL_ISP_URLS", {**ALL_ISP_URLS, "new_isp": "https://new-isp.ec"}):
            with pytest.raises(NotImplementedError, match="new_isp"):
                build_all_scrapers(
                    robots_checker=mock_robots_checker,
                    data_raw_path=tmp_data_raw,
                )
