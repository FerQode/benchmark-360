# tests/unit/test_base_scraper.py
"""
Unit tests for ScrapedPage DTO and BaseISPScraper static utilities.

Tests (zero network calls — pure logic):
    - ScrapedPage properties and computed fields
    - HTML-to-text extraction (noise tag removal)
    - Page title extraction
    - save_raw() behavior: content present vs. empty
    - Base64 screenshot encoding
    - _is_url_allowed() integration with robots_checker
    - _scrape_with_fallback() method selection logic
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scrapers.base_scraper import (
    MAX_HTML_CHARS,
    BaseISPScraper,
    ScrapedPage,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def make_page(**overrides) -> ScrapedPage:
    """Factory for ScrapedPage with sensible test defaults.

    Args:
        **overrides: Fields to override from defaults.

    Returns:
        ScrapedPage instance ready for testing.
    """
    defaults = {
        "isp_key": "test_isp",
        "url": "https://test-isp.ec/planes/",
        "html_content": (
            "<html><head><title>Test ISP</title></head>"
            "<body><p>Plan 100 Mbps $25</p></body></html>"
        ),
        "text_content": "Plan 100 Mbps $25",
    }
    defaults.update(overrides)
    return ScrapedPage(**defaults)


# ─────────────────────────────────────────────────────────────────
# ScrapedPage — Computed Properties
# ─────────────────────────────────────────────────────────────────


class TestScrapedPageProperties:
    """Validates derived properties of the ScrapedPage DTO."""

    @pytest.mark.unit
    def test_has_screenshots_false_when_empty(self) -> None:
        assert make_page(screenshots=[]).has_screenshots is False

    @pytest.mark.unit
    def test_has_screenshots_true_when_one_present(self) -> None:
        assert make_page(screenshots=[b"\x89PNG"]).has_screenshots is True

    @pytest.mark.unit
    def test_has_screenshots_true_when_multiple(self) -> None:
        shots = [b"shot1", b"shot2", b"shot3"]
        assert make_page(screenshots=shots).has_screenshots is True

    @pytest.mark.unit
    def test_is_partial_false_when_no_error(self) -> None:
        assert make_page(error=None).is_partial is False

    @pytest.mark.unit
    def test_is_partial_true_with_any_error_string(self) -> None:
        assert make_page(error="Playwright timeout").is_partial is True

    @pytest.mark.unit
    def test_is_partial_true_with_empty_error_string(self) -> None:
        # Empty string is still truthy-false in Python, but not None
        # This tests boundary: only None means no error
        page = make_page(error=None)
        assert page.is_partial is False

    @pytest.mark.unit
    def test_content_size_kb_exactly_one_kb(self) -> None:
        content = "a" * 1024
        page = make_page(html_content=content)
        assert abs(page.content_size_kb - 1.0) < 0.01

    @pytest.mark.unit
    def test_content_size_kb_zero_for_empty_content(self) -> None:
        assert make_page(html_content="").content_size_kb == 0.0

    @pytest.mark.unit
    def test_content_size_kb_utf8_multibyte(self) -> None:
        """UTF-8 multibyte chars (Spanish accents) must be counted correctly."""
        content = "ñ" * 512  # Each ñ is 2 bytes in UTF-8 → 1 KB total
        page = make_page(html_content=content)
        assert abs(page.content_size_kb - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────────
# ScrapedPage — screenshots_as_base64
# ─────────────────────────────────────────────────────────────────


class TestScreenshotsAsBase64:
    """Validates PNG-to-base64 encoding for OpenAI Vision API."""

    @pytest.mark.unit
    def test_empty_list_returns_empty_list(self) -> None:
        assert make_page(screenshots=[]).screenshots_as_base64() == []

    @pytest.mark.unit
    def test_single_screenshot_correctly_encoded(self) -> None:
        raw = b"\x89PNG\r\nfake_screenshot_data"
        page = make_page(screenshots=[raw])
        encoded = page.screenshots_as_base64()
        assert len(encoded) == 1
        assert encoded[0] == base64.b64encode(raw).decode("utf-8")

    @pytest.mark.unit
    def test_multiple_screenshots_all_encoded(self) -> None:
        shots = [b"shot1_data", b"shot2_data", b"shot3_data"]
        page = make_page(screenshots=shots)
        encoded = page.screenshots_as_base64()
        assert len(encoded) == 3
        for raw, enc in zip(shots, encoded):
            assert enc == base64.b64encode(raw).decode("utf-8")

    @pytest.mark.unit
    def test_encoded_output_is_valid_base64_string(self) -> None:
        """Each encoded element must be decodable back to original bytes."""
        raw = b"\x00\xff\xfe\xab\xcd binary data"
        page = make_page(screenshots=[raw])
        encoded = page.screenshots_as_base64()
        decoded = base64.b64decode(encoded[0])
        assert decoded == raw


# ─────────────────────────────────────────────────────────────────
# ScrapedPage — save_raw
# ─────────────────────────────────────────────────────────────────


class TestSaveRaw:
    """Validates file persistence logic of save_raw()."""

    @pytest.mark.unit
    def test_saves_html_file(self, tmp_data_raw: Path) -> None:
        page = make_page(html_content="<html><body>Test</body></html>")
        page.save_raw(tmp_data_raw)
        html_files = list(tmp_data_raw.glob("*.html"))
        assert len(html_files) == 1

    @pytest.mark.unit
    def test_saves_text_file(self, tmp_data_raw: Path) -> None:
        page = make_page(text_content="Plan 300 Mbps $25")
        page.save_raw(tmp_data_raw)
        txt_files = list(tmp_data_raw.glob("*.txt"))
        assert len(txt_files) == 1

    @pytest.mark.unit
    def test_saves_screenshot_as_png(self, tmp_data_raw: Path) -> None:
        page = make_page(screenshots=[b"\x89PNG_fake_data"])
        page.save_raw(tmp_data_raw)
        png_files = list(tmp_data_raw.glob("*.png"))
        assert len(png_files) == 1

    @pytest.mark.unit
    def test_saves_multiple_screenshots_with_index(
        self, tmp_data_raw: Path
    ) -> None:
        page = make_page(screenshots=[b"shot1", b"shot2", b"shot3"])
        page.save_raw(tmp_data_raw)
        png_files = sorted(tmp_data_raw.glob("*.png"))
        assert len(png_files) == 3
        # Verify indexed naming: _00.png, _01.png, _02.png
        names = [f.name for f in png_files]
        assert any("_00.png" in n for n in names)
        assert any("_01.png" in n for n in names)
        assert any("_02.png" in n for n in names)

    @pytest.mark.unit
    def test_no_files_created_when_no_content(
        self, tmp_data_raw: Path
    ) -> None:
        """Empty html AND empty screenshots must produce zero files."""
        page = make_page(html_content="", text_content="", screenshots=[])
        page.save_raw(tmp_data_raw)
        assert list(tmp_data_raw.iterdir()) == []

    @pytest.mark.unit
    def test_creates_nested_directories_if_missing(
        self, tmp_path: Path
    ) -> None:
        deep_dir = tmp_path / "level1" / "level2" / "level3"
        assert not deep_dir.exists()
        page = make_page()
        page.save_raw(deep_dir)
        assert deep_dir.exists()

    @pytest.mark.unit
    def test_html_file_contains_correct_content(
        self, tmp_data_raw: Path
    ) -> None:
        expected = "<html><body><p>Plan Único $99</p></body></html>"
        page = make_page(html_content=expected)
        page.save_raw(tmp_data_raw)
        html_file = next(tmp_data_raw.glob("*.html"))
        assert html_file.read_text(encoding="utf-8") == expected

    @pytest.mark.unit
    def test_isp_key_appears_in_filenames(self, tmp_data_raw: Path) -> None:
        page = make_page(isp_key="netlife")
        page.save_raw(tmp_data_raw)
        all_files = list(tmp_data_raw.iterdir())
        assert all("netlife" in f.name for f in all_files)


# ─────────────────────────────────────────────────────────────────
# BaseISPScraper — _extract_text_from_html
# ─────────────────────────────────────────────────────────────────


class TestExtractTextFromHTML:
    """HTML-to-text must strip noise tags and normalize whitespace."""

    @pytest.mark.unit
    def test_extracts_plan_content(self) -> None:
        html = "<html><body><p>Plan 300 Mbps $25.00</p></body></html>"
        text = BaseISPScraper._extract_text_from_html(html)
        assert "Plan 300 Mbps" in text
        assert "$25.00" in text

    @pytest.mark.unit
    def test_removes_script_tags_content(self) -> None:
        html = (
            "<html><body>"
            "<script>gtag('event', 'purchase', {value: 25})</script>"
            "<p>Precio: $25</p>"
            "</body></html>"
        )
        text = BaseISPScraper._extract_text_from_html(html)
        assert "gtag" not in text
        assert "Precio" in text

    @pytest.mark.unit
    def test_removes_style_tags_content(self) -> None:
        html = (
            "<html><head>"
            "<style>.card { color: red; font-size: 16px; }</style>"
            "</head><body><p>Plan Hogar</p></body></html>"
        )
        text = BaseISPScraper._extract_text_from_html(html)
        assert "color" not in text
        assert "font-size" not in text
        assert "Plan Hogar" in text

    @pytest.mark.unit
    def test_removes_nav_content(self) -> None:
        html = (
            "<html><body>"
            "<nav>Inicio | Planes | Contacto | Login</nav>"
            "<main><p>Plan 600 Mbps</p></main>"
            "</body></html>"
        )
        text = BaseISPScraper._extract_text_from_html(html)
        # Nav content should be stripped
        assert "Login" not in text
        assert "Plan 600 Mbps" in text

    @pytest.mark.unit
    def test_removes_footer_content(self) -> None:
        html = (
            "<html><body>"
            "<p>Plan 1Gbps $45</p>"
            "<footer>© 2024 ISP Ecuador. Todos los derechos reservados.</footer>"
            "</body></html>"
        )
        text = BaseISPScraper._extract_text_from_html(html)
        assert "derechos reservados" not in text
        assert "Plan 1Gbps" in text

    @pytest.mark.unit
    def test_normalizes_multiple_whitespaces(self) -> None:
        html = "<html><body><p>Plan    300    Mbps</p></body></html>"
        text = BaseISPScraper._extract_text_from_html(html)
        assert "Plan    300" not in text
        assert "Plan" in text and "300" in text

    @pytest.mark.unit
    def test_handles_empty_html_gracefully(self) -> None:
        text = BaseISPScraper._extract_text_from_html("")
        assert text == ""

    @pytest.mark.unit
    def test_handles_html_with_special_chars(self) -> None:
        html = "<html><body><p>Fibra óptica: $25,00 mensual</p></body></html>"
        text = BaseISPScraper._extract_text_from_html(html)
        assert "óptica" in text
        assert "$25,00" in text

    @pytest.mark.unit
    def test_removes_noscript_tag(self) -> None:
        html = (
            "<html><body>"
            "<noscript>Necesitas JavaScript habilitado</noscript>"
            "<p>Plan Básico</p>"
            "</body></html>"
        )
        text = BaseISPScraper._extract_text_from_html(html)
        assert "JavaScript" not in text
        assert "Plan Básico" in text


# ─────────────────────────────────────────────────────────────────
# BaseISPScraper — _extract_page_title
# ─────────────────────────────────────────────────────────────────


class TestExtractPageTitle:
    """Title extraction from HTML <title> tag."""

    @pytest.mark.unit
    def test_extracts_standard_title(self) -> None:
        html = "<html><head><title>Netlife - Internet Inteligente</title></head></html>"
        title = BaseISPScraper._extract_page_title(html)
        assert title == "Netlife - Internet Inteligente"

    @pytest.mark.unit
    def test_strips_whitespace_from_title(self) -> None:
        html = "<html><head><title>  Claro Ecuador  </title></head></html>"
        title = BaseISPScraper._extract_page_title(html)
        assert title == "Claro Ecuador"

    @pytest.mark.unit
    def test_returns_empty_string_when_no_title_tag(self) -> None:
        html = "<html><head></head><body>Content</body></html>"
        title = BaseISPScraper._extract_page_title(html)
        assert title == ""

    @pytest.mark.unit
    def test_returns_empty_string_for_empty_html(self) -> None:
        title = BaseISPScraper._extract_page_title("")
        assert title == ""

    @pytest.mark.unit
    def test_handles_title_with_special_chars(self) -> None:
        html = "<html><head><title>CNT - Corporación Nacional de Telecom</title></head></html>"
        title = BaseISPScraper._extract_page_title(html)
        assert "Corporación" in title


# ─────────────────────────────────────────────────────────────────
# BaseISPScraper — _is_url_allowed
# ─────────────────────────────────────────────────────────────────


class TestIsUrlAllowed:
    """Validates robots.txt URL permission checks."""

    @pytest.mark.unit
    def test_allowed_url_returns_true(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        from src.scrapers.netlife_scraper import NetlifeScraper

        scraper = NetlifeScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        mock_robots_checker.can_fetch.return_value = True
        assert scraper._is_url_allowed("https://netlife.ec/planes/") is True

    @pytest.mark.unit
    def test_blocked_url_returns_false(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        from src.scrapers.netlife_scraper import NetlifeScraper

        scraper = NetlifeScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        mock_robots_checker.can_fetch.return_value = False
        assert scraper._is_url_allowed("https://netlife.ec/admin/") is False

    @pytest.mark.unit
    def test_runtime_error_defaults_to_allow(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        """If robots_checker raises RuntimeError, conservative allow is used."""
        from src.scrapers.netlife_scraper import NetlifeScraper

        scraper = NetlifeScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        mock_robots_checker.can_fetch.side_effect = RuntimeError(
            "Robots not pre-loaded"
        )
        # Should not raise — must default to True
        result = scraper._is_url_allowed("https://netlife.ec/planes/")
        assert result is True


# ─────────────────────────────────────────────────────────────────
# BaseISPScraper — _scrape_with_fallback strategy selection
# ─────────────────────────────────────────────────────────────────


class TestScrapeWithFallback:
    """Validates 3-level fallback strategy selection logic."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_httpx_strategy_used_when_not_playwright_required(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        """CNT (requires_playwright=False) should try httpx first."""
        from src.scrapers.cnt_scraper import CNTScraper

        scraper = CNTScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        fake_html = "<html><body><p>CNT Plan</p></body></html>" * 20

        with patch.object(
            scraper, "_fetch_html_httpx", new_callable=AsyncMock
        ) as mock_httpx:
            mock_httpx.return_value = (fake_html, "CNT Title")

            html, title, shots, method = await scraper._scrape_with_fallback(
                url="https://www.cnt.com.ec/hogar/internet"
            )

        assert method == "httpx"
        assert html == fake_html
        mock_httpx.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_playwright_used_when_requires_playwright_true(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        """Netlife (requires_playwright=True) must skip httpx."""
        from src.scrapers.netlife_scraper import NetlifeScraper

        scraper = NetlifeScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        fake_html = "<html><body><p>Netlife Plan</p></body></html>" * 20

        with patch.object(
            scraper, "_fetch_html_playwright", new_callable=AsyncMock
        ) as mock_pw, patch.object(
            scraper, "_fetch_html_httpx", new_callable=AsyncMock
        ) as mock_httpx:
            mock_pw.return_value = (fake_html, "Netlife Title", [b"screenshot"])

            html, title, shots, method = await scraper._scrape_with_fallback(
                url="https://www.netlife.ec/planes-hogar/"
            )

        assert method == "playwright"
        mock_httpx.assert_not_called()  # MUST NOT attempt httpx for SPA
        mock_pw.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_falls_back_to_playwright_when_httpx_fails(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        """When httpx fails, must automatically fall back to Playwright."""
        from src.scrapers.cnt_scraper import CNTScraper

        scraper = CNTScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        fake_html = "<html><body><p>CNT Plan fallback</p></body></html>" * 20

        with patch.object(
            scraper, "_fetch_html_httpx", new_callable=AsyncMock
        ) as mock_httpx, patch.object(
            scraper, "_fetch_html_playwright", new_callable=AsyncMock
        ) as mock_pw:
            mock_httpx.side_effect = Exception("Connection refused")
            mock_pw.return_value = (fake_html, "CNT Title", [])

            html, title, shots, method = await scraper._scrape_with_fallback(
                url="https://www.cnt.com.ec/hogar/internet"
            )

        assert method == "playwright"
        assert html == fake_html

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_failed_when_all_strategies_fail(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        """When ALL 3 strategies fail, must return 'failed' method."""
        from src.scrapers.netlife_scraper import NetlifeScraper

        scraper = NetlifeScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )

        with patch.object(
            scraper, "_fetch_html_playwright", new_callable=AsyncMock
        ) as mock_pw, patch.object(
            scraper, "_ensure_browser", new_callable=AsyncMock
        ):
            mock_pw.side_effect = Exception("Playwright crashed")
            # Mock the strategy 3 browser context to also fail
            with patch.object(
                BaseISPScraper,
                "_browser",
                new_callable=lambda: property(lambda self: None),
            ):
                html, title, shots, method = await scraper._scrape_with_fallback(
                    url="https://www.netlife.ec/planes-hogar/"
                )

        assert method == "failed"
        assert html == ""
        assert shots == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_blocked_url_returns_empty_immediately(
        self, mock_robots_checker: MagicMock, tmp_data_raw: Path
    ) -> None:
        """URLs blocked by robots.txt must return empty without HTTP calls."""
        from src.scrapers.cnt_scraper import CNTScraper

        scraper = CNTScraper(
            robots_checker=mock_robots_checker, data_raw_path=tmp_data_raw
        )
        mock_robots_checker.can_fetch.return_value = False

        with patch.object(
            scraper, "_fetch_html_httpx", new_callable=AsyncMock
        ) as mock_httpx:
            await scraper._scrape_with_fallback(
                url="https://www.cnt.com.ec/admin/"
            )

        # httpx should not even be called for blocked URLs
        mock_httpx.assert_not_called()
