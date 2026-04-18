# src/scrapers/base_scraper.py
"""
Abstract base scraper implementing Template Method + Strategy patterns.

Defines the contract all ISP scrapers must fulfill and provides
shared infrastructure: polite delays, robots.txt compliance,
3-level fallback strategy, and Playwright browser lifecycle.

Design Patterns Applied:
    - Template Method: scrape() defines the algorithm skeleton
    - Strategy: 3 interchangeable scraping strategies
    - Singleton: shared Playwright browser instance

Typical usage example:
    >>> from src.scrapers.netlife_scraper import NetlifeScraper
    >>> scraper = NetlifeScraper(robots_checker=checker)
    >>> page = await scraper.scrape()
    >>> page.scraping_method
    'playwright'
"""

from __future__ import annotations

import asyncio
import base64
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.robots_checker import PIPELINE_USER_AGENT, RobotsChecker

# ─────────────────────────────────────────────────────────────────
# Constantes del scraping layer
# ─────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_SECONDS: float = 30.0
PLAYWRIGHT_NAVIGATION_TIMEOUT: int = 45_000  # ms
PLAYWRIGHT_WAIT_FOR_NETWORK_IDLE: int = 3_000  # ms after load
MAX_SCREENSHOT_WIDTH: int = 1_920
MAX_SCREENSHOT_HEIGHT: int = 10_000  # Tall pages for full-page capture
MAX_HTML_CHARS: int = 500_000  # 500KB HTML limit


# ─────────────────────────────────────────────────────────────────
# DTO: ScrapedPage — Data Transfer Object
# ─────────────────────────────────────────────────────────────────


@dataclass
class ScrapedPage:
    """Data Transfer Object carrying all raw content from one ISP website.

    This is the contract between the Scraping Layer and the
    LLM Processing Layer. Contains everything needed for both
    text extraction (html_content) and vision extraction (screenshots).

    Attributes:
        isp_key: Pipeline-internal ISP identifier. Lowercase, no spaces.
        url: Primary URL that was scraped.
        html_content: Full page HTML after JavaScript execution.
        text_content: Cleaned plain text extracted from HTML.
        screenshots: Full-page PNG screenshots as bytes list.
            Multiple screenshots if page was scrolled or has tabs.
        additional_urls: Secondary URLs also scraped for this ISP.
        scraped_at: Microsecond-precise extraction timestamp.
        scraping_method: Strategy that succeeded.
            One of: 'httpx', 'playwright', 'playwright_screenshot'.
        page_title: Content of HTML <title> tag.
        error: Non-fatal error description, or None if clean.
    """

    isp_key: str
    url: str
    html_content: str
    text_content: str
    screenshots: list[bytes] = field(default_factory=list)
    additional_urls: list[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.now)
    scraping_method: str = "unknown"
    page_title: str = ""
    error: str | None = None

    @property
    def has_screenshots(self) -> bool:
        """True if at least one screenshot was captured."""
        return len(self.screenshots) > 0

    @property
    def content_size_kb(self) -> float:
        """Approximate HTML content size in kilobytes."""
        return len(self.html_content.encode("utf-8")) / 1024

    @property
    def is_partial(self) -> bool:
        """True if scraping completed with non-fatal errors."""
        return self.error is not None

    def screenshots_as_base64(self) -> list[str]:
        """Encode all screenshots to base64 for OpenAI Vision API."""
        return [
            base64.b64encode(shot).decode("utf-8")
            for shot in self.screenshots
        ]

    def save_raw(self, output_dir: Path) -> None:
        """Persist raw content to disk for debugging and audit trail."""
        if not self.html_content and not self.screenshots:
            logger.warning("[{}] No content to save — skipping save_raw()", self.isp_key)
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        ts = self.scraped_at.strftime("%Y%m%d_%H%M%S")

        if self.html_content:
            (output_dir / f"{self.isp_key}_{ts}.html").write_text(
                self.html_content, encoding="utf-8"
            )
        if self.text_content:
            (output_dir / f"{self.isp_key}_{ts}.txt").write_text(
                self.text_content, encoding="utf-8"
            )
        for i, shot in enumerate(self.screenshots):
            (output_dir / f"{self.isp_key}_{ts}_screenshot_{i:02d}.png").write_bytes(shot)

        logger.info("[{}] 💾 Raw saved → {} ({:.1f} KB, {} shots)",
                    self.isp_key, output_dir, self.content_size_kb, len(self.screenshots))


# ─────────────────────────────────────────────────────────────────
# Abstract Base Class — Template Method Pattern
# ─────────────────────────────────────────────────────────────────


class BaseISPScraper(ABC):
    """Abstract base implementing Template Method + Strategy patterns.

    Provides the algorithm skeleton for all ISP scrapers.
    Concrete subclasses override ONLY the parts that differ:
        - get_plan_urls()
        - requires_playwright()
        - scrape()
    """

    _browser: ClassVar = None
    _playwright: ClassVar = None
    _browser_lock: ClassVar[asyncio.Lock | None] = None

    _BASE_HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": PIPELINE_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-EC,es;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "DNT": "1",
    }

    def __init__(
        self,
        isp_key: str,
        base_url: str,
        robots_checker: RobotsChecker,
        delay_range: tuple[float, float] = (2.0, 5.0),
        data_raw_path: Path = Path("data/raw"),
        max_retries: int = 3,
    ) -> None:
        self.isp_key = isp_key
        self.base_url = base_url
        self.robots_checker = robots_checker
        self.delay_range = delay_range
        self.data_raw_path = data_raw_path
        self.max_retries = max_retries
        self._request_count: int = 0

    @abstractmethod
    async def scrape(self) -> ScrapedPage:
        ...

    @abstractmethod
    def get_plan_urls(self) -> list[str]:
        ...

    @abstractmethod
    def requires_playwright(self) -> bool:
        ...

    @classmethod
    def _get_browser_lock(cls) -> asyncio.Lock:
        """Get or create the browser initialization lock lazily."""
        if cls._browser_lock is None:
            cls._browser_lock = asyncio.Lock()
        return cls._browser_lock

    async def _polite_delay(self) -> None:
        """Apply random jitter delay respecting crawl-delay directives."""
        min_delay = self.robots_checker.get_crawl_delay(self.base_url)
        max_delay = max(min_delay + 1.0, self.delay_range[1])
        delay = random.uniform(min_delay, max_delay)
        self._request_count += 1
        
        logger.debug("[{}] ⏱️  Request #{} → sleeping {:.2f}s",
                     self.isp_key, self._request_count, delay)
        await asyncio.sleep(delay)

    def _is_url_allowed(self, url: str) -> bool:
        """Check URL against robots.txt before any HTTP request."""
        try:
            return self.robots_checker.can_fetch(url)
        except RuntimeError:
            logger.warning("[{}] ⚠️  robots.txt not pre-loaded for {}. Using conservative allow.",
                           self.isp_key, url)
            return True

    @staticmethod
    def _extract_text_from_html(html_content: str) -> str:
        """Extract clean plain text from HTML using BeautifulSoup."""
        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    @staticmethod
    def _extract_page_title(html_content: str) -> str:
        """Extract the page title from HTML."""
        soup = BeautifulSoup(html_content, "lxml")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else ""

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        reraise=True,
    )
    async def _fetch_html_httpx(self, url: str) -> tuple[str, str]:
        """Fetch page HTML using httpx (Strategy Level 1)."""
        if not self._is_url_allowed(url):
            return "", ""

        await self._polite_delay()

        async with httpx.AsyncClient(
            headers=self._BASE_HEADERS,
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            html = response.text[:MAX_HTML_CHARS]
            title = self._extract_page_title(html)

            logger.info("[{}] ✅ httpx → {} (HTTP {}, {:.1f} KB)",
                        self.isp_key, url, response.status_code, len(html) / 1024)
            return html, title

    async def _fetch_html_playwright(
        self,
        url: str,
        take_screenshot: bool = True,
        wait_selector: str | None = None,
    ) -> tuple[str, str, list[bytes]]:
        """Fetch page using Playwright for JS-rendered content (Strategy Level 2)."""
        if not self._is_url_allowed(url):
            return "", "", []

        await self._polite_delay()
        await self._ensure_browser()

        screenshots: list[bytes] = []
        context = None

        try:
            context = await BaseISPScraper._browser.new_context(
                viewport={"width": MAX_SCREENSHOT_WIDTH, "height": 900},
                locale="es-EC",
                timezone_id="America/Guayaquil",
                extra_http_headers=self._BASE_HEADERS,
            )
            page = await context.new_page()

            await page.goto(url, timeout=PLAYWRIGHT_NAVIGATION_TIMEOUT, wait_until="networkidle")
            await asyncio.sleep(PLAYWRIGHT_WAIT_FOR_NETWORK_IDLE / 1000)

            if wait_selector:
                try:
                    await page.wait_for_selector(wait_selector, timeout=10_000)
                    logger.debug("[{}] Selector '{}' found", self.isp_key, wait_selector)
                except Exception:
                    logger.warning("[{}] Selector '{}' not found — continuing anyway",
                                   self.isp_key, wait_selector)

            html = (await page.content())[:MAX_HTML_CHARS]
            title = await page.title()

            logger.info("[{}] ✅ playwright → {} ({:.1f} KB, title='{}')",
                        self.isp_key, url, len(html) / 1024, title[:50])

            if take_screenshot:
                screenshot_bytes = await page.screenshot(full_page=True, type="png")
                screenshots.append(screenshot_bytes)
                logger.debug("[{}] 📸 Screenshot: {:.1f} KB",
                             self.isp_key, len(screenshot_bytes) / 1024)

            return html, title, screenshots

        except Exception as exc:
            logger.error("[{}] ❌ Playwright failed for {}: {}", self.isp_key, url, exc)
            return "", "", []

        finally:
            if context:
                await context.close()

    async def _scrape_with_fallback(
        self,
        url: str,
        wait_selector: str | None = None,
    ) -> tuple[str, str, list[bytes], str]:
        """Execute 3-level scraping strategy with automatic fallback."""
        
        # ── Strategy 1: httpx (only for non-JS sites) ─────────────
        if not self.requires_playwright():
            try:
                html, title = await self._fetch_html_httpx(url)
                if html and len(html) > 500:  # Sanity: not empty/redirect page
                    return html, title, [], "httpx"
                logger.warning("[{}] httpx returned insufficient content ({} chars)",
                               self.isp_key, len(html))
            except Exception as exc:
                logger.warning("[{}] httpx failed: {} → falling back to Playwright",
                               self.isp_key, exc)

        # ── Strategy 2: Playwright (full HTML + screenshot) ────────
        try:
            html, title, shots = await self._fetch_html_playwright(
                url, take_screenshot=True, wait_selector=wait_selector,
            )
            if html:
                return html, title, shots, "playwright"
            elif shots:
                logger.info("[{}] Playwright got screenshot but no HTML for {}",
                            self.isp_key, url)
                return "", "", shots, "playwright_screenshot"
        except Exception as exc:
            logger.warning("[{}] Playwright failed: {} → screenshot-only attempt",
                           self.isp_key, exc)

        # ── Strategy 3: Screenshot only (different approach) ──────
        try:
            await self._ensure_browser()
            context = await BaseISPScraper._browser.new_context(
                viewport={"width": 1280, "height": 900},
            )
            page = await context.new_page()
            await page.goto(url, timeout=20_000, wait_until="load")
            screenshot = await page.screenshot(full_page=True, type="png")
            await context.close()

            if screenshot:
                logger.info("[{}] 📸 Screenshot-only fallback for {}", self.isp_key, url)
                return "", "", [screenshot], "playwright_screenshot"
        except Exception as exc:
            logger.error("[{}] All 3 strategies failed for {}: {}", self.isp_key, url, exc)

        return "", "", [], "failed"

    @classmethod
    async def _ensure_browser(cls) -> None:
        """Lazily initialize the shared Playwright browser (thread-safe)."""
        if cls._browser is not None:
            return

        async with cls._get_browser_lock():
            if cls._browser is not None:
                return

            try:
                from playwright.async_api import async_playwright

                cls._playwright = await async_playwright().start()
                cls._browser = await cls._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-extensions",
                        "--disable-background-networking",
                        "--no-first-run",
                        "--disable-default-apps",
                    ],
                )
                logger.info("[browser] ✅ Chromium launched (shared singleton)")
            except ImportError as exc:
                raise RuntimeError(
                    "Playwright not installed. Run: uv run playwright install chromium"
                ) from exc

    @classmethod
    async def close_browser(cls) -> None:
        """Close the shared browser and cleanup Playwright resources."""
        if cls._browser:
            await cls._browser.close()
            cls._browser = None
        if cls._playwright:
            await cls._playwright.stop()
            cls._playwright = None
            logger.info("[browser] 🔒 Chromium closed and cleaned up")
