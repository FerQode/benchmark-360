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

    Example:
        >>> page = ScrapedPage(
        ...     isp_key="netlife",
        ...     url="https://netlife.ec/planes-hogar",
        ...     html_content="<html>Plan Duo 300Mbps...</html>",
        ...     text_content="Plan Duo 300Mbps $35.90 mensual",
        ...     screenshots=[b"\\x89PNG..."],
        ...     scraping_method="playwright",
        ... )
        >>> page.content_size_kb
        0.04
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
        """Encode all screenshots to base64 for OpenAI Vision API.

        Returns:
            List of base64-encoded PNG strings, ready for
            inclusion in a GPT-4o vision message payload.
        """
        return [
            base64.b64encode(shot).decode("utf-8")
            for shot in self.screenshots
        ]

    def save_raw(self, output_dir: Path) -> None:
        """Persist raw content to disk for debugging and audit trail.

        Creates timestamped files so multiple runs don't overwrite
        each other. This supports the trazabilidad histórica
        objective from the hackathon requirements.

        Args:
            output_dir: Target directory for raw content files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = self.scraped_at.strftime("%Y%m%d_%H%M%S")

        # Save HTML
        html_path = output_dir / f"{self.isp_key}_{ts}.html"
        html_path.write_text(self.html_content, encoding="utf-8")

        # Save text
        txt_path = output_dir / f"{self.isp_key}_{ts}.txt"
        txt_path.write_text(self.text_content, encoding="utf-8")

        # Save screenshots
        for i, shot in enumerate(self.screenshots):
            shot_path = (
                output_dir / f"{self.isp_key}_{ts}_screenshot_{i:02d}.png"
            )
            shot_path.write_bytes(shot)

        logger.info(
            f"[{self.isp_key}] 💾 Raw saved → {output_dir} "
            f"({self.content_size_kb:.1f} KB HTML, "
            f"{len(self.screenshots)} screenshots)"
        )


# ─────────────────────────────────────────────────────────────────
# Abstract Base Class — Template Method Pattern
# ─────────────────────────────────────────────────────────────────


class BaseISPScraper(ABC):
    """Abstract base implementing Template Method + Strategy patterns.

    Provides the algorithm skeleton for all ISP scrapers:
        1. Verify robots.txt compliance (always)
        2. Choose scraping strategy (httpx / playwright / screenshot)
        3. Apply polite delay (always)
        4. Execute fetch with retry logic
        5. Extract and clean text content
        6. Capture screenshots if needed
        7. Return ScrapedPage DTO

    Concrete subclasses override ONLY the parts that differ:
        - get_plan_urls(): Which URLs to target
        - requires_playwright(): Which strategy to use
        - scrape(): Full workflow (calls base helpers)

    Class Attributes:
        _browser: Shared Playwright browser (Singleton pattern).
            One browser process shared across all ISP scrapers
            to save memory and startup time.
        _BASE_HEADERS: HTTP headers mimicking a real browser.
            Identifies as Benchmark360Bot for transparency.

    Args:
        isp_key: Pipeline-internal ISP identifier.
        base_url: Root URL of the ISP website.
        robots_checker: Pre-initialized and pre-loaded compliance checker.
        delay_range: (min_sec, max_sec) random delay between requests.
        data_raw_path: Directory for raw content persistence.
        max_retries: Retry attempts on transient failures.

    Example:
        >>> checker = RobotsChecker()
        >>> await checker.analyze("https://ecuanet.ec")
        >>> scraper = EcuanetScraper(
        ...     robots_checker=checker,
        ...     delay_range=(2.0, 4.0),
        ... )
        >>> page = await scraper.scrape()
    """

    # ── Class-level Singleton for Playwright browser ──────────────
    _browser: ClassVar = None
    _playwright: ClassVar = None

    _BASE_HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": PIPELINE_USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
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

    # ── Abstract Interface (must implement) ───────────────────────

    @abstractmethod
    async def scrape(self) -> ScrapedPage:
        """Execute the complete scraping workflow for this ISP.

        Returns:
            ScrapedPage DTO with all extracted content.
        """
        ...

    @abstractmethod
    def get_plan_urls(self) -> list[str]:
        """Return all URLs containing plan pricing information.

        Returns:
            Ordered list of URLs to scrape for this ISP.
        """
        ...

    @abstractmethod
    def requires_playwright(self) -> bool:
        """Declare whether this ISP requires JavaScript rendering.

        Returns:
            True if Playwright needed, False if httpx suffices.
        """
        ...

    # ── Shared Infrastructure Methods ─────────────────────────────

    async def _polite_delay(self) -> None:
        """Apply random jitter delay respecting crawl-delay directives.

        Computes delay as: max(crawl_delay_from_robots, min_delay) + jitter
        This ensures we never violate any site's crawl-delay while
        also adding unpredictable timing to avoid rate limiting.
        """
        # Honor robots.txt crawl-delay if available
        base = f"{self.base_url.rstrip('/')}"
        if base in self.robots_checker._cache:
            min_delay = self.robots_checker._cache[base].effective_delay
        else:
            min_delay = self.delay_range[0]

        max_delay = max(min_delay + 1.0, self.delay_range[1])
        delay = random.uniform(min_delay, max_delay)
        self._request_count += 1

        logger.debug(
            f"[{self.isp_key}] ⏱️  Request #{self._request_count} → "
            f"sleeping {delay:.2f}s"
        )
        await asyncio.sleep(delay)

    def _is_url_allowed(self, url: str) -> bool:
        """Check URL against robots.txt before any HTTP request.

        Args:
            url: Full URL to validate.

        Returns:
            True if scraping is permitted.
        """
        try:
            return self.robots_checker.can_fetch(url)
        except RuntimeError:
            logger.warning(
                f"[{self.isp_key}] ⚠️  robots.txt not pre-loaded for "
                f"{url}. Using conservative allow."
            )
            return True

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        """Extract clean plain text from HTML using BeautifulSoup.

        Removes all tags, scripts, styles and normalizes whitespace.
        This text is what gets sent to the LLM text processor.

        Args:
            html: Raw HTML string to parse.

        Returns:
            Clean, normalized plain text string.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "noscript", "iframe",
                          "header", "footer", "nav", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)

        return text.strip()

    @staticmethod
    def _extract_page_title(html: str) -> str:
        """Extract the page title from HTML.

        Args:
            html: Raw HTML string.

        Returns:
            Title text or empty string if not found.
        """
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else ""

    @retry(
        retry=retry_if_exception_type(
            (httpx.RequestError, httpx.HTTPStatusError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        reraise=True,
    )
    async def _fetch_html_httpx(self, url: str) -> tuple[str, str]:
        """Fetch page HTML using httpx (Strategy Level 1).

        Fastest strategy — direct HTTP request without browser.
        Use for sites with static HTML (no JavaScript required).

        Args:
            url: Target URL to fetch.

        Returns:
            Tuple of (html_content, page_title).
            Returns ('', '') if URL is disallowed.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx after retries.
            httpx.RequestError: On network failure after retries.
        """
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

            html = response.text[:MAX_HTML_CHARS]  # Size limit
            title = self._extract_page_title(html)

            logger.info(
                f"[{self.isp_key}] ✅ httpx → {url} "
                f"(HTTP {response.status_code}, "
                f"{len(html) / 1024:.1f} KB)"
            )
            return html, title

    async def _fetch_html_playwright(
        self,
        url: str,
        take_screenshot: bool = True,
        wait_selector: str | None = None,
    ) -> tuple[str, str, list[bytes]]:
        """Fetch page using Playwright for JS-rendered content (Strategy Level 2).

        Launches or reuses a shared Chromium browser instance,
        navigates to the URL, waits for network idle, and optionally
        captures a full-page screenshot.

        Args:
            url: Target URL to fetch.
            take_screenshot: If True, captures full-page screenshot.
            wait_selector: Optional CSS selector to wait for before
                capturing content (e.g., '.plan-card' for plan cards).

        Returns:
            Tuple of (html_content, page_title, screenshots_list).
            Returns ('', '', []) if URL is disallowed.

        Raises:
            RuntimeError: If Playwright browser fails to launch.
        """
        if not self._is_url_allowed(url):
            return "", "", []

        await self._polite_delay()

        # Lazy-initialize shared browser (Singleton)
        await self._ensure_browser()

        screenshots: list[bytes] = []

        try:
            context = await BaseISPScraper._browser.new_context(
                viewport={
                    "width": MAX_SCREENSHOT_WIDTH,
                    "height": 900,
                },
                locale="es-EC",
                timezone_id="America/Guayaquil",
                extra_http_headers=self._BASE_HEADERS,
            )
            page = await context.new_page()

            # Navigate with timeout
            await page.goto(
                url,
                timeout=PLAYWRIGHT_NAVIGATION_TIMEOUT,
                wait_until="networkidle",
            )

            # Wait additional time for lazy-loaded content
            await asyncio.sleep(PLAYWRIGHT_WAIT_FOR_NETWORK_IDLE / 1000)

            # Wait for specific element if requested
            if wait_selector:
                try:
                    await page.wait_for_selector(
                        wait_selector,
                        timeout=10_000,
                    )
                    logger.debug(
                        f"[{self.isp_key}] Selector '{wait_selector}' found"
                    )
                except Exception:
                    logger.warning(
                        f"[{self.isp_key}] Selector '{wait_selector}' "
                        f"not found — continuing anyway"
                    )

            # Get full rendered HTML
            html = await page.content()
            html = html[:MAX_HTML_CHARS]
            title = await page.title()

            logger.info(
                f"[{self.isp_key}] ✅ playwright → {url} "
                f"({len(html) / 1024:.1f} KB, title='{title[:50]}')"
            )

            # Capture screenshot(s)
            if take_screenshot:
                screenshot_bytes = await page.screenshot(
                    full_page=True,
                    type="png",
                )
                screenshots.append(screenshot_bytes)
                logger.debug(
                    f"[{self.isp_key}] 📸 Screenshot: "
                    f"{len(screenshot_bytes) / 1024:.1f} KB"
                )

            await context.close()
            return html, title, screenshots

        except Exception as exc:
            logger.error(
                f"[{self.isp_key}] ❌ Playwright failed for {url}: {exc}"
            )
            return "", "", []

    async def _scrape_with_fallback(
        self,
        url: str,
        wait_selector: str | None = None,
    ) -> tuple[str, str, list[bytes], str]:
        """Execute 3-level scraping strategy with automatic fallback.

        Tries strategies in order: httpx → playwright → screenshot-only.
        Falls back gracefully if a strategy fails.

        Strategy selection:
            - If requires_playwright() is False → start with httpx
            - If requires_playwright() is True → start with playwright
            - If both fail → screenshot-only as last resort

        Args:
            url: Target URL to scrape.
            wait_selector: CSS selector to wait for in Playwright.

        Returns:
            Tuple of (html, title, screenshots, method_used).
        """
        if not self.requires_playwright():
            # Strategy 1: httpx (fast path for static sites)
            try:
                html, title = await self._fetch_html_httpx(url)
                if html:
                    return html, title, [], "httpx"
            except Exception as exc:
                logger.warning(
                    f"[{self.isp_key}] httpx failed: {exc}. "
                    "Falling back to Playwright..."
                )

        # Strategy 2: Playwright (JS-rendered)
        try:
            html, title, shots = await self._fetch_html_playwright(
                url,
                take_screenshot=True,
                wait_selector=wait_selector,
            )
            if html:
                method = "playwright"
                return html, title, shots, method
        except Exception as exc:
            logger.warning(
                f"[{self.isp_key}] Playwright failed: {exc}. "
                "Using screenshot-only fallback..."
            )

        # Strategy 3: Screenshot only (Vision LLM will handle it)
        try:
            _, _, shots = await self._fetch_html_playwright(
                url,
                take_screenshot=True,
            )
            if shots:
                logger.info(
                    f"[{self.isp_key}] 📸 Screenshot-only fallback "
                    f"for {url} — Vision LLM will process"
                )
                return "", "", shots, "playwright_screenshot"
        except Exception as exc:
            logger.error(
                f"[{self.isp_key}] All strategies failed for {url}: {exc}"
            )

        return "", "", [], "failed"

    @classmethod
    async def _ensure_browser(cls) -> None:
        """Lazily initialize the shared Playwright browser instance.

        Uses Singleton pattern — only one browser process runs
        for the entire pipeline execution, shared across all
        8 ISP scrapers for memory efficiency.

        Raises:
            RuntimeError: If Playwright is not installed.
        """
        if cls._browser is None:
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
                logger.info(
                    "[browser] ✅ Chromium launched (shared singleton)"
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Playwright not installed. Run: uv run playwright install chromium"
                ) from exc

    @classmethod
    async def close_browser(cls) -> None:
        """Close the shared browser and cleanup Playwright resources.

        Must be called at the end of pipeline execution to
        prevent resource leaks. Called by the orchestrator.
        """
        if cls._browser:
            await cls._browser.close()
            cls._browser = None
        if cls._playwright:
            await cls._playwright.stop()
            cls._playwright = None
            logger.info("[browser] 🔒 Chromium closed and cleaned up")
