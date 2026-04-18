# src/scrapers/netlife_scraper.py
"""
Netlife ISP scraper — Playwright + Vision LLM strategy.

Netlife (operated by MEGADATOS S.A.) uses a JavaScript-heavy
React frontend with plans displayed in dynamic card components.
Some promotional content appears as CSS background images or
<img> banners that require Vision LLM processing.

Scraping Strategy:
    Primary:  Playwright (networkidle wait + .plan-card selector)
    Fallback: Full-page screenshot → Vision LLM

Target URLs:
    Main:   https://netlife.ec/
    Planes: https://netlife.ec/planes-hogar (if exists)

Typical usage example:
    >>> scraper = NetlifeScraper(robots_checker=checker)
    >>> page = await scraper.scrape()
    >>> print(f"Method: {page.scraping_method}")
    >>> print(f"Content: {page.content_size_kb:.1f} KB")
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.scrapers.base_scraper import BaseISPScraper, ScrapedPage
from src.utils.robots_checker import RobotsChecker


class NetlifeScraper(BaseISPScraper):
    """Scraper for Netlife (MEGADATOS S.A.) internet plans.

    Handles the React-based Netlife website which renders plan
    cards dynamically. Captures both HTML content and screenshots
    for hybrid text+vision LLM processing.

    Args:
        robots_checker: Pre-initialized compliance checker.
        delay_range: (min_sec, max_sec) between requests.
        data_raw_path: Directory for raw content storage.
    """

    # CSS selectors to wait for before capturing content
    _PLAN_SELECTOR: str = ".plan-card, [class*='plan'], [class*='Plan']"

    def __init__(
        self,
        robots_checker: RobotsChecker,
        delay_range: tuple[float, float] = (2.5, 5.0),
        data_raw_path: Path = Path("data/raw"),
    ) -> None:
        super().__init__(
            isp_key="netlife",
            base_url="https://netlife.ec",
            robots_checker=robots_checker,
            delay_range=delay_range,
            data_raw_path=data_raw_path,
        )

    def get_plan_urls(self) -> list[str]:
        """Return Netlife URLs containing plan information.

        Returns:
            List of URLs to scrape for plan data.
        """
        return [
            "https://netlife.ec/",
            "https://netlife.ec/planes-hogar",
        ]

    def requires_playwright(self) -> bool:
        """Netlife requires Playwright — React SPA.

        Returns:
            True always for Netlife.
        """
        return True

    async def scrape(self) -> ScrapedPage:
        """Execute Netlife scraping with Playwright + screenshot strategy.

        Tries each plan URL, captures HTML and screenshots,
        combines all content into a single ScrapedPage DTO.

        Returns:
            ScrapedPage with combined content from all plan URLs.
        """
        logger.info(f"[{self.isp_key}] 🕷️  Starting scrape...")

        combined_html: list[str] = []
        combined_text: list[str] = []
        all_screenshots: list[bytes] = []
        method_used = "unknown"
        primary_title = ""
        error: str | None = None

        for url in self.get_plan_urls():
            try:
                html, title, shots, method = await self._scrape_with_fallback(
                    url=url,
                    wait_selector=self._PLAN_SELECTOR,
                )

                if html:
                    combined_html.append(f"<!-- URL: {url} -->\n{html}")
                    text = self._extract_text_from_html(html)
                    combined_text.append(f"[Fuente: {url}]\n{text}")
                    method_used = method

                if title and not primary_title:
                    primary_title = title

                all_screenshots.extend(shots)
                logger.info(
                    f"[{self.isp_key}] ✅ {url} → "
                    f"{len(html) / 1024:.1f} KB, "
                    f"{len(shots)} screenshots"
                )

            except Exception as exc:
                error_msg = f"Failed {url}: {exc}"
                logger.warning(f"[{self.isp_key}] ⚠️  {error_msg}")
                error = error_msg

        page = ScrapedPage(
            isp_key=self.isp_key,
            url=self.get_plan_urls()[0],
            html_content="\n\n".join(combined_html),
            text_content="\n\n".join(combined_text),
            screenshots=all_screenshots,
            additional_urls=self.get_plan_urls()[1:],
            scraping_method=method_used,
            page_title=primary_title,
            error=error,
        )

        # Persist raw for audit trail
        page.save_raw(self.data_raw_path)

        logger.info(
            f"[{self.isp_key}] 🏁 Done — "
            f"{page.content_size_kb:.1f} KB, "
            f"{len(page.screenshots)} screenshots, "
            f"method={page.scraping_method}"
        )
        return page
