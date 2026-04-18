# src/scrapers/cnt_scraper.py
"""
CNT ISP scraper — httpx + Playwright fallback strategy.

CNT uses a mostly SSR frontend. Plan pricing cards usually render
in static HTML, making it perfect for rapid httpx scraping.

Scraping Strategy:
    Primary:  httpx (fast static HTML)
    Fallback: Playwright if dynamically loaded

Target URLs:
    Main:      https://www.cnt.com.ec/hogar/internet
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.scrapers.base_scraper import BaseISPScraper, ScrapedPage
from src.utils.robots_checker import RobotsChecker


class CNTScraper(BaseISPScraper):
    """Scraper for CNT internet plans.

    Handles CNT Ecuador's SSR-based website. Plan cards
    are usually static — httpx preferred. Captures
    HTML for processing.

    Args:
        robots_checker: Pre-initialized compliance checker.
        delay_range: (min_sec, max_sec) between requests.
        data_raw_path: Directory for raw content storage.
    """

    _PLAN_SELECTOR: str = ".plan-card, .tarjeta-plan, [class*='plan']"

    def __init__(
        self,
        robots_checker: RobotsChecker,
        delay_range: tuple[float, float] = (2.5, 5.0),
        data_raw_path: Path = Path("data/raw"),
    ) -> None:
        super().__init__(
            isp_key="cnt",
            base_url="https://www.cnt.com.ec",
            robots_checker=robots_checker,
            delay_range=delay_range,
            data_raw_path=data_raw_path,
        )

    def get_plan_urls(self) -> list[str]:
        """Return CNT Ecuador URLs containing plan information.

        Returns:
            Ordered list: specific plan page first, homepage second
            as fallback in case the plan URL structure changes.
        """
        return [
            "https://www.cnt.com.ec/hogar/internet",
            "https://www.cnt.com.ec/hogar/",
        ]

    def requires_playwright(self) -> bool:
        """CNT can usually be scraped with httpx first.

        Returns:
            False — allows httpx strategy to run before playwright.
        """
        return False

    async def scrape(self) -> ScrapedPage:
        """Execute CNT scraping with httpx + Playwright fallback strategy.

        Iterates all plan URLs, aggregates HTML into
        a single ScrapedPage DTO for downstream LLM processing.

        Returns:
            ScrapedPage with combined content from all plan URLs.
        """
        logger.info("[{}] 🕷️  Starting scrape...", self.isp_key)

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
                    if method_used == "unknown" or method == "playwright":
                        # Upgrade method recorded if playwright was needed
                        method_used = method
                if title and not primary_title:
                    primary_title = title
                all_screenshots.extend(shots)

            except Exception as exc:
                error = f"Failed {url}: {exc}"
                logger.warning("[{}] ⚠️  {}", self.isp_key, error)

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
        page.save_raw(self.data_raw_path)
        logger.info("[{}] 🏁 Done — {:.1f} KB, {} shots, method={}",
                    self.isp_key, page.content_size_kb,
                    len(page.screenshots), page.scraping_method)
        return page
