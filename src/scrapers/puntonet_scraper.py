"""
Puntonet ISP scraper — Playwright + Vision LLM strategy.

Puntonet uses a JavaScript-heavy frontend. Plan pricing cards render dynamically
and promotional banners are image-based, requiring Vision LLM.

Scraping Strategy:
    Primary:  Playwright (networkidle + .plan-card selector)
    Fallback: Full-page screenshot → Vision LLM

Target URLs:
    Main:      https://www.puntonet.ec/hogar/
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.scrapers.base_scraper import BaseISPScraper, ScrapedPage
from src.utils.robots_checker import RobotsChecker


class PuntonetScraper(BaseISPScraper):
    """Scraper for Puntonet internet plans.

    Handles Puntonet Ecuador's JS-based website. Plan cards
    are dynamically rendered — Playwright required. Captures
    both HTML and screenshots for hybrid processing.

    Args:
        robots_checker: Pre-initialized compliance checker.
        delay_range: (min_sec, max_sec) between requests.
        data_raw_path: Directory for raw content storage.
    """

    _PLAN_SELECTOR: str = ".plan-card, [class*='plan'], [class*='Plan']"

    def __init__(
        self,
        robots_checker: RobotsChecker,
        delay_range: tuple[float, float] = (2.5, 5.0),
        data_raw_path: Path = Path("data/raw"),
    ) -> None:
        super().__init__(
            isp_key="puntonet",
            base_url="https://www.puntonet.ec",
            robots_checker=robots_checker,
            delay_range=delay_range,
            data_raw_path=data_raw_path,
        )

    def get_plan_urls(self) -> list[str]:
        """Return Puntonet Ecuador URLs containing plan information.

        Returns:
            Ordered list: specific plan page first, homepage second
            as fallback in case the plan URL structure changes.
        """
        return [
            "https://www.puntonet.ec/hogar/",
            "https://www.puntonet.ec/",
        ]

    def requires_playwright(self) -> bool:
        """Puntonet requires Playwright — SPA.

        Returns:
            True always — Puntonet's frontend is fully JS-rendered.
        """
        return True

    async def scrape(self) -> ScrapedPage:
        """Execute Puntonet scraping with Playwright + screenshot strategy.

        Iterates all plan URLs, aggregates HTML and screenshots into
        a single ScrapedPage DTO for downstream LLM processing.

        Returns:
            ScrapedPage with combined content from all plan URLs.
            Falls back to screenshot-only if HTML extraction fails.
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
