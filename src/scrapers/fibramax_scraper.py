# src/scrapers/fibramax_scraper.py
"""Fibramax ISP scraper — Playwright + Vision LLM strategy."""

from __future__ import annotations
from pathlib import Path
from loguru import logger
from src.scrapers.base_scraper import BaseISPScraper, ScrapedPage
from src.utils.robots_checker import RobotsChecker


class FibramaxScraper(BaseISPScraper):
    """Scraper for Fibramax internet plans."""
    
    _PLAN_SELECTOR: str = ".plan-card, [class*='plan']"

    def __init__(
        self,
        robots_checker: RobotsChecker,
        delay_range: tuple[float, float] = (2.5, 5.0),
        data_raw_path: Path = Path("data/raw"),
    ) -> None:
        super().__init__(
            isp_key="fibramax",
            base_url="https://fibramax.ec",
            robots_checker=robots_checker,
            delay_range=delay_range,
            data_raw_path=data_raw_path,
        )

    def get_plan_urls(self) -> list[str]:
        return [
            "https://fibramax.ec/planes/",
            "https://fibramax.ec/",
        ]

    def requires_playwright(self) -> bool:
        return True

    async def scrape(self) -> ScrapedPage:
        logger.info(f"[{self.isp_key}] 🕷️  Starting scrape...")
        combined_html, combined_text, all_screenshots = [], [], []
        method_used, primary_title, error = "unknown", "", None

        for url in self.get_plan_urls():
            try:
                html, title, shots, method = await self._scrape_with_fallback(
                    url=url, wait_selector=self._PLAN_SELECTOR,
                )
                if html:
                    combined_html.append(f"<!-- URL: {url} -->\n{html}")
                    combined_text.append(f"[Fuente: {url}]\n{self._extract_text_from_html(html)}")
                    method_used = method
                if title and not primary_title: primary_title = title
                all_screenshots.extend(shots)
            except Exception as exc:
                error = f"Failed {url}: {exc}"
                logger.warning(f"[{self.isp_key}] ⚠️  {error}")

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
        return page
