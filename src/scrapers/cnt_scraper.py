# src/scrapers/cnt_scraper.py
"""
CNT ISP scraper — httpx + BeautifulSoup strategy.

CNT (Corporación Nacional de Telecomunicaciones) uses a
server-side rendered website with static HTML plan tables,
making it the simplest scraper in the pipeline.

No JavaScript rendering required — direct httpx requests
are sufficient to capture all plan data.

Scraping Strategy:
    Primary:  httpx (static HTML)
    Fallback: Playwright if httpx returns insufficient content

Target URLs:
    Main:    https://www.cnt.com.ec/
    Internet: https://www.cnt.com.ec/servicios/internet/

Typical usage example:
    >>> scraper = CNTScraper(robots_checker=checker)
    >>> page = await scraper.scrape()
    >>> page.scraping_method
    'httpx'
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.scrapers.base_scraper import BaseISPScraper, ScrapedPage
from src.utils.robots_checker import RobotsChecker

# Minimum content threshold — if httpx returns less, we fallback
_MIN_CONTENT_CHARS: int = 5_000


class CNTScraper(BaseISPScraper):
    """Scraper for CNT internet plans using httpx + BeautifulSoup.

    CNT's website is server-side rendered, making it the fastest
    and simplest ISP to scrape in the Benchmark 360 pipeline.

    Args:
        robots_checker: Pre-initialized compliance checker.
        delay_range: (min_sec, max_sec) between requests.
        data_raw_path: Directory for raw content storage.
    """

    def __init__(
        self,
        robots_checker: RobotsChecker,
        delay_range: tuple[float, float] = (2.0, 4.0),
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
        """Return CNT URLs containing internet plan information.

        Returns:
            List of CNT plan page URLs.
        """
        return [
            "https://www.cnt.com.ec/servicios/internet/",
            "https://www.cnt.com.ec/",
        ]

    def requires_playwright(self) -> bool:
        """CNT does NOT require Playwright — static HTML.

        Returns:
            False — httpx is sufficient for CNT.
        """
        return False

    async def scrape(self) -> ScrapedPage:
        """Execute CNT scraping using httpx with Playwright fallback.

        Returns:
            ScrapedPage with plan content extracted via httpx.
        """
        logger.info(f"[{self.isp_key}] 🕷️  Starting scrape (httpx mode)...")

        combined_html: list[str] = []
        combined_text: list[str] = []
        all_screenshots: list[bytes] = []
        method_used = "httpx"
        primary_title = ""
        error: str | None = None

        for url in self.get_plan_urls():
            try:
                html, title, shots, method = await self._scrape_with_fallback(
                    url=url,
                )

                # Content quality check — if too little, note it
                if len(html) < _MIN_CONTENT_CHARS and html:
                    logger.warning(
                        f"[{self.isp_key}] ⚠️  Low content from {url}: "
                        f"{len(html)} chars — may need Playwright"
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
                error_msg = f"Failed {url}: {exc}"
                logger.error(f"[{self.isp_key}] ❌ {error_msg}")
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

        page.save_raw(self.data_raw_path)
        logger.info(
            f"[{self.isp_key}] 🏁 Done — "
            f"{page.content_size_kb:.1f} KB, method={page.scraping_method}"
        )
        return page
