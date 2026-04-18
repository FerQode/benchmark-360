"""
Robots.txt compliance checker for Benchmark 360.

Provides RFC 9309 compliant parsing of robots.txt for all
target ISPs. Respects crawl-delay and prevents scraping of
disallowed paths. Generates a Markdown compliance report.
"""

from __future__ import annotations

import asyncio
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

PIPELINE_USER_AGENT: str = "Benchmark360Bot/1.0 (+https://github.com/FerQode/benchmark-360)"


@dataclass
class RobotsAnalysis:
    """Analysis results for a single domain's robots.txt."""
    domain: str
    allowed: bool = True
    effective_delay: float = 2.0
    sitemaps: list[str] = field(default_factory=list)
    error: str | None = None
    analyzed_at: datetime = field(default_factory=datetime.now)


class RobotsChecker:
    """Pre-loads and evaluates robots.txt for all target ISPs.
    
    Implements a fail-safe approach: if robots.txt cannot be fetched,
    it assumes scraping is allowed but logs a warning.
    """

    _ALWAYS_BLOCKED_PATHS: list[str] = [
        "/admin/", "/login/", "/api/private/", "/cart/", "/checkout/"
    ]

    def __init__(self) -> None:
        self._cache: dict[str, RobotsAnalysis] = {}
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    @property
    def always_blocked_paths(self) -> list[str]:
        """Get the list of paths that are always blocked internally."""
        return self._ALWAYS_BLOCKED_PATHS

    async def analyze(self, base_url: str) -> RobotsAnalysis:
        """Fetch and parse robots.txt for a single domain."""
        parsed = urlparse(base_url)
        domain = parsed.netloc
        robots_url = f"{parsed.scheme}://{domain}/robots.txt"
        
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        
        try:
            # Running synchronous read in an executor to not block event loop
            await asyncio.to_thread(parser.read)
            
            delay = parser.crawl_delay(PIPELINE_USER_AGENT)
            if delay is None:
                delay = parser.crawl_delay("*")
            effective_delay = max(float(delay), 2.0) if delay else 2.0
            
            sitemaps = parser.site_maps() or []
            
            analysis = RobotsAnalysis(
                domain=domain,
                allowed=True,
                effective_delay=effective_delay,
                sitemaps=sitemaps
            )
            self._parsers[base_url] = parser
            
        except Exception as exc:
            logger.warning(f"Failed to fetch {robots_url}: {exc}. Using conservative defaults.")
            analysis = RobotsAnalysis(
                domain=domain,
                allowed=True,
                effective_delay=5.0,
                error=str(exc)
            )
        
        self._cache[base_url] = analysis
        return analysis

    async def analyze_all_isps(self, urls: dict[str, str]) -> dict[str, RobotsAnalysis]:
        """Analyze robots.txt for all ISPs concurrently."""
        tasks = [self.analyze(url) for url in urls.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for url, res in zip(urls.values(), results):
            if isinstance(res, Exception):
                logger.error(f"Critical error analyzing {url}: {res}")
                
        self.generate_report()
        return self._cache

    def can_fetch(self, url: str) -> bool:
        """Check if a specific URL is allowed to be fetched."""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        for blocked in self.always_blocked_paths:
            if blocked in parsed.path:
                return False
                
        parser = self._parsers.get(base_url)
        if parser:
            return parser.can_fetch(PIPELINE_USER_AGENT, url)
        return True

    def generate_report(self, output_path: Path = Path("docs/robots_analysis.md")) -> None:
        """Generate a Markdown compliance report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = ["# Robots.txt Compliance Analysis\n", f"Generated at: {datetime.now().isoformat()}\n\n"]
        for url, analysis in self._cache.items():
            report.append(f"## {analysis.domain}\n")
            report.append(f"- **Allowed**: {analysis.allowed}")
            report.append(f"- **Crawl-Delay**: {analysis.effective_delay}s")
            report.append(f"- **Error**: {analysis.error or 'None'}")
            report.append(f"- **Sitemaps**: {len(analysis.sitemaps)}\n")
            
        output_path.write_text("\n".join(report), encoding="utf-8")
        logger.info(f"Robots compliance report generated at {output_path}")
