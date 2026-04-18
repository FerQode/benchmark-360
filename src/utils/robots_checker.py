# src/utils/robots_checker.py
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
from datetime import datetime, timezone
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
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


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

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        """Normalize a URL to its canonical base form for cache keying.

        Strips path, query, fragment, and trailing slashes. This ensures
        'https://netlife.ec/planes' and 'https://netlife.ec' map to the
        same cache key: 'https://netlife.ec'.

        Args:
            url: Any URL from the ISP domain.

        Returns:
            Canonical base URL string.
        """
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def get_crawl_delay(self, base_url: str) -> float:
        """Get the effective crawl delay for a domain.

        Args:
            base_url: Base URL of the domain.

        Returns:
            Crawl delay in seconds. Falls back to 2.0 if not analyzed.
        """
        canonical = self._normalize_base_url(base_url)
        analysis = self._cache.get(canonical)
        return analysis.effective_delay if analysis else 2.0

    async def analyze(self, base_url: str) -> RobotsAnalysis:
        """Fetch and parse robots.txt for a single domain."""
        canonical = self._normalize_base_url(base_url)
        robots_url = f"{canonical}/robots.txt"
        
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
                domain=canonical,
                allowed=True,
                effective_delay=effective_delay,
                sitemaps=sitemaps
            )
            self._parsers[canonical] = parser
            
        except Exception as exc:
            logger.warning("Failed to fetch {}: {} → conservative defaults", robots_url, exc)
            analysis = RobotsAnalysis(
                domain=canonical,
                allowed=True,
                effective_delay=5.0,
                error=str(exc)
            )
        
        self._cache[canonical] = analysis
        return analysis

    async def analyze_all_isps(self, urls: dict[str, str]) -> dict[str, RobotsAnalysis]:
        """Analyze robots.txt for all ISPs concurrently.

        Args:
            urls: Dict mapping isp_key → base_url.

        Returns:
            Dict mapping base_url → RobotsAnalysis for ALL ISPs.
            Failed ISPs get a conservative RobotsAnalysis with error set.
        """
        tasks = [self.analyze(url) for url in urls.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final: dict[str, RobotsAnalysis] = {}
        for (isp_key, url), result in zip(urls.items(), results):
            canonical = self._normalize_base_url(url)
            if isinstance(result, Exception):
                logger.error(
                    "Critical error analyzing {} ({}): {} → injecting conservative defaults",
                    isp_key, url, result
                )
                fallback = RobotsAnalysis(
                    domain=canonical,
                    allowed=True,
                    effective_delay=5.0,
                    error=f"gather exception: {result}",
                )
                self._cache[canonical] = fallback
                final[canonical] = fallback
            else:
                final[canonical] = result  # type: ignore[assignment]
                
        self.generate_report()
        return final

    def can_fetch(self, url: str) -> bool:
        """Check if a specific URL is allowed to be fetched."""
        parsed = urlparse(url)
        canonical = self._normalize_base_url(url)
        
        for blocked in self.always_blocked_paths:
            if blocked in parsed.path:
                return False
                
        parser = self._parsers.get(canonical)
        if parser:
            return parser.can_fetch(PIPELINE_USER_AGENT, url)
        return True

    def generate_report(self, output_path: Path = Path("docs/robots_analysis.md")) -> None:
        """Generate a Markdown compliance report for all analyzed ISPs."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines: list[str] = [
            "# Robots.txt Compliance Analysis\n",
            f"**Generated at:** `{datetime.now(tz=timezone.utc).isoformat()}`\n",
            f"**User-Agent declared:** `{PIPELINE_USER_AGENT}`\n",
            "---\n",
        ]
        
        for url, analysis in self._cache.items():
            status = "✅ ALLOWED" if analysis.allowed else "🚫 BLOCKED"
            lines.append(f"## {analysis.domain} — {status}\n")
            lines.append(f"- **Base URL:** `{url}`")
            lines.append(f"- **Crawl-Delay:** `{analysis.effective_delay}s`")
            lines.append(f"- **Sitemaps found:** `{len(analysis.sitemaps)}`")
            lines.append(f"- **Error:** `{analysis.error or 'None'}`")
            lines.append(f"- **Analyzed at:** `{analysis.analyzed_at.isoformat()}`")
            lines.append("")  # blank line after each section
            
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Robots compliance report generated at {}", output_path)
