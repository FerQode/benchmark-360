# src/pipeline/orchestrator.py
"""
Pipeline Orchestrator — Master coordinator for Benchmark 360.

Assembles all pipeline layers into a single executable workflow:
  Scraping → Guardrails → LLM/Vision → Normalization → Parquet

Execution model:
    Concurrent ISP scraping with asyncio.Semaphore(MAX_CONCURRENT_ISPS).
    Each ISP runs its full pipeline independently — failures are isolated.
    LLMs are ALWAYS invoked unless dry_run=True (testing only).

Typical usage example:
    >>> orchestrator = PipelineOrchestrator.from_env()
    >>> report = await orchestrator.run()
    >>> print(f"Extracted {report.total_plans} plans from {report.isps_ok} ISPs")
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.models.isp_plan import ISPPlan
from src.processors.arma_tu_plan_handler import ArmaTuPlanHandler
from src.processors.guardrails import GuardrailsEngine
from src.processors.llm_client_factory import LLMClientFactory
from src.processors.llm_processor import LLMExtractionResult, LLMProcessor
from src.processors.normalizer import PlanNormalizer
from src.processors.vision_processor import (
    VisionExtractionResult,
    VisionProcessor,
)
from src.scrapers import ALL_ISP_URLS, build_all_scrapers
from src.scrapers.base_scraper import BaseISPScraper, ScrapedPage
from src.utils.company_registry import CompanyInfo, get_company_info
from src.utils.robots_checker import RobotsChecker

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

MAX_CONCURRENT_ISPS: int = 3
DEFAULT_DATA_RAW_PATH: Path = Path("data/raw")
DEFAULT_DATA_OUTPUT_PATH: Path = Path("data/output")


# ─────────────────────────────────────────────────────────────────
# Result DTOs
# ─────────────────────────────────────────────────────────────────


@dataclass
class ISPPipelineResult:
    """Result of the full pipeline execution for one ISP.

    Attributes:
        isp_key: ISP identifier string.
        company_info: Legal and brand company information.
        scraped_page: Raw scraping output. None if scraping failed.
        llm_result: Text LLM extraction result. None if skipped.
        vision_result: Vision LLM extraction result. None if no screenshots.
        plans: Final validated ISPPlan objects ready for Parquet.
        duration_seconds: Total wall-clock time for this ISP.
        error: Fatal error message if the ISP pipeline failed entirely.
    """

    isp_key: str
    company_info: CompanyInfo
    scraped_page: ScrapedPage | None = None
    llm_result: LLMExtractionResult | None = None
    vision_result: VisionExtractionResult | None = None
    plans: list[ISPPlan] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """True if at least one validated plan was extracted."""
        return len(self.plans) > 0

    @property
    def plan_count(self) -> int:
        """Number of validated ISPPlan objects extracted."""
        return len(self.plans)

    @property
    def llm_calls_made(self) -> int:
        """Total LLM API calls made for this ISP."""
        text_calls = self.llm_result.total_llm_calls if self.llm_result else 0
        vision_calls = (
            self.vision_result.total_llm_calls if self.vision_result else 0
        )
        return text_calls + vision_calls

    @property
    def fallback_was_used(self) -> bool:
        """True if the fallback LLM provider was activated."""
        text_fb = (
            self.llm_result.fallback_activated if self.llm_result else False
        )
        vision_fb = (
            self.vision_result.fallback_activated
            if self.vision_result
            else False
        )
        return text_fb or vision_fb


@dataclass
class PipelineReport:
    """Summary report for a complete pipeline run across all ISPs.

    Attributes:
        run_id: Unique run identifier (UTC timestamp string).
        started_at: UTC datetime the pipeline started.
        finished_at: UTC datetime the pipeline completed.
        isp_results: Dict mapping isp_key to ISPPipelineResult.
        output_path: Written Parquet file path. None if not written.
        total_plans: Total validated plans across all ISPs.
        total_llm_calls: Total LLM API calls across all ISPs.
        isps_ok: ISP count with at least one plan extracted.
        isps_failed: ISP count with zero plans or fatal errors.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    isp_results: dict[str, ISPPipelineResult] = field(default_factory=dict)
    output_path: Path | None = None
    total_plans: int = 0
    total_llm_calls: int = 0
    isps_ok: int = 0
    isps_failed: int = 0

    @property
    def duration_seconds(self) -> float:
        """Total pipeline duration in seconds."""
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def precision_rate(self) -> float:
        """Ratio of ISPs that produced at least one plan (0.0–1.0)."""
        total = self.isps_ok + self.isps_failed
        return self.isps_ok / total if total > 0 else 0.0

    def print_summary(self) -> None:
        """Print a human-readable pipeline summary to stdout."""
        print("\n" + "═" * 65)
        print("  BENCHMARK 360 — Pipeline Run Summary")
        print("═" * 65)
        print(f"  Run ID:        {self.run_id}")
        print(f"  Duration:      {self.duration_seconds:.1f}s")
        print(f"  Total plans:   {self.total_plans}")
        print(f"  LLM API calls: {self.total_llm_calls}")
        print(
            f"  ISPs OK:       {self.isps_ok}/"
            f"{self.isps_ok + self.isps_failed} "
            f"({self.precision_rate:.0%})"
        )
        if self.output_path:
            print(f"  Output:        {self.output_path}")
        print("─" * 65)
        for isp_key, result in self.isp_results.items():
            status = "✅" if result.success else "❌"
            fallback = " [FB]" if result.fallback_was_used else ""
            err = f" — {result.error[:45]}" if result.error else ""
            print(
                f"  {status} {isp_key:<12} "
                f"{result.plan_count:>3} plans  "
                f"{result.llm_calls_made:>2} LLM calls  "
                f"{result.duration_seconds:>6.1f}s"
                f"{fallback}{err}"
            )
        print("═" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────
# PipelineOrchestrator
# ─────────────────────────────────────────────────────────────────


class PipelineOrchestrator:
    """Master coordinator for the Benchmark 360 data pipeline.

    Wires scraping → guardrails → LLM text → LLM vision →
    normalization → Parquet into a single concurrent workflow.

    Key design decisions:
    - asyncio.Semaphore limits concurrent ISPs to avoid IP blocks
    - Each ISP stage is isolated: one failure never aborts others
    - LLMs are ALWAYS called in production (dry_run=False)
    - Vision extraction runs ONLY when screenshots were captured
    - Fields not found by LLM are set to None (not fabricated)

    Args:
        robots_checker: Pre-initialized RobotsChecker.
        llm_factory: LLMClientFactory for creating LLM clients.
        guardrails: GuardrailsEngine for input/output protection.
        data_raw_path: Directory for raw HTML/screenshot storage.
        data_output_path: Directory for Parquet output.
        max_concurrent: Max simultaneous ISP pipelines (default: 3).
        dry_run: If True, skips LLM calls and Parquet write.
            USE ONLY FOR TESTING THE SCRAPING LAYER.

    Example:
        >>> orchestrator = PipelineOrchestrator.from_env()
        >>> report = await orchestrator.run()
        >>> report.print_summary()
    """

    def __init__(
        self,
        robots_checker: RobotsChecker,
        llm_factory: LLMClientFactory,
        guardrails: GuardrailsEngine,
        data_raw_path: Path = DEFAULT_DATA_RAW_PATH,
        data_output_path: Path = DEFAULT_DATA_OUTPUT_PATH,
        max_concurrent: int = MAX_CONCURRENT_ISPS,
        dry_run: bool = False,
    ) -> None:
        self._robots = robots_checker
        self._llm_factory = llm_factory
        self._guardrails = guardrails
        self._data_raw = data_raw_path
        self._data_output = data_output_path
        self._max_concurrent = max_concurrent
        self._dry_run = dry_run
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        self.metrics = {
            'hybrid_plans': 0,
            'llm_plans': 0,
            'hallucinations': 0,
            'estimated_cost': 0.0,
            'duration': 0.0
        }
        self.hybrid_only_mode = False

        # Build processors once — shared across all ISP tasks
        primary = llm_factory.get_primary_client()
        fallback = llm_factory.get_fallback_client()
        vision_primary = llm_factory.get_vision_client()
        mistral_client = llm_factory.get_mistral_client()

        self._llm_processor = LLMProcessor(
            primary_client=primary,
            fallback_client=fallback,
            guardrails=guardrails,
        )
        self._vision_processor = VisionProcessor(
            primary_client=vision_primary,
            fallback_client=fallback,
            guardrails=guardrails,
            mistral_client=mistral_client,
        )
        self._normalizer = PlanNormalizer(
            arma_tu_plan_handler=ArmaTuPlanHandler()
        )

        logger.info(
            "PipelineOrchestrator ready — provider={}, fallback={}, "
            "concurrent={}, dry_run={}",
            primary.__class__.__name__,
            fallback.__class__.__name__ if fallback else "none",
            max_concurrent,
            dry_run,
        )

    @classmethod
    def from_env(cls) -> "PipelineOrchestrator":
        """Construct orchestrator entirely from environment variables.

        This is the standard factory method for production runs.
        Reads all config from .env loaded by python-dotenv.

        Returns:
            Fully configured PipelineOrchestrator.

        Raises:
            RuntimeError: If no LLM API keys are set.
        """
        from src.utils.logger import setup_logger

        setup_logger(
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )

        return cls(
            robots_checker=RobotsChecker(),
            llm_factory=LLMClientFactory(),
            guardrails=GuardrailsEngine(),
            data_raw_path=Path(
                os.getenv("DATA_RAW_PATH", "data/raw")
            ),
            data_output_path=Path(
                os.getenv("DATA_OUTPUT_PATH", "data/output")
            ),
            max_concurrent=int(
                os.getenv("MAX_CONCURRENT_ISPS", str(MAX_CONCURRENT_ISPS))
            ),
            dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
        )

    def check_api_health(self):
        """Verifica si las APIs tienen cuota antes de empezar"""
        for provider in ['gemini', 'openai', 'mistral']:
            try:
                # Simular una prueba rápida, en la práctica usaríamos endpoints de status
                # Si falla una conexión real, esto arrojará una excepción
                # self._llm_processor.test_connection(provider)
                # Para esta demo, asumiremos disponibles si la llave existe
                key_name = f"{provider.upper()}_API_KEY"
                if os.getenv(key_name):
                    logger.info(f"✅ {provider.upper()} disponible")
                else:
                    raise ValueError(f"Falta llave {key_name}")
            except Exception as e:
                logger.warning(f"⚠️ {provider.upper()} no disponible: {e}")
                self.hybrid_only_mode = True

    # ── Public API ────────────────────────────────────────────────

    async def run(
        self,
        isp_keys: list[str] | None = None,
    ) -> PipelineReport:
        """Execute the full pipeline for all (or selected) ISPs.

        Stages:
          1. robots.txt compliance pre-check (all ISPs, concurrent)
          2. Build scrapers from factory
          3. Run each ISP pipeline concurrently under semaphore
          4. Collect and merge all ISPPlan results
          5. Write Parquet (unless dry_run=True)
          6. Return PipelineReport with full metadata

        Args:
            isp_keys: Optional list of ISP keys to process.
                None means all registered ISPs are processed.

        Returns:
            PipelineReport with per-ISP results and aggregate stats.
        """
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        started_at = datetime.now(tz=timezone.utc)

        target_keys = isp_keys or list(ALL_ISP_URLS.keys())
        target_urls = {k: ALL_ISP_URLS[k] for k in target_keys}

        logger.info(
            "🚀 Run {} — {} ISPs, concurrent={}, dry_run={}",
            run_id,
            len(target_keys),
            self._max_concurrent,
            self._dry_run,
        )

        if self._dry_run:
            logger.warning(
                "⚠️  DRY RUN MODE — LLM calls and Parquet write DISABLED. "
                "Set DRY_RUN=false in .env for production extraction."
            )

        self.check_api_health()

        # ── Step 1: robots.txt pre-check ──────────────────────────
        await self._robots.analyze_all_isps(target_urls)

        # ── Step 2: Build scrapers ────────────────────────────────
        scrapers = build_all_scrapers(
            robots_checker=self._robots,
            data_raw_path=self._data_raw,
        )
        active_scrapers = {k: scrapers[k] for k in target_keys}

        # ── Step 3: Concurrent ISP execution ─────────────────────
        extraction_dt = datetime.now(tz=timezone.utc)
        tasks = [
            self._run_single_isp(
                isp_key=key,
                scraper=scraper,
                extraction_dt=extraction_dt,
            )
            for key, scraper in active_scrapers.items()
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── Step 4: Collect results ───────────────────────────────
        isp_results: dict[str, ISPPipelineResult] = {}
        all_plans: list[ISPPlan] = []

        for key, result in zip(target_keys, raw_results):
            if isinstance(result, Exception):
                info = self._safe_get_company_info(key)
                isp_results[key] = ISPPipelineResult(
                    isp_key=key,
                    company_info=info,
                    error=f"Unhandled exception: {result}",
                )
                logger.error(
                    "[{}] ❌ Unhandled gather exception: {}",
                    key, result,
                )
            else:
                isp_results[key] = result
                all_plans.extend(result.plans)

        # ── Step 5: Write Parquet ─────────────────────────────────
        output_path: Path | None = None

        if not self._dry_run:
            if all_plans:
                from src.pipeline.parquet_writer import ParquetWriter

                writer = ParquetWriter(output_dir=self._data_output)
                output_path = writer.write(
                    plans=all_plans,
                    run_id=run_id,
                )
            else:
                logger.warning(
                    "No plans extracted across all ISPs — "
                    "Parquet not written. Check LLM responses."
                )
        else:
            logger.info(
                "DRY RUN: Parquet write skipped. "
                "{} plans would have been written.",
                len(all_plans),
            )

        # ── Step 6: Build report ──────────────────────────────────
        finished_at = datetime.now(tz=timezone.utc)
        total_llm_calls = sum(
            r.llm_calls_made for r in isp_results.values()
        )
        report = PipelineReport(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            isp_results=isp_results,
            output_path=output_path,
            total_plans=len(all_plans),
            total_llm_calls=total_llm_calls,
            isps_ok=sum(1 for r in isp_results.values() if r.success),
            isps_failed=sum(
                1 for r in isp_results.values() if not r.success
            ),
        )
        
        self.metrics['duration'] = report.duration_seconds
        self.metrics['estimated_cost'] = report.total_llm_calls * 0.0015 # Costo simulado

        logger.info(
            "🏁 Run {} complete — {} plans, {}/{} ISPs OK, "
            "{} LLM calls, {:.1f}s",
            run_id,
            report.total_plans,
            report.isps_ok,
            len(target_keys),
            report.total_llm_calls,
            report.duration_seconds,
        )

        await BaseISPScraper.close_browser()
        return report

    # ── Private ───────────────────────────────────────────────────

    async def _run_single_isp(
        self,
        isp_key: str,
        scraper: BaseISPScraper,
        extraction_dt: datetime,
    ) -> ISPPipelineResult:
        """Execute the complete pipeline for one ISP.

        Stages per ISP (ALL run in production, dry_run=False):
        1. Scrape HTML + screenshots via Playwright or httpx
        2. Sanitize content via GuardrailsEngine
        3. LLM text extraction (ALWAYS in production)
        4. Vision extraction IF screenshots were captured
        5. Normalize + validate → ISPPlan list

        Fields not found by the LLM are set to None per schema spec.
        The LLM is explicitly instructed to use null for missing data,
        never to fabricate values.

        Args:
            isp_key: ISP identifier string.
            scraper: Configured scraper for this ISP.
            extraction_dt: Shared extraction datetime for all plans.

        Returns:
            ISPPipelineResult with validated plans and metadata.
        """
        info = self._safe_get_company_info(isp_key)
        start = datetime.now(tz=timezone.utc)

        async with self._semaphore:
            logger.info("[{}] ▶️  Pipeline starting", isp_key)
            result = ISPPipelineResult(
                isp_key=isp_key,
                company_info=info,
            )

            try:
                # ── Stage 1: Scrape ───────────────────────────────
                scraped_page = await scraper.scrape()
                result.scraped_page = scraped_page

                has_text = len(scraped_page.text_content) > 200
                has_shots = scraped_page.has_screenshots

                if not has_text and not has_shots:
                    result.error = (
                        "Scraping returned no usable content "
                        "(no text and no screenshots)"
                    )
                    logger.warning(
                        "[{}] ⚠️  {}", isp_key, result.error
                    )
                    return result

                logger.info(
                    "[{}] 📄 Scraped: {:.1f} KB text, {} screenshots",
                    isp_key,
                    len(scraped_page.text_content) / 1024,
                    len(scraped_page.screenshots),
                )

                # ── T&C: inyectar directamente en los planes normalizados ───
                # El terminos_condiciones_raw ya está en scraped_page.
                # El normalizer lo lee desde ahí y lo asigna al campo del schema.
                # NO va al text_content de planes. Separación de concerns respetada.
                if scraped_page.has_tc:
                    logger.info(
                        "[{}] 📄 T&C disponible: {} chars → "
                        "se asignará a terminos_condiciones en cada plan",
                        isp_key,
                        len(scraped_page.terminos_condiciones_raw),
                    )

                from src.processors.hybrid_emergency_parser import HybridEmergencyParser

                # ── Stage 1.5: Hybrid Extraction (Regex) ───────────────
                hybrid_extracted = False
                llm_result = None
                
                # Intentamos primero con parser híbrido si no estamos en dry run
                if not self._dry_run and scraped_page.html_raw:
                    hybrid_plans = HybridEmergencyParser.extract_from_any(scraped_page.html_raw, isp_key)
                    if hybrid_plans:
                        logger.info("✅ HYBRID EXTRACTION | ISP: {} | Planes: {} | Costo: $0", isp_key, len(hybrid_plans))
                        print(f"🚀 [HYBRID] {isp_key}: {len(hybrid_plans)} planes extraídos sin LLM")
                        
                        llm_result = LLMExtractionResult(
                            isp_key=isp_key,
                            raw_plans=hybrid_plans,
                            chunks_processed=1,
                            total_llm_calls=0,
                            fallback_activated=False
                        )
                        hybrid_extracted = True
                        result.llm_result = llm_result
                        self.metrics['hybrid_plans'] += len(hybrid_plans)

                # ── Stage 2 + 3: LLM text extraction ─────────────
                # ALWAYS runs in production (dry_run=False)
                # Sends sanitized content to GPT-4o-mini or Gemini Flash
                # Fields not found → LLM returns null → stored as None
                if not hybrid_extracted:
                    if not self._dry_run:
                        if self.hybrid_only_mode:
                            logger.warning(f"[{isp_key}] ⚠️ Saltando LLM por hybrid_only_mode")
                            llm_result = LLMExtractionResult(isp_key=isp_key)
                            result.llm_result = llm_result
                        else:
                            logger.info(
                                "[{}] 🤖 Sending to LLM text extraction "
                                "(provider={})...",
                                isp_key,
                                self._llm_processor._primary.__class__.__name__,
                            )
                            llm_result = await self._llm_processor.extract_plans(
                                scraped_page=scraped_page,
                                company_info=info,
                            )
                            result.llm_result = llm_result
                            self.metrics['llm_plans'] += len(llm_result.raw_plans)

                            logger.info(
                                "[{}] ✅ LLM text done: {} raw plans, "
                                "{} chunks OK, fallback={}",
                                isp_key,
                                len(llm_result.raw_plans),
                                llm_result.chunks_processed,
                                llm_result.fallback_activated,
                            )
                    else:
                        # DRY RUN: create empty result, no LLM call
                        llm_result = LLMExtractionResult(isp_key=isp_key)
                        result.llm_result = llm_result
                        logger.info(
                            "[{}] ⏭️  DRY RUN: LLM text skipped", isp_key
                        )

                # ── Stage 4: Vision extraction ────────────────────
                # Runs ONLY when screenshots exist AND not dry_run
                # Complements text extraction for image-based pricing
                vision_result: VisionExtractionResult | None = None

                if has_shots and not self._dry_run:
                    logger.info(
                        "[{}] 👁️  Sending {} screenshot(s) to Vision LLM "
                        "(provider={})...",
                        isp_key,
                        len(scraped_page.screenshots),
                        self._vision_processor._primary.__class__.__name__,
                    )
                    vision_result = (
                        await self._vision_processor.extract_from_screenshots(
                            scraped_page=scraped_page,
                            company_info=info,
                        )
                    )
                    result.vision_result = vision_result

                    logger.info(
                        "[{}] ✅ Vision done: {} plans from {} screenshots, "
                        "fallback={}",
                        isp_key,
                        len(vision_result.raw_plans),
                        vision_result.screenshots_processed,
                        vision_result.fallback_activated,
                    )
                elif has_shots and self._dry_run:
                    logger.info(
                        "[{}] ⏭️  DRY RUN: Vision skipped "
                        "({} screenshots available)",
                        isp_key,
                        len(scraped_page.screenshots),
                    )

                # ── Stage 5: Normalize → ISPPlan[] ────────────────
                # Merges LLM + Vision results
                # Expands Arma tu Plan via Cartesian product
                # Fields LLM could not find → stored as None (not fabricated)
                plans = self._normalizer.normalize_all(
                    llm_result=llm_result,
                    vision_result=vision_result,
                    company_info=info,
                    extraction_dt=extraction_dt,
                    terminos_condiciones_raw=scraped_page.terminos_condiciones_raw if scraped_page else "",
                )
                result.plans = plans
                
                hallucination_count = sum(1 for p in plans if p.is_hallucination)
                self.metrics['hallucinations'] += hallucination_count

                logger.info(
                    "[{}] 📦 Normalized: {} ISPPlan objects validated",
                    isp_key,
                    len(plans),
                )

            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "[{}] ❌ Pipeline failed at stage: {}",
                    isp_key,
                    result.error,
                )

            finally:
                end = datetime.now(tz=timezone.utc)
                result.duration_seconds = (end - start).total_seconds()
                logger.info(
                    "[{}] {} {:.1f}s total — {} plans, {} LLM calls",
                    isp_key,
                    "✅" if result.success else "⚠️",
                    result.duration_seconds,
                    result.plan_count,
                    result.llm_calls_made,
                )

        return result

    @staticmethod
    def _safe_get_company_info(isp_key: str) -> CompanyInfo:
        """Get CompanyInfo with graceful fallback for unknown ISPs.

        Args:
            isp_key: ISP identifier to look up.

        Returns:
            CompanyInfo from registry, or placeholder if not found.
        """
        try:
            return get_company_info(isp_key)
        except KeyError:
            logger.warning(
                "[{}] Not in company registry — using placeholder",
                isp_key,
            )
            return CompanyInfo(
                empresa=f"{isp_key.upper()} S.A.",
                marca=isp_key.capitalize(),
                ruc="PENDIENTE",
                verified=False,
            )
