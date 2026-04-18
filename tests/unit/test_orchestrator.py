# tests/unit/test_orchestrator.py
"""
Unit tests for PipelineOrchestrator and ParquetWriter.

All tests use mocked scrapers and LLM processors — zero network calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.models.isp_plan import ISPPlan, AdditionalServiceDetail
from src.pipeline.orchestrator import (
    ISPPipelineResult,
    PipelineOrchestrator,
    PipelineReport,
)
from src.pipeline.parquet_writer import ParquetWriter
from src.processors.guardrails import GuardrailsEngine
from src.processors.llm_client_factory import LLMClient, LLMClientFactory
from src.scrapers.base_scraper import ScrapedPage
from src.utils.company_registry import CompanyInfo
from src.utils.robots_checker import RobotsChecker


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """LLMClient mock with openai provider."""
    client = MagicMock(spec=LLMClient)
    client.provider_name = "openai"
    client.text_model = "gpt-4o-mini"
    client.vision_model = "gpt-4o"
    return client


@pytest.fixture
def mock_llm_factory(mock_llm_client: MagicMock) -> MagicMock:
    """LLMClientFactory mock returning the mock client."""
    factory = MagicMock(spec=LLMClientFactory)
    factory.get_primary_client.return_value = mock_llm_client
    factory.get_fallback_client.return_value = None
    return factory


@pytest.fixture
def mock_robots_checker() -> MagicMock:
    """Pre-configured RobotsChecker mock."""
    checker = MagicMock(spec=RobotsChecker)
    checker.get_crawl_delay.return_value = 0.0
    checker.can_fetch.return_value = True
    checker.analyze_all_isps = AsyncMock(return_value={})
    return checker


@pytest.fixture
def sample_isp_plan() -> ISPPlan:
    """Minimal valid ISPPlan for testing."""
    return ISPPlan(
        fecha=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
        empresa="MEGADATOS S.A.",
        marca="Netlife",
        nombre_plan="Plan 300 Mbps",
        velocidad_download_mbps=300.0,
        velocidad_upload_mbps=300.0,
        precio_plan=25.00,
        tecnologia="fibra_optica",
    )


@pytest.fixture
def sample_scraped_page() -> ScrapedPage:
    """ScrapedPage with HTML content and one screenshot."""
    return ScrapedPage(
        isp_key="netlife",
        url="https://www.netlife.ec/planes-hogar/",
        html_content="<html><body><p>Plan 300 Mbps $25.00</p></body></html>",
        text_content="Plan 300 Mbps $25.00",
        screenshots=[b"\x89PNG_fake"],
        scraping_method="playwright",
    )


@pytest.fixture
def orchestrator(
    mock_robots_checker: MagicMock,
    mock_llm_factory: MagicMock,
    tmp_path: Path,
) -> PipelineOrchestrator:
    """PipelineOrchestrator with all dependencies mocked."""
    return PipelineOrchestrator(
        robots_checker=mock_robots_checker,
        llm_factory=mock_llm_factory,
        guardrails=GuardrailsEngine(),
        data_raw_path=tmp_path / "raw",
        data_output_path=tmp_path / "output",
        max_concurrent=2,
        dry_run=True,
    )


# ─────────────────────────────────────────────────────────────────
# ISPPipelineResult — properties
# ─────────────────────────────────────────────────────────────────


class TestISPPipelineResult:
    """Validates ISPPipelineResult computed properties."""

    @pytest.mark.unit
    def test_success_true_when_plans_present(
        self, sample_isp_plan: ISPPlan
    ) -> None:
        info = CompanyInfo(
            empresa="MEGADATOS S.A.", marca="Netlife",
            ruc="123", verified=True,
        )
        result = ISPPipelineResult(
            isp_key="netlife",
            company_info=info,
            plans=[sample_isp_plan],
        )
        assert result.success is True
        assert result.plan_count == 1

    @pytest.mark.unit
    def test_success_false_when_no_plans(self) -> None:
        info = CompanyInfo(
            empresa="MEGADATOS S.A.", marca="Netlife",
            ruc="123", verified=True,
        )
        result = ISPPipelineResult(isp_key="netlife", company_info=info)
        assert result.success is False
        assert result.plan_count == 0


# ─────────────────────────────────────────────────────────────────
# PipelineReport — properties
# ─────────────────────────────────────────────────────────────────


class TestPipelineReport:
    """Validates PipelineReport computed properties."""

    @pytest.mark.unit
    def test_duration_seconds_calculated(self) -> None:
        start = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2024, 6, 15, 10, 1, 30, tzinfo=timezone.utc)
        report = PipelineReport(
            run_id="test_run",
            started_at=start,
            finished_at=end,
        )
        assert abs(report.duration_seconds - 90.0) < 0.1

    @pytest.mark.unit
    def test_print_summary_no_exception(self, capsys) -> None:
        info = CompanyInfo(
            empresa="MEGADATOS S.A.", marca="Netlife",
            ruc="123", verified=True,
        )
        report = PipelineReport(
            run_id="20240615_100000",
            started_at=datetime(2024, 6, 15, tzinfo=timezone.utc),
            finished_at=datetime(2024, 6, 15, tzinfo=timezone.utc),
            isp_results={
                "netlife": ISPPipelineResult(
                    isp_key="netlife", company_info=info,
                    plans=[], duration_seconds=5.0,
                )
            },
            total_plans=0,
            isps_ok=0,
            isps_failed=1,
        )
        report.print_summary()
        captured = capsys.readouterr()
        assert "BENCHMARK 360" in captured.out
        assert "netlife" in captured.out


# ─────────────────────────────────────────────────────────────────
# PipelineOrchestrator — _safe_get_company_info
# ─────────────────────────────────────────────────────────────────


class TestSafeGetCompanyInfo:
    """Validates graceful fallback for unknown ISP keys."""

    @pytest.mark.unit
    def test_known_isp_returns_correct_info(self) -> None:
        info = PipelineOrchestrator._safe_get_company_info("netlife")
        assert info.empresa == "MEGADATOS S.A."
        assert info.marca == "Netlife"

    @pytest.mark.unit
    def test_unknown_isp_returns_placeholder(self) -> None:
        info = PipelineOrchestrator._safe_get_company_info("unknown_isp_xyz")
        assert info.empresa is not None
        assert info.marca is not None
        assert info.verified is False


# ─────────────────────────────────────────────────────────────────
# PipelineOrchestrator — run() with mocks
# ─────────────────────────────────────────────────────────────────


class TestOrchestratorRun:
    """Validates orchestrator run() with fully mocked dependencies."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_returns_pipeline_report(
        self,
        orchestrator: PipelineOrchestrator,
        sample_scraped_page: ScrapedPage,
        sample_isp_plan: ISPPlan,
    ) -> None:
        """run() must always return a PipelineReport."""
        with patch(
            "src.pipeline.orchestrator.build_all_scrapers"
        ) as mock_build:
            mock_scraper = MagicMock()
            mock_scraper.scrape = AsyncMock(return_value=sample_scraped_page)
            mock_build.return_value = {"netlife": mock_scraper}

            with patch.object(
                orchestrator._normalizer,
                "normalize_all",
                return_value=[sample_isp_plan],
            ):
                report = await orchestrator.run(isp_keys=["netlife"])

        assert isinstance(report, PipelineReport)
        assert report.run_id is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dry_run_skips_parquet(
        self,
        orchestrator: PipelineOrchestrator,
        sample_scraped_page: ScrapedPage,
        sample_isp_plan: ISPPlan,
        tmp_path: Path,
    ) -> None:
        """In dry_run mode, no Parquet file must be written."""
        assert orchestrator._dry_run is True

        with patch(
            "src.pipeline.orchestrator.build_all_scrapers"
        ) as mock_build:
            mock_scraper = MagicMock()
            mock_scraper.scrape = AsyncMock(return_value=sample_scraped_page)
            mock_build.return_value = {"netlife": mock_scraper}

            with patch.object(
                orchestrator._normalizer,
                "normalize_all",
                return_value=[sample_isp_plan],
            ):
                report = await orchestrator.run(isp_keys=["netlife"])

        assert report.output_path is None
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failed_scraper_does_not_abort_others(
        self,
        orchestrator: PipelineOrchestrator,
        sample_scraped_page: ScrapedPage,
        sample_isp_plan: ISPPlan,
    ) -> None:
        """One ISP failure must not prevent others from running."""
        failing_scraper = MagicMock()
        failing_scraper.scrape = AsyncMock(
            side_effect=RuntimeError("Playwright crashed")
        )
        good_scraper = MagicMock()
        good_scraper.scrape = AsyncMock(return_value=sample_scraped_page)

        with patch(
            "src.pipeline.orchestrator.build_all_scrapers",
            return_value={
                "netlife": failing_scraper,
                "cnt": good_scraper,
            },
        ):
            with patch.object(
                orchestrator._normalizer,
                "normalize_all",
                return_value=[sample_isp_plan],
            ):
                report = await orchestrator.run(
                    isp_keys=["netlife", "cnt"]
                )

        assert "netlife" in report.isp_results
        assert "cnt" in report.isp_results
        # CNT succeeded despite netlife failing
        assert report.isp_results["cnt"].success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(
        self,
        orchestrator: PipelineOrchestrator,
        sample_scraped_page: ScrapedPage,
        sample_isp_plan: ISPPlan,
    ) -> None:
        """Semaphore must be respected (max_concurrent=2)."""
        import asyncio as _asyncio

        concurrent_count = 0
        max_seen = 0

        async def slow_scrape():
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await _asyncio.sleep(0.05)
            concurrent_count -= 1
            return sample_scraped_page

        scrapers = {}
        for key in ["netlife", "cnt", "claro", "xtrim"]:
            m = MagicMock()
            m.scrape = AsyncMock(side_effect=slow_scrape)
            scrapers[key] = m

        with patch(
            "src.pipeline.orchestrator.build_all_scrapers",
            return_value=scrapers,
        ):
            with patch.object(
                orchestrator._normalizer,
                "normalize_all",
                return_value=[sample_isp_plan],
            ):
                await orchestrator.run(
                    isp_keys=["netlife", "cnt", "claro", "xtrim"]
                )

        assert max_seen <= 2


# ─────────────────────────────────────────────────────────────────
# ParquetWriter
# ─────────────────────────────────────────────────────────────────


class TestParquetWriter:
    """Validates Parquet serialization and schema enforcement."""

    @pytest.mark.unit
    def test_write_creates_parquet_file(
        self,
        tmp_path: Path,
        sample_isp_plan: ISPPlan,
    ) -> None:
        writer = ParquetWriter(output_dir=tmp_path)
        path = writer.write(plans=[sample_isp_plan], run_id="test_001")
        assert path.exists()
        assert path.suffix == ".parquet"
        assert path.name == "benchmark_industria.parquet"

    @pytest.mark.unit
    def test_write_creates_snapshot_file(
        self,
        tmp_path: Path,
        sample_isp_plan: ISPPlan,
    ) -> None:
        writer = ParquetWriter(output_dir=tmp_path)
        writer.write(plans=[sample_isp_plan], run_id="snap_001")
        snapshot = tmp_path / "benchmark_industria_snap_001.parquet"
        assert snapshot.exists()

    @pytest.mark.unit
    def test_written_file_readable_as_dataframe(
        self,
        tmp_path: Path,
        sample_isp_plan: ISPPlan,
    ) -> None:
        writer = ParquetWriter(output_dir=tmp_path)
        path = writer.write(plans=[sample_isp_plan], run_id="read_test")
        df = writer.read(path=path)
        assert len(df) == 1
        assert df.iloc[0]["marca"] == "Netlife"
        assert df.iloc[0]["nombre_plan"] == "Plan 300 Mbps"

    @pytest.mark.unit
    def test_all_required_columns_present(
        self,
        tmp_path: Path,
        sample_isp_plan: ISPPlan,
    ) -> None:
        writer = ParquetWriter(output_dir=tmp_path)
        path = writer.write(plans=[sample_isp_plan], run_id="cols_test")
        df = writer.read(path=path)
        required = [
            "fecha", "anio", "mes", "dia",
            "empresa", "marca", "nombre_plan",
            "velocidad_download_mbps", "velocidad_upload_mbps",
            "precio_plan", "pys_adicionales", "pys_adicionales_detalle",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    @pytest.mark.unit
    def test_pys_adicionales_detalle_is_valid_json_string(
        self,
        tmp_path: Path,
    ) -> None:
        plan = ISPPlan(
            fecha=datetime(2024, 6, 15, tzinfo=timezone.utc),
            empresa="MEGADATOS S.A.",
            marca="Netlife",
            nombre_plan="Plan con Disney+",
            velocidad_download_mbps=300.0,
            velocidad_upload_mbps=300.0,
            precio_plan=25.0,
            pys_adicionales_detalle={
                "disney_plus": AdditionalServiceDetail(
                    tipo_plan="disney_plus_premium",
                    meses=9,
                    categoria="streaming",
                )
            },
        )
        writer = ParquetWriter(output_dir=tmp_path)
        path = writer.write(plans=[plan], run_id="pys_test")
        df = writer.read(path=path)
        raw_pys = df.iloc[0]["pys_adicionales_detalle"]
        parsed = json.loads(raw_pys)
        assert "disney_plus" in parsed
        assert parsed["disney_plus"]["categoria"] == "streaming"

    @pytest.mark.unit
    def test_write_raises_on_empty_list(self, tmp_path: Path) -> None:
        writer = ParquetWriter(output_dir=tmp_path)
        with pytest.raises(ValueError, match="empty plans list"):
            writer.write(plans=[], run_id="empty_test")

    @pytest.mark.unit
    def test_read_raises_on_missing_file(self, tmp_path: Path) -> None:
        writer = ParquetWriter(output_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            writer.read()

    @pytest.mark.unit
    def test_multiple_plans_multiple_isps(self, tmp_path: Path) -> None:
        """Verify multi-ISP Parquet contains correct row counts."""
        plans = []
        configs = [
            ("MEGADATOS S.A.", "Netlife", "Plan 300 Mbps", 300.0, 25.0),
            ("MEGADATOS S.A.", "Netlife", "Plan 600 Mbps", 600.0, 35.0),
            ("CONECEL S.A.",   "Claro",   "Plan 100 Mbps", 100.0, 15.0),
        ]
        for empresa, marca, nombre, vel, precio in configs:
            plans.append(ISPPlan(
                fecha=datetime(2024, 6, 15, tzinfo=timezone.utc),
                empresa=empresa,
                marca=marca,
                nombre_plan=nombre,
                velocidad_download_mbps=vel,
                velocidad_upload_mbps=vel,
                precio_plan=precio,
            ))

        writer = ParquetWriter(output_dir=tmp_path)
        path = writer.write(plans=plans, run_id="multi_test")
        df = writer.read(path=path)

        assert len(df) == 3
        assert set(df["marca"].unique()) == {"Netlife", "Claro"}
        assert len(df[df["marca"] == "Netlife"]) == 2
