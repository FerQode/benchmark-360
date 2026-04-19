# tests/test_errors.py
"""Tests de la jerarquía de excepciones personalizadas."""

from __future__ import annotations

import pytest

from src.utils.errors import (
    BenchmarkError,
    GuardrailsViolation,
    LLMProcessingError,
    LLMQuotaExhaustedError,
    RobotsDisallowedError,
    ScrapingError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Tests que verifican la jerarquía de herencia."""

    def test_scraping_error_is_benchmark_error(self):
        with pytest.raises(BenchmarkError):
            raise ScrapingError(isp="netlife", url="https://netlife.ec", reason="timeout")

    def test_llm_quota_is_benchmark_error(self):
        with pytest.raises(BenchmarkError):
            raise LLMQuotaExhaustedError()

    def test_llm_quota_is_llm_processing_error(self):
        with pytest.raises(LLMProcessingError):
            raise LLMQuotaExhaustedError()

    def test_guardrails_violation_is_benchmark_error(self):
        with pytest.raises(BenchmarkError):
            raise GuardrailsViolation(isp="claro", pattern="DROP TABLE")

    def test_robots_disallowed_is_benchmark_error(self):
        with pytest.raises(BenchmarkError):
            raise RobotsDisallowedError(url="https://example.com/admin")


class TestExceptionMessages:
    """Tests que verifican los mensajes de error."""

    def test_scraping_error_message_format(self):
        exc = ScrapingError(isp="netlife", url="https://netlife.ec", reason="timeout")
        assert "netlife" in str(exc)
        assert "timeout" in str(exc)
        assert exc.isp == "netlife"

    def test_llm_quota_default_message(self):
        exc = LLMQuotaExhaustedError()
        assert "exhausted" in str(exc).lower() or "waterfall" in str(exc).lower()

    def test_guardrails_stores_pattern(self):
        exc = GuardrailsViolation(isp="xtrim", pattern="DROP TABLE")
        assert exc.isp == "xtrim"
        assert exc.pattern == "DROP TABLE"

    def test_validation_error_stores_field(self):
        exc = ValidationError(isp="cnt", field="precio_plan", value="-5", reason="negativo")
        assert exc.field == "precio_plan"
        assert "negativo" in str(exc)

    def test_robots_disallowed_includes_url(self):
        url = "https://example.com/private"
        exc = RobotsDisallowedError(url=url)
        assert url in str(exc)


class TestExceptionCatching:
    """Tests del catch selectivo por tipo — patrón orquestador."""

    def test_catch_scraping_separately(self):
        """ScrapingError puede capturarse antes del catch genérico."""
        caught = None
        try:
            raise ScrapingError("netlife", "https://netlife.ec", "connection refused")
        except ScrapingError as e:
            caught = "scraping"
        except BenchmarkError:
            caught = "generic"
        assert caught == "scraping"

    def test_robots_not_caught_as_scraping(self):
        """RobotsDisallowedError es distinto de ScrapingError."""
        caught = None
        try:
            raise RobotsDisallowedError("https://blocked.com")
        except ScrapingError:
            caught = "scraping"
        except BenchmarkError:
            caught = "benchmark"
        assert caught == "benchmark"
