# tests/test_settings.py
"""Tests del módulo de configuración centralizado."""

from __future__ import annotations

import pytest

from src.config.settings import Settings, get_settings


class TestSettingsDefaults:
    """Tests de valores por defecto y tipos."""

    def test_get_settings_returns_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # Mismo objeto (lru_cache)

    def test_default_cache_ttl(self):
        s = get_settings()
        assert s.cache_ttl_hours == 24

    def test_default_concurrent_scrapers(self):
        s = get_settings()
        assert s.max_concurrent_scrapers == 3

    def test_default_playwright_timeout(self):
        s = get_settings()
        assert s.playwright_timeout_ms > 0

    def test_default_parquet_compression(self):
        s = get_settings()
        assert s.parquet_compression in ("snappy", "gzip", "brotli", "zstd")

    def test_settings_has_api_key_fields(self):
        s = get_settings()
        # Deben existir como atributos (pueden estar vacíos en tests)
        assert hasattr(s, "openai_api_key")
        assert hasattr(s, "gemini_api_key")
        assert hasattr(s, "groq_api_key")
        assert hasattr(s, "mistral_api_key")
        assert hasattr(s, "deepseek_api_key")

    def test_directories_exist_after_init(self):
        """get_settings() debe crear los directorios de datos."""
        from pathlib import Path
        s = get_settings()
        assert Path(s.cache_dir).exists()
        assert Path(s.output_dir).exists()
        assert Path(s.logs_dir).exists()
