# src/config/settings.py
"""Fuente única de verdad para toda la configuración del proyecto Benchmark 360.

Inspirado en el 12-Factor App: la configuración viene del entorno,
nunca de constantes hardcodeadas dispersas por el código.

Uso:
    from src.config.settings import get_settings
    s = get_settings()
    print(s.cache_ttl_hours)  # 24
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración global leída desde .env con defaults tipados.

    Attributes:
        deepseek_api_key: API key de DeepSeek.
        gemini_api_key: API key de Google Gemini.
        groq_api_key: API key de Groq.
        mistral_api_key: API key de Mistral AI.
        openai_api_key: API key de OpenAI.
        request_delay_min: Delay mínimo entre requests (segundos).
        request_delay_max: Delay máximo entre requests (segundos).
        max_concurrent_scrapers: ISPs procesados en paralelo.
        playwright_timeout_ms: Timeout de Playwright en ms.
        semantic_tile_height: Alto de cada tile de screenshot (px).
        semantic_tile_overlap: Overlap entre tiles (px).
        max_retries_per_provider: Intentos por proveedor LLM.
        circuit_breaker_cooldown_seconds: Cooldown del circuit breaker.
        cache_ttl_hours: TTL del caché de respuestas LLM.
        cache_dir: Directorio de caché.
        output_dir: Directorio de salida de datos.
        logs_dir: Directorio de logs rotativos.
        parquet_compression: Algoritmo de compresión Parquet.
    """

    # ── API Keys ──────────────────────────────────────────────────
    deepseek_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    mistral_api_key: str = ""
    openai_api_key: str = ""

    # ── Scraping ──────────────────────────────────────────────────
    request_delay_min: float = 2.0
    request_delay_max: float = 5.0
    max_concurrent_scrapers: int = 3
    playwright_timeout_ms: int = 45_000
    semantic_tile_height: int = 1_200
    semantic_tile_overlap: int = 300

    # ── LLM & Resiliencia ─────────────────────────────────────────
    max_retries_per_provider: int = 3
    circuit_breaker_cooldown_seconds: int = 300

    # ── Cache & Almacenamiento ────────────────────────────────────
    cache_ttl_hours: int = 24
    cache_dir: str = "data/cache"
    output_dir: str = "data/output"
    logs_dir: str = "data/logs"
    parquet_compression: str = "snappy"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignorar variables desconocidas en .env
    )

    @property
    def base_dir(self) -> Path:
        """Raíz del proyecto (directorio que contiene pyproject.toml)."""
        return Path(__file__).parent.parent.parent


@lru_cache()
def get_settings() -> Settings:
    """Retorna instancia singleton de Settings (creada una sola vez).

    Asegura también que los directorios críticos existan en disco.

    Returns:
        Settings: Instancia singleton con toda la configuración.
    """
    settings = Settings()

    # Crear directorios si no existen
    for directory in [settings.cache_dir, settings.output_dir, settings.logs_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    return settings
