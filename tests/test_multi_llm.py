# tests/test_multi_llm.py
"""
Tests de integración del sistema Multi-LLM Waterfall.

Valida el comportamiento de fallback sin llamadas reales a APIs.
Usa unittest.mock para simular respuestas y errores.

Ejecutar con:
    uv run pytest tests/test_multi_llm.py -v

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.processors.llm_client_factory import LLMClientFactory
from src.processors.multi_provider_adapter import MultiProviderAdapter
from src.processors.provider_registry import (
    PROVIDER_REGISTRY,
    ProviderStatus,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_provider_registry():
    """Restaura el estado COMPLETO del registry antes y después de cada test.

    Guarda y restaura TODOS los campos mutables para garantizar
    aislamiento total entre tests.
    """
    original_state = []
    for p in PROVIDER_REGISTRY:
        original_state.append({
            "name": p.name,
            "status": p.status,
            "requests": p.requests_this_session,
            "tokens": p.tokens_input_session,
            "fails": p._consecutive_failures,
            "rpm_limit": p.rpm_limit,
            "last_ts": p._last_request_ts,
        })

    yield

    for i, p in enumerate(PROVIDER_REGISTRY):
        s = original_state[i]
        p.status = s["status"]
        p.requests_this_session = s["requests"]
        p.tokens_input_session = s["tokens"]
        p._consecutive_failures = s["fails"]
        p.rpm_limit = s["rpm_limit"]
        p._last_request_ts = s["last_ts"]


@pytest.fixture
def mock_openai_response():
    """Crea un mock de respuesta AsyncOpenAI con planes válidos."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps([
        {
            "nombre_plan": "Plan Test 200 Mbps",
            "velocidad_download_mbps": 200.0,
            "precio_plan": 21.74,
        }
    ])
    return mock_response


# ── Tests de Comportamiento ───────────────────────────────────────

class TestWaterfallFallback:
    """Tests del comportamiento de cascada entre proveedores."""

    @pytest.mark.asyncio
    async def test_fallback_cuando_gemini_agotado(
        self, mock_openai_response
    ):
        """Verifica que el sistema usa otro proveedor cuando Gemini está agotado."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-fake-key"}):
            for p in PROVIDER_REGISTRY:
                if p.name == "gemini_flash":
                    p.status = ProviderStatus.QUOTA_EXHAUSTED
                if p.name == "deepseek":
                    p.status = ProviderStatus.AVAILABLE

            factory = LLMClientFactory()
            client = factory.get_text_client()

            with patch.object(
                client, "_call_provider",
                new_callable=AsyncMock,
                return_value=mock_openai_response.choices[0].message.content
            ):
                response = await client.generate(
                    prompt="Extrae planes",
                    content="<div>Plan 200 Mbps $22</div>",
                )

            assert response.model_used != "gemini_flash"
            assert response.model_used != "none_error"
            assert response.content != "[]"

    @pytest.mark.asyncio
    async def test_cache_hit_evita_llamada_api(self):
        """Verifica que la segunda llamada con mismo contenido usa caché."""
        factory = LLMClientFactory()
        client = factory.get_text_client()
        content = "<div>Plan Único 500 Mbps $30</div>"

        client.cache.set(
            content,
            "text",
            [{"nombre_plan": "Plan Único 500 Mbps", "precio_plan": 30.0}],
            provider_name="test_fixture",
        )

        with patch.object(
            client, "_call_provider",
            new_callable=AsyncMock
        ) as mock_call:
            response = await client.generate(
                prompt="Extrae planes", content=content
            )
            mock_call.assert_not_called()

        assert response.model_used == "cache_disk"

    @pytest.mark.asyncio
    async def test_todos_los_proveedores_fallan(self):
        """Verifica comportamiento cuando todos los proveedores fallan."""
        with patch("src.processors.multi_provider_adapter.get_text_providers", return_value=[]):
            factory = LLMClientFactory()
            client = factory.get_text_client()

            response = await client.generate(
                prompt="Extrae planes",
                content="<div>cualquier contenido</div>",
            )

            assert response.model_used == "none_error"
            assert response.content == "[]"
            assert client.stats["failed_calls"] == 1


class TestRateLimiting:
    """Tests del sistema de rate limiting por proveedor."""

    @pytest.mark.asyncio
    async def test_wait_rate_limit_respeta_rpm(self):
        """Verifica que wait_rate_limit espera el tiempo correcto.

        Usa rpm_limit alto (600 RPM = 0.1s entre requests) para
        que el test sea rápido (<200ms) pero verificable.
        """
        groq = next(p for p in PROVIDER_REGISTRY if p.name == "groq")
        groq.rpm_limit = 600
        groq._last_request_ts = time.monotonic()

        start = time.monotonic()
        await groq.wait_rate_limit()
        elapsed = time.monotonic() - start

        expected = 60.0 / 600
        assert elapsed >= expected * 0.8

    @pytest.mark.asyncio
    async def test_segunda_llamada_inmediata_no_espera(self):
        """Verifica que llamada tras esperar el intervalo es inmediata."""
        groq = next(p for p in PROVIDER_REGISTRY if p.name == "groq")
        groq.rpm_limit = 600
        groq._last_request_ts = time.monotonic() - 10.0

        start = time.monotonic()
        await groq.wait_rate_limit()
        elapsed = time.monotonic() - start

        assert elapsed < 0.05


class TestCircuitBreaker:
    """Tests del Circuit Breaker por fallos consecutivos."""

    def test_tres_fallos_consecutivos_marca_error(self):
        """Proveedor con 3 fallos seguidos se marca como ERROR."""
        groq = next(p for p in PROVIDER_REGISTRY if p.name == "groq")

        for _ in range(3):
            groq.record_usage(success=False)

        assert groq.status == ProviderStatus.ERROR

    def test_exito_resetea_contador_fallos(self):
        """Un éxito después de fallos resetea el contador."""
        groq = next(p for p in PROVIDER_REGISTRY if p.name == "groq")

        groq.record_usage(success=False)
        groq.record_usage(success=False)
        assert groq._consecutive_failures == 2

        groq.record_usage(success=True)
        assert groq._consecutive_failures == 0
        assert groq.status != ProviderStatus.ERROR


class TestParseResponse:
    """Tests del parser de respuestas LLM con formatos variados."""

    def setup_method(self):
        """Crea adapter para cada test."""
        self.adapter = MultiProviderAdapter()

    def test_parse_json_array_limpio(self):
        """Parsea un JSON array directo."""
        raw = '[{"nombre_plan": "Test", "precio_plan": 20.0}]'
        result = self.adapter._parse_response(raw, "test")
        assert len(result) == 1
        assert result[0]["nombre_plan"] == "Test"

    def test_parse_json_con_markdown(self):
        """Parsea JSON envuelto en markdown code block."""
        raw = '```json\n[{"nombre_plan": "Test", "precio_plan": 20.0}]\n```'
        result = self.adapter._parse_response(raw, "test")
        assert len(result) == 1

    def test_parse_json_objeto_con_key_planes(self):
        """Parsea objeto con clave 'planes'."""
        raw = '{"planes": [{"nombre_plan": "Test"}]}'
        result = self.adapter._parse_response(raw, "test")
        assert len(result) == 1

    def test_parse_json_invalido_retorna_lista_vacia(self):
        """JSON completamente inválido retorna lista vacía (no lanza)."""
        raw = "Esto no es JSON para nada"
        result = self.adapter._parse_response(raw, "test")
        assert result == []

    def test_parse_string_vacio_retorna_lista_vacia(self):
        """String vacío retorna lista vacía."""
        result = self.adapter._parse_response("", "test")
        assert result == []
