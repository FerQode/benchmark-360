# src/processors/llm_client_factory.py
"""
Factory de clientes LLM con cascada de fallback automática.

Jerarquía de modelos:
    Texto  → gemini-2.5-flash (pago, rápido y suficiente para HTML)
    Vision → gemini-3.1-pro-preview (universitario, máxima calidad)
    Tier 3 → gemini-2.5-flash-lite (gratuito Gemini)
    Tier 4 → gpt-4o-mini (OpenAI, último recurso)

IMPORTANTE — Compatibilidad:
    GeminiAdapter expone generate_text() y generate_vision()
    con la misma firma que el cliente anterior. El LLMProcessor
    NO requiere cambios en su lógica de negocio.

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum

from google import genai
from google.genai import types as genai_types
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


# ─────────────────────────────────────────────────────────────────
# Enums y constantes
# ─────────────────────────────────────────────────────────────────


class TaskType(str, Enum):
    """Tipo de tarea LLM que determina el modelo primario a usar.

    Attributes:
        TEXT: Extracción de HTML/texto estructurado → Flash.
        VISION: Extracción de imágenes/banners → Pro.
        COMPLEX: Razonamiento complejo → Pro.
    """

    TEXT = "text"
    VISION = "vision"
    COMPLEX = "complex"


# Tokens conocidos que indican error de cuota en Gemini y OpenAI
_QUOTA_ERROR_SIGNALS: tuple[str, ...] = (
    "RESOURCE_EXHAUSTED",
    "RATE_LIMIT_EXCEEDED",
    "quota",
    "429",
    "insufficient_quota",
    "rate limit",
)


def _is_quota_error(exc: Exception) -> bool:
    """Detecta si una excepción es por cuota/rate-limit agotada.

    Args:
        exc: Excepción capturada de cualquier proveedor LLM.

    Returns:
        True si el error indica cuota o rate-limit.
    """
    exc_str = str(exc).upper()
    return any(signal.upper() in exc_str for signal in _QUOTA_ERROR_SIGNALS)


# Rate limits conservadores por modelo (requests por minuto)
_MODEL_RPM_LIMITS: dict[str, float] = {
    "gemini-3.1-pro-preview": 10,   # Pro: 10 RPM en free tier
    "gemini-2.5-flash": 15,          # Flash: 15 RPM
    "gemini-2.5-flash-lite": 30,     # Lite: 30 RPM
    "gpt-4o-mini": 3,                # OpenAI free: muy restrictivo
    "pixtral-12-2409": 5,            # Mistral: conservador
}

# Tiempo mínimo entre llamadas al mismo modelo (segundos)
_MIN_DELAY_BETWEEN_CALLS: dict[str, float] = {
    model: 60.0 / rpm
    for model, rpm in _MODEL_RPM_LIMITS.items()
}


# ─────────────────────────────────────────────────────────────────
# Dataclass de resultado
# ─────────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Respuesta normalizada de cualquier proveedor LLM.

    Attributes:
        content: Texto de respuesta del modelo.
        model_used: Nombre exacto del modelo que respondió.
        tier_used: Tier numérico (0=primario, 1+=fallback).
        fallback_activated: True si se usó un modelo de respaldo.
        tokens_used: Tokens consumidos (si disponible).
    """

    content: str
    model_used: str
    tier_used: int = 0
    fallback_activated: bool = False
    tokens_used: int | None = None


# ─────────────────────────────────────────────────────────────────
# GeminiAdapter — Compatible con interfaz anterior
# ─────────────────────────────────────────────────────────────────


class GeminiAdapter:
    """Cliente LLM unificado con cascada automática de fallback.

    Soporta texto e imágenes a través de una interfaz única.
    Escala automáticamente por cuota agotada sin intervención manual.

    La cadena de fallback se configura por tipo de tarea:
    - TEXT:   Flash 2.5 → Flash Lite → GPT-4o-mini
    - VISION: Pro 3.1   → Flash 2.5  → Flash Lite → GPT-4o-mini

    Args:
        task_type: Tipo de tarea (TEXT o VISION).
        gemini_api_key: API key de Google. Por defecto usa GEMINI_API_KEY.
        openai_api_key: API key de OpenAI. Por defecto usa OPENAI_API_KEY.

    Example:
        >>> client = GeminiAdapter(task_type=TaskType.VISION)
        >>> response = await client.generate("Extrae planes", images=[b64])
        >>> print(response.model_used)
        'gemini-3.1-pro-preview'
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.TEXT,
        gemini_api_key: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        """Inicializa el adapter con la cadena de fallback por tarea.

        Args:
            task_type: Determina el modelo primario y la cadena.
            gemini_api_key: Key de Gemini. Default: env GEMINI_API_KEY.
            openai_api_key: Key de OpenAI. Default: env OPENAI_API_KEY.
        """
        self._task_type = task_type
        self._gemini_key = gemini_api_key or os.environ.get(
            "GEMINI_API_KEY", ""
        )
        self._openai_key = openai_api_key or os.environ.get(
            "OPENAI_API_KEY", ""
        )

        # Construir cadena según tarea
        self._fallback_chain: list[str] = self._build_chain(task_type)
        self._current_tier: int = 0
        self._total_calls: int = 0
        self._fallback_activations: int = 0
        # Registro de última llamada por modelo para rate limiting
        self._last_call_time: dict[str, float] = {}

        logger.info(
            "GeminiAdapter init — task={}, chain={}",
            task_type.value,
            self._fallback_chain,
        )

    # ── API pública ────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> LLMResponse:
        """Genera texto con cascada automática de fallback.

        Punto de entrada único para texto e imágenes.
        No requiere conocer el modelo subyacente.

        Args:
            prompt: Prompt de instrucción para el modelo.
            images_b64: Lista de imágenes en base64 (vision tasks).
            temperature: Temperatura (0.0 = determinístico).
            max_tokens: Máximo de tokens en la respuesta.

        Returns:
            LLMResponse con contenido y metadata del modelo usado.

        Raises:
            RuntimeError: Si TODOS los modelos de la cadena fallan.
        """
        last_exc: Exception | None = None

        for tier in range(self._current_tier, len(self._fallback_chain)):
            model_id = self._fallback_chain[tier]

            # Respetar rate limit antes de cada llamada
            await self._wait_for_rate_limit(model_id)

            try:
                self._total_calls += 1

                if model_id.startswith("gpt"):
                    content = await self._call_openai(
                        model_id=model_id,
                        prompt=prompt,
                        images_b64=images_b64,
                        max_tokens=max_tokens,
                    )
                else:
                    content = await self._call_gemini(
                        model_id=model_id,
                        prompt=prompt,
                        images_b64=images_b64,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                # ── Éxito ──
                if tier > 0:
                    self._fallback_activations += 1
                    self._current_tier = tier  # Persistir tier activo
                    logger.warning(
                        "⚠️  Fallback tier {} activado: {} "
                        "(cuota agotada en tier superior)",
                        tier,
                        model_id,
                    )

                return LLMResponse(
                    content=content,
                    model_used=model_id,
                    tier_used=tier,
                    fallback_activated=tier > 0,
                )

            except Exception as exc:
                last_exc = exc

                if _is_quota_error(exc):
                    logger.warning(
                        "💳 Cuota agotada en '{}' → escalando a tier {}",
                        model_id,
                        tier + 1,
                    )
                    continue  # Siguiente tier en la cadena
                else:
                    # Error de red/API no relacionado con cuota → retry
                    logger.error(
                        "❌ Error no-cuota en '{}': {}",
                        model_id,
                        exc,
                    )
                    raise

        raise RuntimeError(
            f"Todos los modelos LLM fallaron. Cadena: "
            f"{self._fallback_chain}. Último error: {last_exc}"
        )

    # ── Propiedades de observabilidad ─────────────────────────────

    @property
    def provider_name(self) -> str:
        """Modelo activo actualmente en la cadena."""
        return self._fallback_chain[self._current_tier]

    @property
    def fallback_activated(self) -> bool:
        """True si algún fallback se activó en esta sesión."""
        return self._fallback_activations > 0

    @property
    def total_calls(self) -> int:
        """Total de llamadas LLM realizadas."""
        return self._total_calls

    async def _wait_for_rate_limit(self, model_id: str) -> None:
        """Espera el tiempo mínimo entre llamadas para respetar RPM.

        Args:
            model_id: Modelo al que se va a llamar.
        """
        min_delay = _MIN_DELAY_BETWEEN_CALLS.get(model_id, 4.0)
        last_call = self._last_call_time.get(model_id, 0.0)
        elapsed = time.monotonic() - last_call
        wait_time = max(0.0, min_delay - elapsed)

        if wait_time > 0.1:
            logger.debug(
                "⏳ Rate limit: esperando {:.1f}s antes de llamar a {}",
                wait_time,
                model_id,
            )
            await asyncio.sleep(wait_time)

        self._last_call_time[model_id] = time.monotonic()

    # ── Construcción de cadena ─────────────────────────────────────

    @staticmethod
    def _build_chain(task_type: TaskType) -> list[str]:
        """Construye la cadena de fallback según el tipo de tarea.

        Args:
            task_type: Tipo de tarea a procesar.

        Returns:
            Lista ordenada de model_ids de mayor a menor calidad.
        """
        text_model = os.getenv(
            "GEMINI_MODEL_TEXT", "gemini-2.5-flash"
        )
        vision_model = os.getenv(
            "GEMINI_MODEL_VISION", "gemini-3.1-pro-preview"
        )
        fallback_model = os.getenv(
            "GEMINI_MODEL_FALLBACK", "gemini-2.5-flash-lite"
        )
        openai_model = os.getenv(
            "OPENAI_MODEL_FALLBACK", "gpt-4o-mini"
        )

        if task_type == TaskType.VISION:
            return [vision_model, text_model, fallback_model, openai_model]
        else:
            return [text_model, fallback_model, openai_model]

    # ── Llamadas a APIs ────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception(
            lambda e: not _is_quota_error(e)
        ),
    )
    async def _call_gemini(
        self,
        model_id: str,
        prompt: str,
        images_b64: list[str] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Llama a la API de Gemini (nuevo SDK google.genai) con retry.

        Usa el nuevo SDK google-genai que reemplaza google-generativeai.
        El retry se aplica SOLO a errores no relacionados con cuota.

        Args:
            model_id: Identificador del modelo Gemini.
            prompt: Texto del prompt.
            images_b64: Imágenes en base64 (None para text-only).
            temperature: Temperatura de generación.
            max_tokens: Máximo tokens de output.

        Returns:
            Texto de respuesta del modelo.
        """
        import base64

        client = genai.Client(api_key=self._gemini_key)

        contents: list = [prompt]

        if images_b64:
            for img_b64 in images_b64:
                contents.append(
                    genai_types.Part.from_bytes(
                        data=base64.b64decode(img_b64),
                        mime_type="image/png",
                    )
                )

        response = await client.aio.models.generate_content(
            model=model_id,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        self._last_call_time[model_id] = time.monotonic()
        return response.text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception(
            lambda e: not _is_quota_error(e)
        ),
    )
    async def _call_openai(
        self,
        model_id: str,
        prompt: str,
        images_b64: list[str] | None,
        max_tokens: int,
    ) -> str:
        """Fallback a OpenAI cuando toda la cadena Gemini agota cuota.

        Args:
            model_id: ID del modelo OpenAI (ej: gpt-4o-mini).
            prompt: Texto del prompt.
            images_b64: Imágenes en base64 para vision (opcional).
            max_tokens: Máximo tokens de output.

        Returns:
            Texto de respuesta del modelo OpenAI.

        Raises:
            RuntimeError: Si OPENAI_API_KEY no está configurada.
        """
        if not self._openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY no configurada. "
                "Añádela al .env para usar el fallback de OpenAI."
            )

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._openai_key)
        user_content: list = [{"type": "text", "text": prompt}]

        if images_b64:
            for img_b64 in images_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high",
                    },
                })

        response = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────
# LLMClientFactory — Interfaz pública sin cambios
# ─────────────────────────────────────────────────────────────────


class LLMClientFactory:
    """Factory que crea GeminiAdapters según el tipo de tarea.

    Mantiene la interfaz que espera el Orchestrator actual.
    El Orchestrator NO requiere ningún cambio.

    Example:
        >>> factory = LLMClientFactory()
        >>> text_client = factory.get_text_client()
        >>> vision_client = factory.get_vision_client()
    """

    def get_text_client(self) -> GeminiAdapter:
        """Crea cliente optimizado para extracción de HTML/texto.

        Returns:
            GeminiAdapter con Flash como primario (rápido y barato).
        """
        return GeminiAdapter(task_type=TaskType.TEXT)

    def get_vision_client(self) -> GeminiAdapter:
        """Crea cliente optimizado para extracción de imágenes.

        Returns:
            GeminiAdapter con Pro 3.1 como primario (máxima calidad).
        """
        return GeminiAdapter(task_type=TaskType.VISION)

    def get_mistral_client(self):
        """Crea cliente Mistral Pixtral para tier 2 de Vision.

        Returns:
            MistralVisionClient si MISTRAL_API_KEY está configurada.
        """
        from src.processors.mistral_vision_client import MistralVisionClient
        return MistralVisionClient()

    def get_primary_client(self) -> GeminiAdapter:
        """Compatibilidad con Orchestrator actual (usa TEXT por defecto).

        Returns:
            GeminiAdapter para tareas de texto.
        """
        return self.get_text_client()

    def get_fallback_client(self) -> GeminiAdapter:
        """Compatibilidad con Orchestrator actual.

        Returns:
            GeminiAdapter con cadena completa de fallback.
        """
        return GeminiAdapter(task_type=TaskType.TEXT)
