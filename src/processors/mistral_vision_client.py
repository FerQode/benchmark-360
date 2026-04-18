# src/processors/mistral_vision_client.py
"""
Mistral Pixtral Vision Client — Tier 2 para extracción de imágenes.

Pixtral-12B es el modelo multimodal de Mistral, excelente para:
- OCR de imágenes con precios y tablas
- Extracción de datos de banners publicitarios
- Lectura de texto en imágenes con fuentes no estándar

Posición en la cadena de fallback:
    gemini-3.1-pro-preview → pixtral-12-2409 → gemini-2.5-flash-lite
    → gpt-4o-mini

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# Importación segura de mistralai al nivel del módulo
# mistralai v2 el cliente principal está en mistralai.client.sdk
try:
    from mistralai.client.sdk import Mistral as _MistralClient
    _MISTRAL_AVAILABLE = True
except ImportError:
    try:
        # Fallback para v1 (por compatibilidad)
        from mistralai import Mistral as _MistralClient  # type: ignore
        _MISTRAL_AVAILABLE = True
    except ImportError:
        _MistralClient = None  # type: ignore
        _MISTRAL_AVAILABLE = False


class MistralVisionClient:
    """Cliente Mistral Pixtral para extracción visual de planes ISP.

    Pixtral-12B complementa a Gemini Pro cuando:
    - Gemini agota cuota en Vision
    - Las imágenes tienen texto muy pequeño (Pixtral destaca en OCR)
    - Los banners son en español con fuentes decorativas

    Args:
        api_key: Clave API de Mistral. Default: env MISTRAL_API_KEY.
        model: Modelo Mistral a usar (default: pixtral-12-2409).

    Example:
        >>> client = MistralVisionClient()
        >>> plans = await client.extract_from_image(image_path, "Xtrim")
        >>> print(f"Planes extraídos: {len(plans)}")
    """

    DEFAULT_MODEL = "pixtral-12-2409"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self._model = model

        if not self._api_key:
            logger.warning(
                "MISTRAL_API_KEY no configurada. "
                "Mistral Vision no estará disponible como fallback."
            )

    @property
    def is_available(self) -> bool:
        """True si mistralai está instalado Y la API key está configurada."""
        return _MISTRAL_AVAILABLE and bool(self._api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
    )
    async def extract_plans_from_image(
        self,
        image_b64: str,
        marca: str,
        tile_context: str = "",
    ) -> list[dict]:
        """Extrae planes ISP de una imagen usando Pixtral-12B.

        Args:
            image_b64: Imagen codificada en base64.
            marca: Nombre de la marca ISP para contexto.
            tile_context: Contexto adicional del tile (número, posición).

        Returns:
            Lista de planes crudos extraídos de la imagen.
        """
        if not self.is_available:
            if not _MISTRAL_AVAILABLE:
                logger.warning(
                    "Mistral SDK no instalado. "
                    "Ejecuta: uv add mistralai"
                )
            else:
                logger.warning("Mistral no disponible — API key faltante")
            return []

        # Usar el cliente importado al nivel del módulo (sin import lazy)
        client = _MistralClient(api_key=self._api_key)

        prompt = self._build_prompt(marca=marca, context=tile_context)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            }
        ]

        try:
            # mistralai v2: usar chat.complete (sin _async; el cliente es sync)
            # Para async usamos asyncio.to_thread para no bloquear el event loop
            import asyncio
            response = await asyncio.to_thread(
                client.chat.complete,
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096,
            )

            raw_text = response.choices[0].message.content
            logger.info(
                "[{}] Pixtral-12B respondió ({} chars)",
                marca,
                len(raw_text),
            )

            return self._parse_response(raw=raw_text, marca=marca)

        except Exception as exc:
            logger.error(
                "[{}] Mistral Vision error: {}", marca, exc
            )
            raise

    @staticmethod
    def _build_prompt(marca: str, context: str) -> str:
        """Construye el prompt para Pixtral.

        Args:
            marca: Nombre del ISP.
            context: Contexto adicional del tile.

        Returns:
            Prompt optimizado para Pixtral-12B.
        """
        return f"""Eres un extractor de datos de planes de internet ISP en Ecuador.

REGLAS:
1. JSON válido únicamente. Sin texto adicional.
2. NUNCA inventes datos. Si no está visible → null.
3. Sin planes visibles → {{"planes": []}}.
4. Precios en USD sin IVA (Ecuador IVA=15%).
5. Velocidades en Mbps.

EMPRESA: {marca}
{f"CONTEXTO: {context}" if context else ""}

Extrae TODOS los planes visibles:
{{
  "planes": [
    {{
      "nombre_plan": string | null,
      "velocidad_download_mbps": number | null,
      "velocidad_upload_mbps": number | null,
      "precio_plan": number | null,
      "precio_plan_descuento": number | null,
      "meses_descuento": integer | null,
      "costo_instalacion": number | null,
      "tecnologia": string | null,
      "pys_adicionales_detalle": {{}},
      "beneficios_publicitados": string | null
    }}
  ]
}}"""

    @staticmethod
    def _parse_response(raw: str, marca: str) -> list[dict]:
        """Parsea la respuesta JSON de Pixtral.

        Args:
            raw: Texto de respuesta del modelo.
            marca: Marca para logging.

        Returns:
            Lista de planes. Vacía si falla el parseo.
        """
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            return data.get("planes", [])
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning("[{}] Pixtral JSON inválido", marca)
            return []
