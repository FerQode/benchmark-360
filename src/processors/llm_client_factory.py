# src/processors/llm_client_factory.py
"""
Factory de clientes LLM con arquitectura Multi-Provider Waterfall.

Usa el patrón Singleton para el MultiProviderAdapter, garantizando
que el client pool HTTP y el caché se reutilicen en toda la sesión
del pipeline, no solo entre llamadas al mismo ISP.

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import logging
from typing import Any

from src.processors.multi_provider_adapter import LLMResponse, MultiProviderAdapter
from src.processors.provider_registry import TaskType

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """Factory singleton para MultiProviderAdapter.

    Mantiene UNA sola instancia del adapter por tipo de tarea
    para preservar el client pool HTTP entre todos los ISPs
    procesados en una sesión del pipeline.

    Attributes:
        _text_adapter: Instancia singleton para tareas de texto.
        _vision_adapter: Instancia singleton para tareas de visión.

    Example:
        >>> factory = LLMClientFactory()
        >>> # Mismo pool para todos los ISPs:
        >>> client_netlife = factory.get_text_client()
        >>> client_claro   = factory.get_text_client()  # misma instancia
        >>> assert client_netlife is client_claro  # True
    """

    def __init__(self) -> None:
        """Inicializa el factory con adapters lazy (se crean al primer uso)."""
        self._text_adapter: MultiProviderAdapter | None = None
        self._vision_adapter: MultiProviderAdapter | None = None

    def get_text_client(self) -> MultiProviderAdapter:
        """Retorna el adapter singleton para texto.

        El adapter es creado en el primer llamado y reutilizado
        en todas las llamadas subsiguientes de la sesión.

        Returns:
            MultiProviderAdapter configurado para extracción de texto.
        """
        if self._text_adapter is None:
            self._text_adapter = MultiProviderAdapter(
                task_type=TaskType.TEXT
            )
            logger.info("MultiProviderAdapter TEXT inicializado (singleton).")
        return self._text_adapter

    def get_vision_client(self) -> MultiProviderAdapter:
        """Retorna el adapter singleton para visión.

        Returns:
            MultiProviderAdapter configurado para análisis de imágenes.
        """
        if self._vision_adapter is None:
            self._vision_adapter = MultiProviderAdapter(
                task_type=TaskType.VISION
            )
            logger.info("MultiProviderAdapter VISION inicializado (singleton).")
        return self._vision_adapter

    def get_primary_client(self) -> MultiProviderAdapter:
        """Compatibilidad con Orchestrator — alias de get_text_client.

        Returns:
            MultiProviderAdapter de texto.
        """
        return self.get_text_client()

    def get_fallback_client(self) -> MultiProviderAdapter:
        """Compatibilidad con Orchestrator — alias de get_text_client.

        El waterfall interno ya maneja los fallbacks automáticamente.

        Returns:
            MultiProviderAdapter de texto (el fallback es interno).
        """
        return self.get_text_client()

    def get_mistral_client(self) -> MultiProviderAdapter:
        """Compatibilidad — Pixtral ya está integrado en el waterfall de visión.

        Nota: mistral_pixtral es el Tier 2 de visión en PROVIDER_REGISTRY.
        No se necesita un cliente separado.

        Returns:
            MultiProviderAdapter de visión que incluye Pixtral.
        """
        return self.get_vision_client()

    def print_session_report(self) -> None:
        """Imprime el resumen de toda la sesión (texto + visión)."""
        for label, adapter in [
            ("TEXTO", self._text_adapter),
            ("VISIÓN", self._vision_adapter),
        ]:
            if adapter is None:
                continue
            summary = adapter.get_execution_summary()
            logger.info(f"\n{'═'*55}")
            logger.info(f"  📊 REPORTE SESIÓN [{label}]")
            logger.info(f"{'═'*55}")
            logger.info(f"  Pipeline: {summary['pipeline']}")
            logger.info(f"  Caché:    {summary['cache']}")
            for p in summary["providers"]:
                logger.info(f"  {p['name']:<18} {p['requests']} req | ${p['cost_usd']:.4f}")


# Alias de compatibilidad total — código existente que importa GeminiAdapter
# no requiere cambios
GeminiAdapter = MultiProviderAdapter
