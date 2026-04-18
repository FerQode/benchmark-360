# src/processors/__init__.py
"""
Processors package — LLM extraction, vision, normalization layer.

Complete Phase 5 + Phase 6 exports for the pipeline orchestrator.

Typical usage example:
    >>> from src.processors import (
    ...     LLMClientFactory, LLMProcessor, VisionProcessor,
    ...     ArmaTuPlanHandler, PlanNormalizer, GuardrailsEngine,
    ... )
"""

from src.processors.arma_tu_plan_handler import ArmaTuPlanHandler, ExpandedPlan
from src.processors.guardrails import GuardrailResult, GuardrailsEngine
from src.processors.llm_client_factory import LLMClient, LLMClientFactory
from src.processors.llm_processor import LLMExtractionResult, LLMProcessor
from src.processors.normalizer import PlanNormalizer
from src.processors.pys_catalog import (
    PYS_ALIAS_MAP,
    PYS_CATEGORY_MAP,
    get_pys_category,
    normalize_pys_detalle,
    normalize_pys_key,
)
from src.processors.vision_processor import (
    VisionExtractionResult,
    VisionProcessor,
)

__all__ = [
    # Factory
    "LLMClientFactory",
    "LLMClient",
    # Processors
    "LLMProcessor",
    "VisionProcessor",
    "ArmaTuPlanHandler",
    "PlanNormalizer",
    # Guardrails
    "GuardrailsEngine",
    "GuardrailResult",
    # PYS Catalog
    "PYS_ALIAS_MAP",
    "PYS_CATEGORY_MAP",
    "normalize_pys_key",
    "normalize_pys_detalle",
    "get_pys_category",
    # Result DTOs
    "LLMExtractionResult",
    "VisionExtractionResult",
    "ExpandedPlan",
]
