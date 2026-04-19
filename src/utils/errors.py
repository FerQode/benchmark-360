# src/utils/errors.py
"""Jerarquía de excepciones personalizadas para Benchmark 360.

Inspirado en la jerarquía de errores de Stripe: cada tipo de fallo
tiene su propia clase, permitiendo al orquestador decidir si hace
retry, degrada gracefully, o lanza alerta crítica.

Uso:
    from src.utils.errors import ScrapingError, LLMQuotaExhaustedError
    raise ScrapingError(isp="netlife", url="https://netlife.ec", reason="timeout")
"""

from __future__ import annotations


class BenchmarkError(Exception):
    """Clase base para todos los errores del pipeline Benchmark 360."""


class ScrapingError(BenchmarkError):
    """Error en la capa de extracción de datos web.

    Args:
        isp: Clave del ISP que falló (ej: 'netlife').
        url: URL que no pudo ser procesada.
        reason: Descripción del problema.
    """

    def __init__(self, isp: str, url: str, reason: str) -> None:
        self.isp = isp
        self.url = url
        self.reason = reason
        super().__init__(f"[{isp}] Scraping failed for {url}: {reason}")


class LLMProcessingError(BenchmarkError):
    """Error en la capa de procesamiento con IA.

    Args:
        provider: Nombre del proveedor LLM que falló.
        reason: Descripción del problema.
    """

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"[{provider}] LLM processing failed: {reason}")


class LLMQuotaExhaustedError(LLMProcessingError):
    """Lanzada cuando TODOS los proveedores del waterfall fallaron o agotaron cuota.

    Permite al orquestador degradar gracefully en lugar de propagar
    una excepción genérica.
    """

    def __init__(self, reason: str = "All providers in waterfall exhausted") -> None:
        super().__init__("Waterfall", reason)


class ValidationError(BenchmarkError):
    """Error en la validación Pydantic o normalización de datos.

    Args:
        isp: Clave del ISP con el dato inválido.
        field: Nombre del campo que falló validación.
        value: Valor recibido que no cumplió el schema.
        reason: Descripción del problema.
    """

    def __init__(self, isp: str, field: str, value: str, reason: str) -> None:
        self.isp = isp
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(
            f"[{isp}] Validation failed for '{field}'='{value}': {reason}"
        )


class GuardrailsViolation(BenchmarkError):
    """Se detectó una amenaza de seguridad (inyección de prompt, XSS, etc.).

    Esta excepción debe tratarse como alerta crítica, no como error recuperable.

    Args:
        isp: Clave del ISP cuyo contenido fue bloqueado.
        pattern: Patrón de ataque detectado.
        context: Contexto donde se detectó ('input' o 'output').
    """

    def __init__(
        self, isp: str, pattern: str, context: str = "input"
    ) -> None:
        self.isp = isp
        self.pattern = pattern
        self.context = context
        super().__init__(
            f"[{isp}] SECURITY: {context.upper()} injection detected → '{pattern}'"
        )


class RobotsDisallowedError(BenchmarkError):
    """La URL está prohibida por la política robots.txt del sitio.

    Este error NO debe reintentarse — respetar siempre robots.txt.

    Args:
        url: URL bloqueada por robots.txt.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(f"robots.txt disallows access to: {url}")
