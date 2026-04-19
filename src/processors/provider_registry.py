# src/processors/provider_registry.py
"""
Registry centralizado de proveedores LLM para el pipeline Benchmark 360.

Cada proveedor expone una interfaz compatible con OpenAI para que el
MultiProviderAdapter pueda enrutarlos de forma transparente.

Principio de diseño: Open/Closed — agregar un proveedor nuevo = 1 entrada
en PROVIDER_REGISTRY. Cero cambios en el resto del pipeline.

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Enums de estado
# ─────────────────────────────────────────────────────────────────

class ProviderStatus(str, Enum):
    """Estado operativo de un proveedor LLM.

    Attributes:
        AVAILABLE: Proveedor operativo con cuota disponible.
        QUOTA_EXHAUSTED: Sin tokens o requests disponibles.
        ERROR: Fallo técnico temporal (se reintenta en próxima sesión).
        DISABLED: Desactivado manualmente o sin API key.
    """

    AVAILABLE = "available"
    QUOTA_EXHAUSTED = "quota_exhausted"
    ERROR = "error"
    DISABLED = "disabled"


class TaskType(str, Enum):
    """Tipo de tarea para enrutamiento inteligente.

    Attributes:
        TEXT: Extracción desde HTML/texto plano.
        VISION: Extracción desde imágenes/capturas de pantalla.
    """

    TEXT = "text"
    VISION = "vision"


# ─────────────────────────────────────────────────────────────────
# Dataclass de configuración de proveedor
# ─────────────────────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    """Configuración completa de un proveedor LLM.

    Todos los proveedores usan la interfaz compatible con OpenAI.
    Esto permite usar un único cliente AsyncOpenAI para todos.

    Attributes:
        name: Identificador único del proveedor.
        base_url: URL base de la API (compatible OpenAI v1).
        api_key_env: Variable de entorno que contiene la API key.
        text_model: Nombre del modelo para tareas de texto.
        vision_model: Nombre del modelo para tareas de visión.
        priority_text: Prioridad en fallback de texto (1 = primero).
        priority_vision: Prioridad en fallback de visión (1 = primero).
        supports_vision: Si el proveedor acepta imágenes en el payload.
        supports_json_mode: Si acepta response_format=json_object.
        rpm_limit: Requests por minuto del tier gratuito/activo.
        daily_request_limit: Límite diario de requests (0 = sin límite).
        status: Estado operativo actual.
        requests_this_session: Contador de requests en la sesión actual.
        tokens_input_session: Tokens de entrada consumidos en la sesión.
        cost_per_1k_input_usd: Costo USD por 1,000 tokens de entrada.
    """

    name: str
    base_url: str
    api_key_env: str
    text_model: str
    vision_model: str | None
    priority_text: int
    priority_vision: int
    supports_vision: bool = False
    supports_json_mode: bool = True
    rpm_limit: int = 20
    daily_request_limit: int = 0
    status: ProviderStatus = ProviderStatus.AVAILABLE
    requests_this_session: int = field(default=0, repr=False)
    tokens_input_session: int = field(default=0, repr=False)
    cost_per_1k_input_usd: float = 0.0
    _last_request_ts: float = field(default=0.0, repr=False)
    
    # Lock para rate limiting coroutine-safe
    _rate_lock: asyncio.Lock | None = field(default=None, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Inicializa el Lock en el event loop activo.

        asyncio.Lock debe crearse dentro de una corrutina o cuando
        hay un event loop activo. Usar __post_init__ garantiza esto
        al momento de instanciar el ProviderConfig, no al importar.
        """
        # Inicialización diferida del Lock — compatible con Python 3.12
        object.__setattr__(self, "_rate_lock", asyncio.Lock())

    # ── Propiedades ───────────────────────────────────────────────

    @property
    def api_key(self) -> str | None:
        """Retorna la API key desde las variables de entorno.

        Returns:
            Valor de la variable de entorno, o None si no existe.
        """
        return os.getenv(self.api_key_env)

    @property
    def is_available(self) -> bool:
        """Verifica si el proveedor está listo para recibir requests.

        Un proveedor está disponible si tiene API key configurada
        y su estado es AVAILABLE.

        Returns:
            True si está operativo y configurado.
        """
        return (
            self.status == ProviderStatus.AVAILABLE
            and bool(self.api_key)
        )

    @property
    def estimated_cost_usd(self) -> float:
        """Costo estimado acumulado en la sesión actual.

        Returns:
            Costo en USD basado en tokens consumidos.
        """
        return (self.tokens_input_session / 1_000) * self.cost_per_1k_input_usd

    def get_model(self, task: TaskType) -> str:
        """Retorna el modelo apropiado para el tipo de tarea.

        Args:
            task: Tipo de tarea a ejecutar.

        Returns:
            Nombre del modelo configurado para la tarea.

        Raises:
            ValueError: Si se solicita visión pero el proveedor no la soporta.
        """
        if task == TaskType.VISION:
            if not self.supports_vision or not self.vision_model:
                raise ValueError(
                    f"Proveedor '{self.name}' no soporta visión. "
                    "Seleccionar otro proveedor del cascade."
                )
            return self.vision_model
        return self.text_model

    async def wait_rate_limit(self) -> None:
        """Espera respetando RPM con protección contra race conditions."""
        # Guard por si el lock no fue inicializado (edge case en tests)
        lock = self._rate_lock
        if lock is None:
            object.__setattr__(self, "_rate_lock", asyncio.Lock())
            lock = self._rate_lock

        async with lock:
            min_interval = 60.0 / max(self.rpm_limit, 1)
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request_ts = time.monotonic()

    def record_usage(
        self,
        tokens_in: int = 0,
        tokens_out: int = 0,
        success: bool = True,
    ) -> None:
        """Registra consumo y auto-agota el proveedor si alcanza límites.

        Args:
            tokens_in: Tokens de entrada de la llamada.
            tokens_out: Tokens de salida generados.
            success: False si la llamada terminó en error.
        """
        self.requests_this_session += 1
        self.tokens_input_session += tokens_in

        # Auto-exhaustion por límite diario de requests (ej. Groq 14.4k/day)
        if (
            self.daily_request_limit > 0
            and self.requests_this_session >= self.daily_request_limit * 0.9
        ):
            logger.warning(
                f"[{self.name}] Al 90% del límite diario "
                f"({self.requests_this_session}/{self.daily_request_limit}). "
                "Marcando como QUOTA_EXHAUSTED preventivamente."
            )
            self.status = ProviderStatus.QUOTA_EXHAUSTED

        # Escalamiento por fallos consecutivos (Circuit Breaker simple)
        if not success:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                logger.error(
                    f"[{self.name}] {self._consecutive_failures} fallos consecutivos. "
                    "Marcando como ERROR."
                )
                self.status = ProviderStatus.ERROR
        else:
            self._consecutive_failures = 0


# ─────────────────────────────────────────────────────────────────
# REGISTRY: Definición de todos los proveedores disponibles
# ─────────────────────────────────────────────────────────────────

PROVIDER_REGISTRY: list[ProviderConfig] = [

    # ──────────────────────────────────────────────────────────────
    # TIER 1 TEXTO — DeepSeek V3
    # ──────────────────────────────────────────────────────────────
    ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        text_model="deepseek-chat",
        vision_model=None,
        priority_text=1,
        priority_vision=99,
        supports_vision=False,
        supports_json_mode=True,
        rpm_limit=60,
        daily_request_limit=0,
        cost_per_1k_input_usd=0.00014,
    ),

    # ──────────────────────────────────────────────────────────────
    # TIER 2 TEXTO / TIER 1 VISIÓN — Google Gemini 2.5 Flash
    # ──────────────────────────────────────────────────────────────
    ProviderConfig(
        name="gemini_flash",
        base_url=(
            "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
        api_key_env="GEMINI_API_KEY",
        text_model="gemini-2.5-flash",
        vision_model="gemini-2.5-flash",
        priority_text=2,
        priority_vision=1,
        supports_vision=True,
        supports_json_mode=True,
        rpm_limit=15,
        daily_request_limit=1_500,
        cost_per_1k_input_usd=0.0,
    ),

    # ──────────────────────────────────────────────────────────────
    # TIER 3 TEXTO / TIER 3 VISIÓN — Groq + Llama 3.3 70B
    # ──────────────────────────────────────────────────────────────
    ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        text_model="llama-3.3-70b-versatile",
        vision_model="llama-3.2-11b-vision-preview",
        priority_text=3,
        priority_vision=3,
        supports_vision=True,
        supports_json_mode=True,
        rpm_limit=30,
        daily_request_limit=14_400,
        cost_per_1k_input_usd=0.0,
    ),

    # ──────────────────────────────────────────────────────────────
    # TIER 4 TEXTO / TIER 2 VISIÓN — Mistral + Pixtral
    # ──────────────────────────────────────────────────────────────
    ProviderConfig(
        name="mistral_pixtral",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        text_model="mistral-small-latest",
        vision_model="pixtral-large-latest",
        priority_text=4,
        priority_vision=2,
        supports_vision=True,
        supports_json_mode=True,
        rpm_limit=10,
        daily_request_limit=0,
        cost_per_1k_input_usd=0.0002,
    ),

    # ──────────────────────────────────────────────────────────────
    # TIER 5 — Ollama Local (INFINITO, $0, sin límites)
    # ──────────────────────────────────────────────────────────────
    ProviderConfig(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        text_model="llama3.2:3b",
        vision_model="llava:13b",
        priority_text=5,
        priority_vision=5,
        supports_vision=True,
        supports_json_mode=False,
        rpm_limit=999,
        daily_request_limit=0,
        cost_per_1k_input_usd=0.0,
    ),

    # ──────────────────────────────────────────────────────────────
    # TIER 6 — OpenAI GPT-4o-mini (último recurso con billing)
    # ──────────────────────────────────────────────────────────────
    ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        text_model="gpt-4o-mini",
        vision_model="gpt-4o-mini",
        priority_text=6,
        priority_vision=4,
        supports_vision=True,
        supports_json_mode=True,
        rpm_limit=20,
        daily_request_limit=0,
        cost_per_1k_input_usd=0.00015,
    ),
]


def get_text_providers() -> list[ProviderConfig]:
    """Retorna proveedores disponibles para texto, por prioridad.

    Returns:
        Lista filtrada y ordenada de proveedores disponibles para texto.
    """
    return sorted(
        [p for p in PROVIDER_REGISTRY if p.is_available],
        key=lambda p: p.priority_text,
    )


def get_vision_providers() -> list[ProviderConfig]:
    """Retorna proveedores disponibles para visión, por prioridad.

    Returns:
        Lista filtrada de proveedores con soporte de visión, ordenada.
    """
    return sorted(
        [
            p for p in PROVIDER_REGISTRY
            if p.is_available and p.supports_vision
        ],
        key=lambda p: p.priority_vision,
    )


def validate_env_on_startup() -> dict[str, bool]:
    """Valida qué proveedores están configurados al iniciar.

    Llamar en scripts/run_pipeline.py antes de iniciar el pipeline
    para dar feedback claro al usuario sobre qué APIs funcionarán.

    Returns:
        Dict de nombre_proveedor → tiene_api_key.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="🔑 Configuración de Proveedores LLM")
    table.add_column("Proveedor", style="cyan")
    table.add_column("Variable de Entorno", style="yellow")
    table.add_column("Estado", justify="center")
    table.add_column("Prioridad Texto", justify="center")

    results = {}
    for p in PROVIDER_REGISTRY:
        has_key = bool(p.api_key)
        results[p.name] = has_key
        status_icon = "✅ Configurado" if has_key else "❌ Faltante"
        style = "green" if has_key else "red"
        table.add_row(
            p.name,
            p.api_key_env,
            f"[{style}]{status_icon}[/{style}]",
            str(p.priority_text),
        )

    console.print(table)

    available = sum(results.values())
    console.print(
        f"\n[bold]Proveedores listos: {available}/{len(results)}[/bold]"
    )
    if available == 0:
        console.print(
            "[red bold]⚠️  NINGÚN proveedor configurado. "
            "El pipeline no podrá extraer datos.[/red bold]"
        )

    return results
