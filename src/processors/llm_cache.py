# src/processors/llm_cache.py
"""
Sistema de caché inteligente para respuestas LLM del pipeline Benchmark 360.

Persiste en disco las respuestas LLM con TTL de 24 horas.
Utiliza compresión Gzip para ahorrar espacio en disco y SHA-256 (32 chars)
para evitar colisiones de clave.

Arquitectura: Cache-Aside pattern con SHA-256 como clave.

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path


_CACHE_DIR = Path(".llm_cache")
_DEFAULT_TTL = 86_400   # 24 horas en segundos


class LLMResponseCache:
    """Caché de disco con TTL y compresión para respuestas LLM.

    Utiliza hash SHA-256 del contenido como clave de caché.
    Cada entrada almacena el resultado JSON comprimido con Gzip.

    Attributes:
        cache_dir: Directorio donde se almacenan los archivos de caché.
        ttl_seconds: Tiempo de vida de cada entrada en segundos.
    """

    def __init__(
        self,
        cache_dir: Path = _CACHE_DIR,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        """Inicializa el sistema de caché.

        Args:
            cache_dir: Directorio de almacenamiento. Se crea si no existe.
            ttl_seconds: Segundos de vida de cada entrada (default: 24h).
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits: int = 0
        self._misses: int = 0

    # ── Métodos públicos ──────────────────────────────────────────

    def get(
        self,
        content: str,
        task_type: str,
    ) -> list[dict] | None:
        """Recupera resultado cacheado si existe y no ha expirado.

        Args:
            content: Contenido procesado (HTML o path de imagen).
            task_type: Tipo de tarea ('text' o 'vision').

        Returns:
            Lista de planes extraídos, o None si no hay caché válida.
        """
        key = self._make_key(content, task_type)
        path = self._entry_path(key)

        if not path.exists():
            self._misses += 1
            return None

        try:
            with gzip.open(path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            
            age_seconds = time.time() - data.get("timestamp", 0)

            if age_seconds > self.ttl_seconds:
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

            self._hits += 1
            return data.get("result", [])

        except (json.JSONDecodeError, KeyError, OSError, EOFError):
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

    def set(
        self,
        content: str,
        task_type: str,
        result: list[dict],
        provider_name: str = "unknown",
    ) -> None:
        """Almacena un resultado en caché comprimido con Gzip.

        Args:
            content: Contenido procesado (para generar la clave).
            task_type: Tipo de tarea ('text' o 'vision').
            result: Lista de planes extraídos por el LLM.
            provider_name: Proveedor que generó la respuesta.
        """
        key = self._make_key(content, task_type)
        path = self._entry_path(key)
        entry = {
            "timestamp": time.time(),
            "task_type": task_type,
            "provider": provider_name,
            "content_length": len(content),
            "result": result,
        }
        
        raw_data = json.dumps(entry, ensure_ascii=False).encode("utf-8")
        with gzip.open(path, "wb") as f:
            f.write(raw_data)

    def invalidate(self, content: str, task_type: str) -> bool:
        """Invalida manualmente una entrada del caché.

        Args:
            content: Contenido cuya entrada se desea invalidar.
            task_type: Tipo de tarea de la entrada.

        Returns:
            True si la entrada existía y fue eliminada.
        """
        key = self._make_key(content, task_type)
        path = self._entry_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_expired(self) -> int:
        """Elimina todas las entradas expiradas del caché.

        Returns:
            Número de entradas eliminadas.
        """
        removed = 0
        for path in self.cache_dir.glob("*.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    data = json.loads(f.read().decode("utf-8"))
                age = time.time() - data.get("timestamp", 0)
                if age > self.ttl_seconds:
                    path.unlink()
                    removed += 1
            except (json.JSONDecodeError, OSError, EOFError):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    # ── Propiedades de estadísticas ───────────────────────────────

    @property
    def hit_rate_pct(self) -> float:
        """Porcentaje de aciertos del caché en esta sesión.

        Returns:
            Float entre 0.0 y 100.0.
        """
        total = self._hits + self._misses
        return round(self._hits / total * 100, 2) if total > 0 else 0.0

    @property
    def total_entries(self) -> int:
        """Número total de entradas en el caché.

        Returns:
            Conteo de archivos .gz en el directorio de caché.
        """
        return len(list(self.cache_dir.glob("*.gz")))

    @property
    def estimated_tokens_saved(self) -> int:
        """Estimación de tokens ahorrados por caché en esta sesión.

        Asume ~2,500 tokens de entrada promedio por request evitado.

        Returns:
            Tokens aproximados no enviados a la API.
        """
        return self._hits * 2_500

    def stats(self) -> dict:
        """Retorna diccionario con estadísticas completas del caché.

        Returns:
            Dict con hits, misses, hit_rate_pct, entries y savings.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": self.hit_rate_pct,
            "total_entries": self.total_entries,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "ttl_hours": self.ttl_seconds // 3600,
        }

    # ── Métodos privados ──────────────────────────────────────────

    def _make_key(self, content: str, task_type: str) -> str:
        """Genera clave de caché usando SHA-256 (32 caracteres).

        Args:
            content: Contenido a hashear.
            task_type: Tipo de tarea incluido en el hash.

        Returns:
            String hex de 32 caracteres.
        """
        raw = f"{task_type}::{content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _entry_path(self, key: str) -> Path:
        """Retorna el path del archivo de caché para una clave.

        Args:
            key: Clave SHA-256 truncada.

        Returns:
            Path al archivo .gz de la entrada.
        """
        return self.cache_dir / f"{key}.json.gz"
