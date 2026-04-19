# src/scrapers/tc_scraper.py
"""Extractor de Términos y Condiciones con estrategia de fallback en cascada.

Inspirado en el principio de Graceful Degradation: siempre hay
un resultado, incluso si la fuente primaria falla.

Estrategias en orden:
  1. URL directa de T&C (si está configurada en ISPConfig).
  2. Búsqueda de links legales en el footer de la página de planes.
  3. Mensaje de no disponible (nunca retorna None).

Uso:
    from src.scrapers.tc_scraper import extract_terms_and_conditions
    text = await extract_terms_and_conditions(
        terms_url="https://netlife.ec/terminos",
        plans_url="https://netlife.ec/",
        isp_key="netlife",
    )
"""

from __future__ import annotations

from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.utils.logger import logger

# Keywords para buscar links de T&C en el footer
_TC_KEYWORDS = [
    "términos",
    "condiciones",
    "terminos",
    "contrato",
    "legal",
    "terms",
    "conditions",
    "privacidad",
    "política",
]

# Máximo de caracteres a enviar al LLM (aprox 2000 tokens)
_MAX_TC_CHARS = 8_000


async def extract_terms_and_conditions(
    terms_url: str | None,
    plans_url: str,
    isp_key: str,
) -> str:
    """Extrae T&C con múltiples niveles de búsqueda y fallback.

    Args:
        terms_url: URL directa de T&C (None si no está disponible).
        plans_url: URL principal del ISP (usada para buscar links en footer).
        isp_key: Identificador del ISP para contexto de logs.

    Returns:
        Texto plano de T&C extraído, o mensaje de no disponible.
    """
    log = logger.bind(isp=isp_key, phase="tc_scraper")

    # ── Nivel 1: URL directa ──────────────────────────────────────
    if terms_url:
        text = await _fetch_and_clean(terms_url, isp_key)
        if text:
            log.info(f"T&C extraídos directamente de {terms_url} ({len(text)} chars)")
            return text

    # ── Nivel 2: Buscar link legal en el footer de planes ─────────
    log.debug(f"URL directa falló — buscando links legales en {plans_url}")
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(plans_url, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")

            for link in soup.find_all("a", href=True):
                link_text = link.get_text(strip=True).lower()
                if any(kw in link_text for kw in _TC_KEYWORDS):
                    href = link["href"]
                    target_url = href if href.startswith("http") else urljoin(plans_url, href)

                    log.info(f"Link legal encontrado: {target_url}")
                    text = await _fetch_and_clean(target_url, isp_key)
                    if text:
                        return text
    except Exception as exc:
        log.warning(f"Búsqueda de links legales falló: {exc}")

    # ── Nivel 3: Fallback de última instancia ─────────────────────
    log.warning("T&C no disponibles en ninguna fuente")
    return "Términos y condiciones no disponibles o no detectados en el sitio web."


async def _fetch_and_clean(url: str, isp_key: str) -> str | None:
    """Descarga una URL y extrae solo el texto relevante.

    Args:
        url: URL a descargar.
        isp_key: Identificador del ISP para logs.

    Returns:
        Texto limpio con máximo _MAX_TC_CHARS caracteres, o None si falló.
    """
    log = logger.bind(isp=isp_key, phase="tc_fetch")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Benchmark360/1.0)"}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, timeout=15, headers=headers)
            if resp.status_code != 200:
                log.debug(f"HTTP {resp.status_code} para {url}")
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            # Eliminar ruido: scripts, estilos, navegación
            for tag in soup(["script", "style", "nav", "header", "footer", "iframe", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            if len(text) < 200:  # Muy corto = probablemente vacío
                return None

            return text[:_MAX_TC_CHARS]

    except Exception as exc:
        log.debug(f"Error descargando {url}: {exc}")
        return None


# ── Backward Compatibility ────────────────────────────────────────
# base_scraper.py importa TCHTMLScraper por clase. Esta clase envuelve
# la nueva función extract_terms_and_conditions() sin romper nada.

class TCHTMLScraper:
    """Wrapper de compatibilidad para código que usa la interfaz de clase.

    Uso legacy:
        scraper = TCHTMLScraper(terms_url="...", plans_url="...", isp_key="netlife")
        text = await scraper.extract()
    """

    def __init__(
        self,
        terms_url: str | None,
        plans_url: str,
        isp_key: str,
    ) -> None:
        self._terms_url = terms_url
        self._plans_url = plans_url
        self._isp_key = isp_key

    async def extract(self) -> str:
        """Extrae T&C. Delegado a extract_terms_and_conditions()."""
        return await extract_terms_and_conditions(
            terms_url=self._terms_url,
            plans_url=self._plans_url,
            isp_key=self._isp_key,
        )

    # Alias común
    async def scrape(self) -> str:
        return await self.extract()
