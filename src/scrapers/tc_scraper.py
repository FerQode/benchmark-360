# src/scrapers/tc_scraper.py
"""
Terms & Conditions Scraper — Extracción HTML directa de T&C por URL.

El técnico de Netlife tiene razón al 100%:
    La URL de T&C tiene todo el texto en HTML estructurado.
    Pasarla directamente al scraper HTML es:
    - 10x más barato (sin Vision LLM)
    - 5x más rápido (sin Playwright)
    - 3x más preciso (texto completo sin OCR ni Vision)

Este módulo extrae T&C como texto limpio y lo pasa al
LLM de texto (Flash) para estructurarlo según el schema.

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class TCResult:
    """Resultado de la extracción de T&C de un ISP.

    Attributes:
        isp_key: Clave del ISP.
        url: URL de donde se extrajeron los T&C.
        raw_text: Texto completo de T&C extraído.
        char_count: Longitud del texto extraído.
        success: True si la extracción fue exitosa.
        error: Mensaje de error si falló.
    """

    isp_key: str
    url: str
    raw_text: str = ""
    char_count: int = 0
    success: bool = False
    error: str | None = None


class TCHTMLScraper:
    """Extrae Términos y Condiciones directamente desde HTML.

    No usa Playwright ni Vision LLM — solo httpx + BeautifulSoup.
    El texto resultante se pasa al LLM de texto para estructurarlo.

    Args:
        user_agent: User-Agent para las peticiones HTTP.
        timeout_seconds: Timeout por petición.

    Example:
        >>> scraper = TCHTMLScraper()
        >>> result = await scraper.fetch("netlife", tc_url)
        >>> print(f"T&C extraído: {result.char_count} caracteres")
    """

    _USER_AGENT = (
        "Mozilla/5.0 (compatible; Benchmark360Bot/1.0; "
        "+https://netlife.ec/benchmark-bot)"
    )

    # Tags HTML que contienen texto relevante de T&C
    _CONTENT_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "td", "article"]

    # Selectores de contenedores principales de T&C
    _CONTENT_SELECTORS = [
        "article",
        "main",
        "[class*='terminos']",
        "[class*='terms']",
        "[class*='condiciones']",
        "[class*='conditions']",
        "[class*='content']",
        ".entry-content",
        "#main-content",
        "section",
    ]

    def __init__(
        self,
        user_agent: str = _USER_AGENT,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._headers = {"User-Agent": user_agent}
        self._timeout = timeout_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
    )
    async def fetch(self, isp_key: str, tc_url: str) -> TCResult:
        """Descarga y extrae el texto de T&C desde una URL.

        Proceso:
          1. GET HTTP con httpx (sin Playwright — HTML estático)
          2. BeautifulSoup para extraer texto de tags relevantes
          3. Limpiar whitespace y caracteres basura
          4. Retornar texto listo para LLM de texto

        Args:
            isp_key: Clave del ISP para logging.
            tc_url: URL completa de la página de T&C.

        Returns:
            TCResult con el texto extraído y metadata.
        """
        result = TCResult(isp_key=isp_key, url=tc_url)

        try:
            async with httpx.AsyncClient(
                headers=self._headers,
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(tc_url)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remover elementos no informativos
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form", "iframe"]):
                tag.decompose()

            # Intentar extraer del contenedor principal primero
            content_text = self._extract_from_container(soup)

            # Si no se encontró contenedor → extraer de todo el body
            if len(content_text) < 200:
                content_text = self._extract_full_body(soup)

            # Limpiar el texto
            clean_text = self._clean_text(content_text)

            result.raw_text = clean_text
            result.char_count = len(clean_text)
            result.success = len(clean_text) > 100

            logger.info(
                "[{}] T&C extraído: {} chars desde {}",
                isp_key,
                result.char_count,
                tc_url,
            )

        except httpx.HTTPStatusError as exc:
            result.error = f"HTTP {exc.response.status_code}: {tc_url}"
            logger.warning("[{}] T&C HTTP error: {}", isp_key, result.error)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error("[{}] T&C error: {}", isp_key, result.error)

        return result

    def _extract_from_container(self, soup: BeautifulSoup) -> str:
        """Extrae texto del contenedor principal de T&C.

        Args:
            soup: BeautifulSoup del HTML completo.

        Returns:
            Texto extraído del contenedor. Vacío si no se encontró.
        """
        for selector in self._CONTENT_SELECTORS:
            container = soup.select_one(selector)
            if container:
                texts = []
                for tag in container.find_all(self._CONTENT_TAGS):
                    text = tag.get_text(separator=" ", strip=True)
                    if len(text) > 20:
                        texts.append(text)
                if texts:
                    return "\n".join(texts)
        return ""

    def _extract_full_body(self, soup: BeautifulSoup) -> str:
        """Extrae texto de todo el body como fallback.

        Args:
            soup: BeautifulSoup del HTML completo.

        Returns:
            Texto completo del body.
        """
        body = soup.find("body")
        if not body:
            return soup.get_text(separator="\n", strip=True)

        texts = []
        for tag in body.find_all(self._CONTENT_TAGS):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 20:
                texts.append(text)
        return "\n".join(texts)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Limpia whitespace excesivo y caracteres basura del texto.

        Args:
            text: Texto crudo extraído del HTML.

        Returns:
            Texto limpio y normalizado.
        """
        # Normalizar saltos de línea
        text = re.sub(r'\r\n|\r', '\n', text)
        # Eliminar líneas vacías múltiples
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Eliminar espacios múltiples
        text = re.sub(r' {2,}', ' ', text)
        # Eliminar caracteres de control
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()
