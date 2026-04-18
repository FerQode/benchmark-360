# src/scrapers/base_scraper.py
"""
Dynamic ISP Scraper — Scraper unificado para todos los ISPs.

Reemplaza a todos los scrapers hijos individuales (ClaroScraper,
XtrimScraper, etc.). Las particularidades por ISP se configuran
en ISPStrategy (isp_url_strategy.py), no en subclases.

Flujo por ISP:
    1. Cargar ISPStrategy → lista de páginas por tipo
    2. Para PLAN pages:
       a. Playwright renderiza la página
       b. CookieConsentHandler descarta el banner
       c. HTML scraping → text_content
       d. _capture_tiles() → tiles PNG en disco para Vision
    3. Para TC pages:
       a. TCHTMLScraper (solo httpx+BS4, sin Playwright)
       b. Resultado → ScrapedPage.terminos_condiciones_raw
    4. Retornar ScrapedPage unificado

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.scrapers.cookie_handler import CookieConsentHandler
from src.scrapers.isp_url_strategy import (
    ISPStrategy,
    PageType,
    get_strategy,
)
from src.scrapers.tc_scraper import TCHTMLScraper


# ─────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_TILE_HEIGHT: int = int(os.getenv("VISION_TILE_HEIGHT", "1200"))
_TILE_OVERLAP: int = int(os.getenv("VISION_TILE_OVERLAP", "300"))
_MAX_TILES: int = int(os.getenv("VISION_MAX_TILES", "5"))
_PAGE_LOAD_WAIT_MS: int = 2500  # Espera post-navegación para JS
_SCROLL_WAIT_S: float = 0.5    # Espera entre scrolls para lazy-load

# Selectores CSS para detectar contenedores de planes
# Ordenados de más específico a más genérico
_PLAN_CONTAINER_SELECTORS: list[str] = [
    # Específicos ISPs Ecuador (actualizar conforme se descubren)
    ".plan-card",
    ".plan-box",
    "[class*='plan-item']",
    "[class*='pricing-card']",
    # Semánticos genéricos
    "[class*='plan']",
    "[class*='precio']",
    "[class*='paquete']",
    "[class*='oferta']",
    "[class*='price']",
    ".card",
]

# Tags HTML informativos para extracción de texto
_TEXT_TAGS: list[str] = [
    "h1", "h2", "h3", "h4", "h5",
    "p", "li", "td", "th", "span",
    "div", "article", "section",
]

# Tags a eliminar antes de extraer texto
_NOISE_TAGS: list[str] = [
    "script", "style", "nav", "footer",
    "header", "aside", "form", "iframe",
    "noscript", "svg", "meta",
]


# ─────────────────────────────────────────────────────────────────
# ScrapedPage — DTO de resultado
# ─────────────────────────────────────────────────────────────────


@dataclass
class ScrapedPage:
    """Resultado completo del scraping de un ISP.

    Contiene tanto el contenido de planes (texto + tiles de imagen)
    como el texto de T&C extraído por separado.

    Attributes:
        isp_key: Clave identificadora del ISP.
        marca: Nombre comercial del ISP.
        text_content: Texto extraído de las páginas de PLANES.
            NO incluye T&C (mantenidos separados por diseño).
        screenshots: Lista de rutas a tiles PNG capturados.
        terminos_condiciones_raw: Texto completo de T&C extraído
            directamente por TCHTMLScraper desde la URL de T&C.
            Se mapea 1:1 al campo terminos_condiciones del schema.
        urls_scraped: URLs de planes efectivamente procesadas.
        tc_url_scraped: URL de T&C procesada (para auditoría).
        html_raw: HTML completo de la primera página de planes.
        error: Error fatal si el scraping falló por completo.
    """

    isp_key: str
    marca: str
    text_content: str = ""
    screenshots: list[Path] = field(default_factory=list)
    terminos_condiciones_raw: str = ""
    urls_scraped: list[str] = field(default_factory=list)
    tc_url_scraped: str = ""
    html_raw: str = ""
    error: str | None = None

    @property
    def has_screenshots(self) -> bool:
        """True si se capturaron tiles de screenshot."""
        return len(self.screenshots) > 0

    @property
    def has_plans_text(self) -> bool:
        """True si se extrajo texto suficiente de páginas de planes."""
        return len(self.text_content) > 200

    @property
    def has_tc(self) -> bool:
        """True si se extrajo texto de T&C."""
        return len(self.terminos_condiciones_raw) > 100


# ─────────────────────────────────────────────────────────────────
# BaseISPScraper — Scraper dinámico unificado
# ─────────────────────────────────────────────────────────────────


class BaseISPScraper:
    """Scraper dinámico unificado para todos los ISPs.

    Reemplaza a todas las subclases individuales. El comportamiento
    por ISP se configura en ISPStrategy, no en herencia.

    Proceso de scraping:
        1. Cargar ISPStrategy para el ISP dado
        2. Para cada plan_page: Playwright + Cookie dismiss + tiles
        3. Para cada tc_page: TCHTMLScraper (solo HTTP, sin browser)
        4. Consolidar en ScrapedPage y retornar

    Args:
        isp_key: Clave del ISP (debe existir en ISP_STRATEGIES).
        data_raw_path: Directorio raíz para guardar HTML y tiles.
        robots_checker: Verificador de robots.txt.

    Example:
        >>> scraper = BaseISPScraper("xtrim", Path("data/raw"))
        >>> page = await scraper.scrape()
        >>> print(f"Texto: {len(page.text_content)} chars")
        >>> print(f"Tiles: {len(page.screenshots)}")
        >>> print(f"T&C: {len(page.terminos_condiciones_raw)} chars")
    """

    # Browser compartido entre instancias (singleton por proceso)
    _browser: Browser | None = None
    _playwright_instance = None

    def __init__(
        self,
        isp_key: str,
        data_raw_path: Path = Path("data/raw"),
        robots_checker=None,
    ) -> None:
        """Inicializa el scraper dinámico para un ISP.

        Args:
            isp_key: Clave del ISP registrada en ISP_STRATEGIES.
            data_raw_path: Directorio para guardar datos crudos.
            robots_checker: Instancia de RobotsChecker (opcional).

        Raises:
            KeyError: Si isp_key no existe en ISP_STRATEGIES.
        """
        self.isp_key = isp_key
        self._data_raw = data_raw_path
        self._robots = robots_checker
        self._strategy: ISPStrategy = get_strategy(isp_key)
        self._cookie_handler = CookieConsentHandler()
        self._tc_scraper = TCHTMLScraper()

        # Directorio de salida para este ISP
        self._isp_dir = data_raw_path / isp_key
        self._isp_dir.mkdir(parents=True, exist_ok=True)
        (self._isp_dir / "screenshots").mkdir(exist_ok=True)

        logger.info(
            "[{}] Scraper init — {} plan pages, {} T&C pages",
            isp_key,
            len(self._strategy.plan_pages),
            len(self._strategy.tc_pages),
        )

    # ── API pública ────────────────────────────────────────────────

    async def scrape(self) -> ScrapedPage:
        """Ejecuta el scraping completo del ISP.

        Procesa en paralelo:
          - Plan pages → Playwright + Cookie handler + tiles
          - TC pages   → HTTPx directo (sin browser)

        Returns:
            ScrapedPage con todo el contenido extraído.
        """
        result = ScrapedPage(
            isp_key=self.isp_key,
            marca=self._strategy.marca,
        )

        try:
            # ── Paso 1: T&C (httpx puro — no necesita browser) ────
            # Se ejecuta PRIMERO: es rápido y no necesita Playwright
            if self._strategy.tc_pages:
                tc_page = self._strategy.tc_pages[0]  # Primera URL de T&C
                await self._scrape_tc_page(
                    result=result,
                    tc_url=tc_page.url,
                )

            # ── Paso 2: Plan pages (Playwright + Cookie + tiles) ───
            if self._strategy.plan_pages:
                await self._scrape_plan_pages(result=result)

            # ── Validación mínima ──────────────────────────────────
            if not result.has_plans_text and not result.has_screenshots:
                result.error = (
                    "Sin contenido útil: ni texto de planes "
                    "ni tiles de screenshot disponibles."
                )
                logger.warning("[{}] ⚠️  {}", self.isp_key, result.error)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[{}] ❌ Scraping failed: {}", self.isp_key, exc
            )

        return result

    # ── Plan pages (Playwright) ────────────────────────────────────

    async def _scrape_plan_pages(self, result: ScrapedPage) -> None:
        """Scrapeaa todas las páginas de planes del ISP con Playwright.

        Para cada plan_page en la estrategia:
          1. Navegar con Playwright
          2. Dismiss cookie banner
          3. Extraer texto HTML limpio
          4. Capturar tiles para Vision

        Args:
            result: ScrapedPage a enriquecer con el contenido.
        """
        all_texts: list[str] = []
        all_tiles: list[Path] = []

        browser = await self._get_browser()

        for plan_page in self._strategy.plan_pages:
            # Verificar robots.txt
            if self._robots and not self._robots.can_fetch(plan_page.url):
                logger.warning(
                    "[{}] robots.txt bloquea: {}", self.isp_key, plan_page.url
                )
                continue

            logger.info(
                "[{}] 🌐 Scraping plan page: {}", self.isp_key, plan_page.url
            )

            context: BrowserContext = await browser.new_context(
                user_agent=_USER_AGENT,
                viewport={"width": 1920, "height": _TILE_HEIGHT},
                locale="es-EC",
            )
            page: Page = await context.new_page()

            try:
                # Navegar y esperar carga completa
                await page.goto(
                    plan_page.url,
                    wait_until="networkidle",
                    timeout=30_000,
                )
                await page.wait_for_timeout(_PAGE_LOAD_WAIT_MS)

                # ── MEJORA 1: Cookie handler ───────────────────────
                await self._cookie_handler.screenshot_ready(
                    page=page,
                    isp_key=self.isp_key,
                )

                # ── Extracción de texto HTML ───────────────────────
                html = await page.content()
                page_text = self._extract_text_from_html(html)
                all_texts.append(page_text)

                # Guardar HTML crudo de la primera página
                if not result.html_raw:
                    result.html_raw = html
                    html_path = self._isp_dir / "page.html"
                    html_path.write_text(html, encoding="utf-8")

                # ── Captura de tiles para Vision LLM ──────────────
                tiles = await self._capture_tiles(page=page)
                all_tiles.extend(tiles)

                result.urls_scraped.append(plan_page.url)
                logger.info(
                    "[{}] ✅ Plan page OK: {} chars, {} tiles",
                    self.isp_key,
                    len(page_text),
                    len(tiles),
                )

                # Delay respetuoso entre páginas del mismo dominio
                await asyncio.sleep(2.0)

            except Exception as exc:
                logger.error(
                    "[{}] Error en plan page {}: {}",
                    self.isp_key,
                    plan_page.url,
                    exc,
                )
            finally:
                await context.close()

        result.text_content = "\n\n---PAGINA---\n\n".join(all_texts)
        result.screenshots = all_tiles

    # ── T&C page (httpx o Playwright según estrategia) ────────────

    async def _scrape_tc_page(
        self,
        result: ScrapedPage,
        tc_url: str,
    ) -> None:
        """Extrae T&C — httpx primero, Playwright como fallback.

        Determina el método según `use_playwright` en ISPPageTarget:
        - use_playwright=False → httpx + BeautifulSoup (rápido, sin browser)
        - use_playwright=True  → Playwright (para sitios con JS/CORS strict)

        El texto extraído va al campo terminos_condiciones_raw del
        ScrapedPage. NO se mezcla con el texto de planes.

        Args:
            result: ScrapedPage a enriquecer.
            tc_url: URL de la página de T&C.
        """
        # Determinar si esta URL requiere Playwright
        tc_page_target = next(
            (p for p in self._strategy.tc_pages if p.url == tc_url),
            None,
        )
        needs_playwright = (
            tc_page_target.use_playwright if tc_page_target else False
        )

        # Intentar httpx primero (siempre, más rápido)
        tc_result = await self._tc_scraper.fetch(
            isp_key=self.isp_key,
            tc_url=tc_url,
        )

        if tc_result.success:
            result.terminos_condiciones_raw = tc_result.raw_text
            result.tc_url_scraped = tc_url
            tc_path = self._isp_dir / "terminos_condiciones.txt"
            tc_path.write_text(tc_result.raw_text, encoding="utf-8")
            logger.info(
                "[{}] 📄 T&C extraído: {} chars (via HTTP directo)",
                self.isp_key,
                tc_result.char_count,
            )
            return

        # ── Fallback: Playwright si httpx falló ───────────────────
        if needs_playwright or tc_result.error:
            logger.info(
                "[{}] T&C httpx falló → intentando con Playwright: {}",
                self.isp_key,
                tc_url,
            )
            try:
                browser = await self._get_browser()
                context = await browser.new_context(
                    user_agent=_USER_AGENT,
                    locale="es-EC",
                )
                page = await context.new_page()
                try:
                    await page.goto(tc_url, wait_until="domcontentloaded", timeout=20_000)
                    await page.wait_for_timeout(1500)
                    html = await page.content()
                    tc_text = self._extract_text_from_html(html)
                    if len(tc_text) > 100:
                        result.terminos_condiciones_raw = tc_text
                        result.tc_url_scraped = tc_url
                        tc_path = self._isp_dir / "terminos_condiciones.txt"
                        tc_path.write_text(tc_text, encoding="utf-8")
                        logger.info(
                            "[{}] 📄 T&C extraído via Playwright: {} chars",
                            self.isp_key,
                            len(tc_text),
                        )
                    else:
                        logger.warning(
                            "[{}] ⚠️  T&C Playwright: texto insuficiente ({} chars)",
                            self.isp_key,
                            len(tc_text),
                        )
                finally:
                    await context.close()
            except Exception as exc:
                logger.warning(
                    "[{}] ⚠️  T&C Playwright también falló: {}",
                    self.isp_key,
                    exc,
                )
        else:
            logger.warning(
                "[{}] ⚠️  T&C no disponible: {}",
                self.isp_key,
                tc_result.error,
            )

    # ── Tiling de screenshots ──────────────────────────────────────

    async def _capture_tiles(self, page: Page) -> list[Path]:
        """Captura tiles PNG para Vision LLM y los guarda en disco.

        Intenta DOM-sectioning primero usando selectores específicos
        del ISP (definidos en ISPStrategy.specific_plan_selectors)
        y luego los genéricos. Fallback a viewport-scroll.

        Los tiles se guardan en:
            data/raw/{isp_key}/screenshots/tile_dom_XX.png
            data/raw/{isp_key}/screenshots/tile_scroll_XX.png

        Args:
            page: Objeto Page de Playwright con la página cargada.

        Returns:
            Lista de rutas PNG de los tiles capturados.
        """
        screenshot_dir = self._isp_dir / "screenshots"
        tiles: list[Path] = []

        # Combinar selectores específicos del ISP + genéricos
        all_selectors = (
            self._strategy.specific_plan_selectors
            + _PLAN_CONTAINER_SELECTORS
        )

        # ── Intento 1: DOM-based sectioning ───────────────────────
        for selector in all_selectors:
            try:
                elements = await page.query_selector_all(selector)

                # Filtrar: solo elementos con tamaño razonable de plan card
                valid_elements = []
                for el in elements:
                    bbox = await el.bounding_box()
                    if (
                        bbox
                        and bbox["height"] > 150
                        and bbox["width"] > 200
                    ):
                        valid_elements.append(el)

                if len(valid_elements) >= 2:
                    logger.info(
                        "[{}] DOM selector '{}' → {} plan cards",
                        self.isp_key,
                        selector,
                        len(valid_elements),
                    )

                    for idx, element in enumerate(
                        valid_elements[:_MAX_TILES]
                    ):
                        tile_path = (
                            screenshot_dir / f"tile_dom_{idx:02d}.png"
                        )
                        await element.screenshot(path=str(tile_path))
                        tiles.append(tile_path)

                    logger.info(
                        "[{}] ✅ DOM tiling: {} tiles capturados",
                        self.isp_key,
                        len(tiles),
                    )
                    return tiles  # DOM exitoso → no necesitamos scroll

            except Exception as exc:
                logger.debug(
                    "[{}] Selector '{}' falló: {}",
                    self.isp_key,
                    selector,
                    exc,
                )
                continue

        # ── Intento 2: Viewport scroll con overlap ─────────────────
        logger.info(
            "[{}] DOM sin resultados → viewport scroll tiling",
            self.isp_key,
        )

        page_height: int = await page.evaluate(
            "() => document.body.scrollHeight"
        )
        await page.set_viewport_size(
            {"width": 1920, "height": _TILE_HEIGHT}
        )

        step = _TILE_HEIGHT - _TILE_OVERLAP
        y_positions = list(range(0, page_height, step))

        for idx, y_start in enumerate(y_positions[:_MAX_TILES]):
            await page.evaluate(f"window.scrollTo(0, {y_start})")
            await asyncio.sleep(_SCROLL_WAIT_S)

            tile_path = screenshot_dir / f"tile_scroll_{idx:02d}.png"
            await page.screenshot(
                path=str(tile_path),
                full_page=False,
                type="png",
            )
            tiles.append(tile_path)

            logger.debug(
                "[{}] Scroll tile {}: y={}",
                self.isp_key,
                idx,
                y_start,
            )

        # Reset al top
        await page.evaluate("window.scrollTo(0, 0)")
        logger.info(
            "[{}] ✅ Scroll tiling: {} tiles capturados",
            self.isp_key,
            len(tiles),
        )
        return tiles

    # ── Extracción de texto HTML ───────────────────────────────────

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        """Extrae texto informativo del HTML eliminando ruido.

        Args:
            html: HTML completo de la página.

        Returns:
            Texto limpio, apto para enviar al LLM de texto.
        """
        soup = BeautifulSoup(html, "lxml")

        # Eliminar tags de ruido
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # Extraer texto de tags informativos
        texts: list[str] = []
        for tag in soup.find_all(_TEXT_TAGS):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 15:  # Filtrar textos muy cortos (menús, etc.)
                texts.append(text)

        raw_text = "\n".join(texts)

        # Limpieza de whitespace
        raw_text = re.sub(r'\r\n|\r', '\n', raw_text)
        raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
        raw_text = re.sub(r' {2,}', ' ', raw_text)

        return raw_text.strip()

    # ── Browser singleton ──────────────────────────────────────────

    @classmethod
    async def _get_browser(cls) -> Browser:
        """Obtiene o crea el browser Playwright compartido.

        Singleton por proceso para reutilizar entre ISPs
        y reducir overhead de inicialización.

        Returns:
            Instancia de Browser lista para usar.
        """
        if cls._browser is None or not cls._browser.is_connected():
            cls._playwright_instance = await async_playwright().start()
            cls._browser = await cls._playwright_instance.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--lang=es-EC",
                ],
            )
            logger.info("🌐 Browser Chromium iniciado (singleton)")
        return cls._browser

    @classmethod
    async def close_browser(cls) -> None:
        """Cierra el browser y la instancia de Playwright.

        Debe llamarse al finalizar TODOS los ISPs en el pipeline.
        """
        if cls._browser:
            await cls._browser.close()
            cls._browser = None
        if cls._playwright_instance:
            await cls._playwright_instance.stop()
            cls._playwright_instance = None
        logger.info("🔒 Browser cerrado correctamente")
