# src/scrapers/cookie_handler.py
"""
Cookie Consent Handler — Elimina banners de cookies antes de screenshots.

El banner de cookies con fondo gris/oscuro contamina todos los tiles
y confunde al LLM Vision haciéndole extraer datos incorrectos o nulos.

Estrategia:
    1. Detectar banner por selectores conocidos (lista exhaustiva)
    2. Hacer click en "Aceptar" / "Accept" / "Acepto"
    3. Esperar a que el banner desaparezca del DOM
    4. Si no hay banner → continuar sin error

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from playwright.async_api import Page


# ─────────────────────────────────────────────────────────────────
# Selectores exhaustivos para botones de aceptar cookies
# Cubre los CMP (Consent Management Platforms) más usados en Ecuador
# ─────────────────────────────────────────────────────────────────

_COOKIE_ACCEPT_SELECTORS: list[str] = [
    # Por ID
    "#cookiescript_accept",
    "#accept-cookies",
    "#cookie-accept",
    "#acceptCookies",
    "#btnAcceptCookies",
    "#onetrust-accept-btn-handler",   # OneTrust (muy común)
    "#CybotCookiebotDialogBodyButtonAccept",  # Cookiebot
    # Por clase
    ".cookie-accept",
    ".accept-cookies",
    ".btn-accept-cookies",
    ".cookie-consent-accept",
    "[class*='cookie'][class*='accept']",
    "[class*='accept'][class*='cookie']",
    # Por texto del botón (aria-label / texto visible)
    "button:has-text('Aceptar')",
    "button:has-text('Aceptar todo')",
    "button:has-text('Aceptar todas')",
    "button:has-text('Accept')",
    "button:has-text('Accept All')",
    "button:has-text('Acepto')",
    "button:has-text('De acuerdo')",
    "button:has-text('Entendido')",
    "button:has-text('OK')",
    # Genéricos de último recurso
    "[aria-label*='accept' i]",
    "[aria-label*='aceptar' i]",
    "button[class*='consent']",
]

# Selectores para detectar SI existe un banner (antes de intentar click)
_COOKIE_BANNER_PRESENCE_SELECTORS: list[str] = [
    "#cookiescript_injected",
    "#onetrust-banner-sdk",
    "#CybotCookiebotDialog",
    "[class*='cookie-banner']",
    "[class*='cookie-notice']",
    "[class*='consent-banner']",
    "[id*='cookie']",
    "[class*='gdpr']",
]


class CookieConsentHandler:
    """Detecta y descarta banners de cookies antes de capturar screenshots.

    Previene que el overlay gris de cookie consent contamine los tiles
    de screenshot enviados al LLM Vision.

    Args:
        timeout_ms: Tiempo máximo de espera por selector en ms.
        wait_after_accept_ms: Espera tras aceptar para que el banner
            desaparezca con su animación CSS.

    Example:
        >>> handler = CookieConsentHandler()
        >>> dismissed = await handler.dismiss(page)
        >>> print("Banner eliminado" if dismissed else "Sin banner")
    """

    def __init__(
        self,
        timeout_ms: int = 3000,
        wait_after_accept_ms: int = 800,
    ) -> None:
        self._timeout_ms = timeout_ms
        self._wait_after_ms = wait_after_accept_ms

    async def dismiss(self, page: Page, isp_key: str = "") -> bool:
        """Detecta y hace click en el botón de aceptar cookies.

        Prueba cada selector de la lista hasta encontrar uno clickeable.
        Si no hay banner de cookies, retorna False sin error.

        Args:
            page: Objeto Page de Playwright con la página cargada.
            isp_key: Clave del ISP para logging contextual.

        Returns:
            True si se encontró y descartó un banner.
            False si no se detectó ningún banner (sitio sin cookies popup).
        """
        # Verificar si existe un banner antes de intentar click
        banner_present = await self._detect_banner_presence(page)

        if not banner_present:
            logger.debug(
                "[{}] Sin banner de cookies detectado", isp_key
            )
            return False

        logger.info(
            "[{}] 🍪 Banner de cookies detectado → intentando dismiss",
            isp_key,
        )

        # Probar cada selector de aceptar
        for selector in _COOKIE_ACCEPT_SELECTORS:
            try:
                element = await page.wait_for_selector(
                    selector,
                    timeout=self._timeout_ms,
                    state="visible",
                )
                if element:
                    await element.click()
                    await asyncio.sleep(self._wait_after_ms / 1000)
                    logger.info(
                        "[{}] ✅ Cookie banner dismissado con selector: {}",
                        isp_key,
                        selector,
                    )
                    return True

            except Exception:
                continue  # Este selector no funcionó → probar el siguiente

        # Último recurso: tecla Escape
        try:
            await page.keyboard.press("Escape")
            await asyncio.sleep(0.3)
            logger.warning(
                "[{}] ⚠️  Cookie banner: ningún selector funcionó, "
                "intenté Escape como último recurso",
                isp_key,
            )
        except Exception:
            pass

        return False

    async def _detect_banner_presence(self, page: Page) -> bool:
        """Verifica si hay algún banner de cookies visible en la página.

        Args:
            page: Página Playwright cargada.

        Returns:
            True si se detecta un elemento de banner de cookies.
        """
        for selector in _COOKIE_BANNER_PRESENCE_SELECTORS:
            try:
                element = await page.query_selector(selector)
                if element:
                    is_visible = await element.is_visible()
                    if is_visible:
                        return True
            except Exception:
                continue
        return False

    async def screenshot_ready(
        self,
        page: Page,
        isp_key: str = "",
    ) -> None:
        """Prepara la página para screenshots limpios.

        Ejecuta en orden:
          1. Dismiss cookie banner
          2. Ocultar overlays residuales via CSS inject
          3. Scroll al top para empezar desde arriba
          4. Esperar a que lazy-loaded images se carguen

        Args:
            page: Página Playwright con el sitio cargado.
            isp_key: Clave del ISP para logging.
        """
        # 1. Dismiss banner
        await self.dismiss(page, isp_key)

        # 2. Inyectar CSS para ocultar overlays residuales
        await page.add_style_tag(content="""
            /* Ocultar modales de cookies y overlays residuales */
            [id*='cookie'],
            [class*='cookie-banner'],
            [class*='cookie-notice'],
            [class*='consent-banner'],
            [class*='gdpr'],
            [id*='onetrust'],
            .cookie-overlay,
            body > div[class*='overlay'] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
            }
            /* Restaurar scroll del body si el modal lo bloqueó */
            body, html {
                overflow: auto !important;
            }
        """)

        # 3. Scroll al top
        await page.evaluate("window.scrollTo(0, 0)")

        # 4. Esperar lazy images
        await asyncio.sleep(1.0)

        logger.info(
            "[{}] 📸 Página lista para screenshots limpios", isp_key
        )
