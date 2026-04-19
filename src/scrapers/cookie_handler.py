# src/scrapers/cookie_handler.py
"""Manejo robusto de banners de cookies — Patrón Dismiss-and-Verify.

Inspirado en la resiliencia de Netflix: ante sistemas impredecibles
(banners de cookies que cambian constantemente), usar múltiples
estrategias en cascada garantiza que el scraping no se bloquee.

Uso:
    from src.scrapers.cookie_handler import dismiss_cookies
    dismissed = await dismiss_cookies(page)
    if dismissed:
        logger.info("Banner de cookies cerrado")
"""

from __future__ import annotations

from playwright.async_api import Page

from src.utils.logger import logger

# Selectores en orden de prioridad (español primero, luego inglés genérico)
COOKIE_DISMISS_SELECTORS: list[str] = [
    "button:has-text('Aceptar')",
    "button:has-text('Acepto')",
    "button:has-text('Entendido')",
    "button:has-text('Continuar')",
    "button:has-text('Accept')",
    "button:has-text('Agree')",
    "button:has-text('OK')",
    "[id*='cookie'] button",
    "[class*='cookie'] button",
    "[id*='consent'] button",
    "[class*='consent'] button",
    "[aria-label*='cookie']",
    "[aria-label*='consent']",
    "#accept-cookies",
    ".accept-cookies",
    ".cookie-accept",
    "#cookie-accept",
]


async def dismiss_cookies(page: Page, timeout_ms: int = 4_000) -> bool:
    """Intenta cerrar banners de consentimiento con múltiples estrategias.

    Estrategias en orden:
    1. Iterar selectores CSS predefinidos e intentar click.
    2. Fallback: eliminar overlays de cookies por DOM manipulation (JS).

    Args:
        page: Instancia activa de la página Playwright.
        timeout_ms: Tiempo máximo por intento de selector.

    Returns:
        True si se detectó y cerró algún banner, False en caso contrario.
    """
    log = logger.bind(isp="cookie_handler", phase="cookie_dismiss")

    # Estrategia 1: Selectores CSS con timeout rápido
    for selector in COOKIE_DISMISS_SELECTORS:
        try:
            button = page.locator(selector).first
            # Usar 500ms para no bloquear — la mayoría de banners son rápidos
            if await button.is_visible(timeout=500):
                await button.click()
                log.info(f"Cookie banner cerrado con: {selector}")
                # Espera breve para que el banner desaparezca del DOM
                await page.wait_for_timeout(800)
                return True
        except Exception:
            continue  # Selector no encontrado — intentar el siguiente

    # Estrategia 2: DOM Manipulation via JavaScript (último recurso)
    try:
        removed_count: int = await page.evaluate("""
            () => {
                const selectors = [
                    '[class*="cookie"]', '[id*="cookie"]',
                    '[class*="consent"]', '[id*="consent"]',
                    '[class*="banner"]', '[class*="overlay"]',
                    '[class*="modal"]'
                ];
                let count = 0;
                selectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => {
                        // Solo eliminar elementos pequeños (banners, no el body)
                        if (el.innerText && el.innerText.length < 800) {
                            el.remove();
                            count++;
                        }
                    });
                });
                return count;
            }
        """)
        if removed_count > 0:
            log.debug(f"Fallback JS: {removed_count} elemento(s) eliminados del DOM")
    except Exception as exc:
        log.debug(f"Fallback JS no aplicable: {exc}")

    return False

# ── Backward Compatibility ─────────────────────────────────────
# base_scraper.py importa CookieConsentHandler por clase. Esta clase
# envuelve la nueva función dismiss_cookies() sin romper nada.

class CookieConsentHandler:
    """Manejo de banners de cookies para preparar la captura."""

    def __init__(self, timeout_ms: int = 4_000) -> None:
        self._timeout_ms = timeout_ms

    async def screenshot_ready(self, page: Page, isp_key: str) -> bool:
        """Intenta cerrar el banner antes de tomar screenshot."""
        return await dismiss_cookies(page, self._timeout_ms)
