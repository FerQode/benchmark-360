# src/scrapers/isp_registry.py
"""Registry centralizado de ISPs — fuente única de verdad de configuración.

Inspirado en el registry pattern de Airbnb: cada proveedor se registra
con sus capacidades completas. Para agregar un ISP nuevo, solo se agrega
una entrada al dict ISP_REGISTRY — cero código nuevo.

Uso:
    from src.scrapers.isp_registry import get_isp_config, get_all_isps
    config = get_isp_config("netlife")
    print(config.legal_name)  # "MEGADATOS S.A."
    print(config.brand)       # "Netlife"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScrapingMode(str, Enum):
    """Modo de extracción de datos web.

    Attributes:
        PLAYWRIGHT: Renderizado JS completo (SPAs, lazy-loading).
        HTTPX: Petición HTTP directa sin JS (páginas estáticas).
        HYBRID: Combina ambos según el tipo de contenido.
    """

    PLAYWRIGHT = "playwright"
    HTTPX = "httpx"
    HYBRID = "hybrid"


@dataclass
class ISPConfig:
    """Configuración integral de un ISP para el pipeline de scraping.

    Esta clase reemplaza tanto a ISPStrategy como a CompanyInfo,
    consolidando toda la metadata en un único objeto tipado.

    Attributes:
        brand: Nombre comercial del ISP (ej: "Netlife").
        legal_name: Razón social (Superintendencia de Compañías).
        plans_url: URL principal para extraer planes y precios.
        terms_url: URL directa de Términos y Condiciones (opcional).
        scraping_mode: Modo de extracción por defecto.
        requires_cookie_dismiss: Si True, ejecutar dismiss_cookies() primero.
        plan_selectors: Selectores CSS para identificar tarjetas de planes.
        wait_for_selector: Esperar este elemento antes de capturar pantalla.
        notes: Observaciones técnicas sobre quirks del sitio.
    """

    brand: str
    legal_name: str
    plans_url: str
    terms_url: str | None = None
    scraping_mode: ScrapingMode = ScrapingMode.PLAYWRIGHT
    requires_cookie_dismiss: bool = False
    plan_selectors: list[str] = field(default_factory=list)
    wait_for_selector: str | None = None
    notes: str = ""


# ── Registro Central de ISPs ──────────────────────────────────────

ISP_REGISTRY: dict[str, ISPConfig] = {
    # --- COMPETENCIA NACIONAL (FIBRA/COBRE) ---
    "netlife": ISPConfig(
        brand="Netlife",
        legal_name="MEGADATOS S.A.",
        plans_url="https://netlife.ec/",
        terms_url="https://netlife.ec/terminos-y-condiciones-del-servicio",
        requires_cookie_dismiss=True,
        plan_selectors=[".plan-card", "[class*='netlife-plan']", ".pricing-section .card"],
        wait_for_selector=".plan-card",
        notes="Requiere Playwright — el servidor rechaza httpx sin headers correctos.",
    ),
    "claro": ISPConfig(
        brand="Claro",
        legal_name="CONECEL S.A.",
        plans_url="https://www.claro.com.ec/personas/internet/",
        terms_url="https://www.claro.com.ec/sitios/personas/terminos-condiciones/",
        requires_cookie_dismiss=True,
        plan_selectors=[".claro-plan", ".product-card", "[class*='plan-card']"],
        wait_for_selector=".claro-plan",
        notes="React SPA — esperar hidratación del DOM antes de capturar.",
    ),
    "xtrim": ISPConfig(
        brand="Xtrim",
        legal_name="MEGAPROSER S.A.",
        plans_url="https://www.xtrim.com.ec/planes-hogar",
        terms_url="https://www.xtrim.com.ec/contrato-abonado",
        requires_cookie_dismiss=True,
        plan_selectors=[".plan-box", ".card-plan", ".plans-container .item"],
        wait_for_selector=".plan-box",
        notes="Sistema 'Arma tu Plan' con sliders — selectores .plan-box confirmados.",
    ),
    "cnt": ISPConfig(
        brand="CNT",
        legal_name="CORPORACION NACIONAL DE TELECOMUNICACIONES CNT EP",
        plans_url="https://www.cnt.com.ec/internet-hogar/",
        terms_url="https://www.cnt.com.ec/soporte/terminos-condiciones/",
        scraping_mode=ScrapingMode.HYBRID,
        plan_selectors=[".plan", ".cnt-plan", ".precio-plan"],
        wait_for_selector=".plan",
        notes="Mezcla planes residenciales y corporativos — HTML generalmente bien estructurado.",
    ),
    "celerity": ISPConfig(
        brand="Celerity",
        legal_name="PUNTONET S.A.",
        plans_url="https://www.celerity.ec/",
        scraping_mode=ScrapingMode.HTTPX,
        plan_selectors=["[class*='plan']", ".precio"],
        notes="Rebrandeada desde Puntonet — HTML relativamente estático.",
    ),
    "ecuanet": ISPConfig(
        brand="Ecuanet",
        legal_name="ECUADORTELECOM S.A.",
        plans_url="https://ecuanet.ec/",
        terms_url="https://ecuanet.ec/terminos-condiciones",
        plan_selectors=["[class*='plan']", ".precio"],
        notes="Home page contiene los planes principales.",
    ),
    "fibramax": ISPConfig(
        brand="Fibramax",
        legal_name="FIBRAMAX S.A.",
        plans_url="https://fibramax.ec/",
        terms_url="https://fibramax.ec/terminos-condiciones",
        scraping_mode=ScrapingMode.HYBRID,
        plan_selectors=["[class*='plan']", ".precio-fibramax"],
        notes="Precios principalmente en banners de imagen — Vision LLM prioritario.",
    ),
    "alfanet": ISPConfig(
        brand="Alfanet",
        legal_name="ALFANET S.A.",
        plans_url="https://www.alfanet.ec/",
        plan_selectors=["[class*='plan']", ".card-plan"],
        notes="Página mayormente estática.",
    ),
    
    # --- COMPETENCIA INTERNACIONAL Y DISRUPTIVA (SATELITAL / NUEVOS ACTORES) ---
    "starlink": ISPConfig(
        brand="Starlink",
        legal_name="STARLINK ECUADOR S.A.",
        plans_url="https://www.starlink.com/ec/residential",
        terms_url="https://www.starlink.com/legal",
        scraping_mode=ScrapingMode.PLAYWRIGHT,
        requires_cookie_dismiss=True,
        plan_selectors=["[class*='price']", ".layout-container", "md-card"],
        wait_for_selector="text='Mensual'",
        notes="Single Page Application muy pesada (Next.js/React). Vital usar Playwright.",
    ),
    "hughesnet": ISPConfig(
        brand="HughesNet",
        legal_name="HUGHES DE ECUADOR S.A.",
        plans_url="https://www.hughesnet.com.ec/planes",
        scraping_mode=ScrapingMode.PLAYWRIGHT,
        requires_cookie_dismiss=True,
        plan_selectors=[".plan-card", ".price-box", ".offer-details"],
        wait_for_selector=".plan-card",
        notes="Fuerte competidor rural. Pop-ups de geolocalización.",
    ),
    "dfibra": ISPConfig(
        brand="DFibra (DirecTV)",
        legal_name="DIRECTV ECUADOR C. LTDA.",
        plans_url="https://www.directv.com.ec/dfibra",
        scraping_mode=ScrapingMode.PLAYWRIGHT,
        plan_selectors=[".card-plan", ".price-container", "[class*='dfibra-plan']"],
        wait_for_selector=".price-container",
        notes="Entrante agresivo apalancado en base de usuarios de TV.",
    ),
}


def get_isp_config(isp_key: str) -> ISPConfig:
    """Obtiene la configuración completa de un ISP registrado.

    Args:
        isp_key: Clave del ISP en minúsculas (ej: 'netlife', 'claro').

    Returns:
        ISPConfig con toda la metadata del ISP.

    Raises:
        KeyError: Si el ISP no está en el registro.
    """
    if isp_key not in ISP_REGISTRY:
        available = list(ISP_REGISTRY.keys())
        raise KeyError(
            f"ISP '{isp_key}' no está registrado. "
            f"Disponibles: {available}"
        )
    return ISP_REGISTRY[isp_key]


def get_all_isps() -> list[str]:
    """Retorna las claves de todos los ISPs registrados.

    Returns:
        Lista de claves en el orden de inserción del registro.
    """
    return list(ISP_REGISTRY.keys())


def brand_to_legal_name(brand: str) -> str:
    """Convierte un nombre de marca en la razón social oficial.

    Args:
        brand: Nombre comercial del ISP (ej: 'Netlife').

    Returns:
        Razón social si se encuentra, o el mismo brand como fallback.
    """
    for config in ISP_REGISTRY.values():
        if config.brand.lower() == brand.lower():
            return config.legal_name
    return brand  # Fallback: retornar la marca como estaba
