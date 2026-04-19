# src/scrapers/isp_url_strategy.py
"""
ISP URL Strategy Map — Define qué URLs scrapeaar por tipo de dato.

Principio clave:
    Cada tipo de dato tiene su mejor fuente:
    - Planes/Precios  → Home page o página de planes (HTML + Vision)
    - T&C             → URL directa de T&C (HTML puro, sin Vision)
    - Sectores/Coberturas → URL de cobertura (HTML puro)

El técnico de Netlife tiene razón:
    Pasar la URL de T&C directamente al scraper HTML es más eficiente,
    más barato y más preciso que intentar extraerlo de un screenshot.

Google Style Docstrings — PEP8 compliant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PageType(str, Enum):
    """Tipo de página a scrapeaar y su método de extracción.

    Attributes:
        PLANS_HOME: Página principal con planes (HTML + Vision).
        PLANS_DEDICATED: Página dedicada solo a planes (HTML + Vision).
        TERMS_CONDITIONS: T&C — SOLO HTML, nunca Vision.
        COVERAGE: Cobertura/sectores — SOLO HTML.
        PROMOTIONS: Promociones temporales (HTML + Vision).
    """

    PLANS_HOME = "plans_home"
    PLANS_DEDICATED = "plans_dedicated"
    TERMS_CONDITIONS = "terms_conditions"   # ← HTML DIRECTO, sin screenshot
    COVERAGE = "coverage"
    PROMOTIONS = "promotions"


@dataclass
class ISPPageTarget:
    """Define una URL objetivo con su tipo y método de extracción.

    Attributes:
        url: URL completa a scrapeaar.
        page_type: Tipo de página (determina el método de extracción).
        use_vision: Si True, captura tiles para Vision LLM.
        use_playwright: Si True, renderiza JS antes de extraer.
        priority: Orden de procesamiento (1=más importante).
        description: Descripción humana de qué contiene esta URL.
    """

    url: str
    page_type: PageType
    use_vision: bool
    use_playwright: bool
    priority: int = 1
    description: str = ""


@dataclass
class ISPStrategy:
    """Estrategia completa de scraping para un ISP.

    Attributes:
        isp_key: Clave identificadora del ISP.
        empresa: Razón social registrada.
        marca: Marca comercial.
        pages: Lista de páginas a scrapeaar, ordenadas por prioridad.
        specific_plan_selectors: Selectores CSS específicos para este ISP.
        notes: Notas técnicas del ISP (quirks conocidos).
    """

    isp_key: str
    empresa: str
    marca: str
    pages: list[ISPPageTarget]
    specific_plan_selectors: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def plan_pages(self) -> list[ISPPageTarget]:
        """Páginas que contienen planes de internet."""
        return [
            p for p in self.pages
            if p.page_type in (
                PageType.PLANS_HOME,
                PageType.PLANS_DEDICATED,
                PageType.PROMOTIONS,
            )
        ]

    @property
    def tc_pages(self) -> list[ISPPageTarget]:
        """Páginas de Términos y Condiciones (HTML puro)."""
        return [
            p for p in self.pages
            if p.page_type == PageType.TERMS_CONDITIONS
        ]

    @property
    def vision_pages(self) -> list[ISPPageTarget]:
        """Páginas que requieren Vision LLM (tienen imágenes/banners)."""
        return [p for p in self.pages if p.use_vision]


# ─────────────────────────────────────────────────────────────────
# Estrategias por ISP — definidas manualmente con conocimiento
# específico de cada sitio web
# ─────────────────────────────────────────────────────────────────

ISP_STRATEGIES: dict[str, ISPStrategy] = {

    "netlife": ISPStrategy(
        isp_key="netlife",
        empresa="MEGADATOS S.A.",
        marca="Netlife",
        pages=[
            ISPPageTarget(
                url="https://netlife.ec/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Home con planes principales y banners de precios",
            ),
            ISPPageTarget(
                url="https://netlife.ec/planes",
                page_type=PageType.PLANS_DEDICATED,
                use_vision=True,
                use_playwright=True,
                priority=2,
                description="Página dedicada de planes con precios detallados",
            ),
            ISPPageTarget(
                # Netlife T&C — requiere Playwright (servidor rechaza httpx)
                url="https://netlife.ec/terminos-y-condiciones-del-servicio",
                page_type=PageType.TERMS_CONDITIONS,
                use_vision=False,
                use_playwright=True,   # ← Playwright porque el servidor rechaza headers simples
                priority=3,
                description="T&C Netlife — vía Playwright para evitar header error",
            ),
        ],
        specific_plan_selectors=[
            ".plan-card",
            "[class*='netlife-plan']",
            ".pricing-section .card",
        ],
        notes="Usa Playwright: menú de planes requiere hover JS",
    ),

    "xtrim": ISPStrategy(
        isp_key="xtrim",
        empresa="XTRIM S.A.",
        marca="Xtrim",
        pages=[
            ISPPageTarget(
                url="https://www.xtrim.com.ec/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Home con planes destacados y precios",
            ),
            ISPPageTarget(
                url="https://www.xtrim.com.ec/planes-hogar",
                page_type=PageType.PLANS_DEDICATED,
                use_vision=True,
                use_playwright=True,
                priority=2,
                description="Catálogo completo de planes hogar",
            ),
            ISPPageTarget(
                # Xtrim T&C: tínicamente disponible como PDF/página JavaScript
                # Se usa Playwright para renderizar y extraer el texto completo
                url="https://www.xtrim.com.ec/contrato-abonado",
                page_type=PageType.TERMS_CONDITIONS,
                use_vision=False,
                use_playwright=True,
                priority=3,
                description="T&C Xtrim — Playwright para renderizado JS",
            ),
        ],
        specific_plan_selectors=[
            ".plan-box",
            "[class*='plan-xtrim']",
            ".plans-container .item",
        ],
        notes="DOM-sectioning funciona bien — selectores .plan-box confirmados",
    ),

    "claro": ISPStrategy(
        isp_key="claro",
        empresa="CONECEL S.A.",
        marca="Claro",
        pages=[
            ISPPageTarget(
                url="https://www.claro.com.ec/personas/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Home personas con banners de internet fijo",
            ),
            ISPPageTarget(
                url="https://www.claro.com.ec/personas/internet/",
                page_type=PageType.PLANS_DEDICATED,
                use_vision=True,
                use_playwright=True,
                priority=2,
                description="Página internet hogar — planes detallados",
            ),
            ISPPageTarget(
                url="https://www.claro.com.ec/sitios/personas/terminos-condiciones/",
                page_type=PageType.TERMS_CONDITIONS,
                use_vision=False,
                use_playwright=True,
                priority=3,
                description="T&C Claro Ecuador — Playwright para JS",
            ),
        ],
        specific_plan_selectors=[
            ".claro-plan",
            "[class*='plan-card']",
            ".product-card",
        ],
        notes="Fallback a viewport scroll — DOM inconsistente en Claro EC",
    ),

    "cnt": ISPStrategy(
        isp_key="cnt",
        empresa="CORPORACION NACIONAL DE TELECOMUNICACIONES CNT EP",
        marca="CNT",
        pages=[
            ISPPageTarget(
                url="https://www.cnt.com.ec/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Home CNT con planes internet fijo",
            ),
            ISPPageTarget(
                url="https://www.cnt.com.ec/internet-hogar/",
                page_type=PageType.PLANS_DEDICATED,
                use_vision=True,
                use_playwright=True,
                priority=2,
                description="Planes internet hogar CNT",
            ),
            ISPPageTarget(
                url="https://www.cnt.com.ec/soporte/terminos-condiciones/",
                page_type=PageType.TERMS_CONDITIONS,
                use_vision=False,
                use_playwright=True,
                priority=3,
                description="T&C CNT — vía Playwright",
            ),
        ],
        specific_plan_selectors=[".plan", ".precio-plan", ".cnt-plan"],
        notes="HTML generalmente bien estructurado — CNT tiene tablas HTML",
    ),

    "ecuanet": ISPStrategy(
        isp_key="ecuanet",
        empresa="ECUADORTELECOM S.A.",
        marca="Ecuanet",
        pages=[
            ISPPageTarget(
                url="https://ecuanet.ec/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Home Ecuanet con planes y precios",
            ),
            ISPPageTarget(
                url="https://ecuanet.ec/terminos-condiciones",
                page_type=PageType.TERMS_CONDITIONS,
                use_vision=False,
                use_playwright=False,
                priority=2,
                description="T&C Ecuanet — HTML directo",
            ),
        ],
        specific_plan_selectors=["[class*='plan']", ".precio"],
        notes="",
    ),

    "fibramax": ISPStrategy(
        isp_key="fibramax",
        empresa="FIBRAMAX S.A.",
        marca="Fibramax",
        pages=[
            ISPPageTarget(
                url="https://fibramax.ec/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,        # ← Fibramax usa muchos banners
                use_playwright=True,
                priority=1,
                description="Home Fibramax — banners de precios prominentes",
            ),
            ISPPageTarget(
                url="https://fibramax.ec/terminos-condiciones",
                page_type=PageType.TERMS_CONDITIONS,
                use_vision=False,
                use_playwright=False,
                priority=2,
                description="T&C Fibramax — HTML directo",
            ),
        ],
        specific_plan_selectors=["[class*='plan']", ".precio-fibramax"],
        notes="Fibramax publica precios principalmente en imágenes/banners",
    ),

    "alfanet": ISPStrategy(
        isp_key="alfanet",
        empresa="ALFANET S.A.",
        marca="Alfanet",
        pages=[
            ISPPageTarget(
                url="https://www.alfanet.ec/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Home Alfanet con planes de fibra óptica",
            ),
        ],
        specific_plan_selectors=["[class*='plan']", ".card-plan"],
        notes="",
    ),

    "puntonet": ISPStrategy(
        isp_key="puntonet",
        empresa="CELERITY NETWORKS S.A.",
        marca="Celerity",
        pages=[
            ISPPageTarget(
                url="https://www.celerity.ec/",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Home Celerity/Puntonet con planes",
            ),
        ],
        specific_plan_selectors=["[class*='plan']", ".precio"],
        notes="Marca Puntonet migró a Celerity — misma empresa",
    ),

    "starlink": ISPStrategy(
        isp_key="starlink",
        empresa="STARLINK ECUADOR S.A.",
        marca="Starlink",
        pages=[
            ISPPageTarget(
                url="https://www.starlink.com/ec/residential",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="Starlink Residential Plans",
            ),
            ISPPageTarget(
                url="https://www.starlink.com/legal",
                page_type=PageType.TERMS_CONDITIONS,
                use_vision=False,
                use_playwright=False,
                priority=2,
                description="T&C Starlink",
            ),
        ],
    ),

    "hughesnet": ISPStrategy(
        isp_key="hughesnet",
        empresa="HUGHES DE ECUADOR S.A.",
        marca="HughesNet",
        pages=[
            ISPPageTarget(
                url="https://www.hughesnet.com.ec/planes",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="HughesNet Plans",
            ),
        ],
    ),

    "dfibra": ISPStrategy(
        isp_key="dfibra",
        empresa="DIRECTV ECUADOR C. LTDA.",
        marca="DFibra (DirecTV)",
        pages=[
            ISPPageTarget(
                url="https://www.directv.com.ec/dfibra",
                page_type=PageType.PLANS_HOME,
                use_vision=True,
                use_playwright=True,
                priority=1,
                description="DFibra Plans",
            ),
        ],
    ),
}


def get_strategy(isp_key: str) -> ISPStrategy:
    """Obtiene la estrategia de scraping para un ISP específico.

    Args:
        isp_key: Clave del ISP (ej: 'netlife', 'xtrim').

    Returns:
        ISPStrategy configurada para ese ISP.

    Raises:
        KeyError: Si el ISP no está registrado en el mapa.
    """
    if isp_key not in ISP_STRATEGIES:
        raise KeyError(
            f"ISP '{isp_key}' no registrado. "
            f"ISPs disponibles: {list(ISP_STRATEGIES.keys())}"
        )
    return ISP_STRATEGIES[isp_key]
