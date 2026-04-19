# src/utils/company_lookup.py
"""Helper de búsqueda de información de empresa.

Este módulo reemplaza company_registry.py consolidando todos los datos
en el ISP_REGISTRY como fuente única. Provee compatibilidad hacia atrás
para código que aún use la interfaz de company_registry.

Uso:
    from src.utils.company_lookup import brand_to_legal_name, get_company_info
    legal = brand_to_legal_name("Netlife")   # "MEGADATOS S.A."
    brand = legal_name_to_brand("CONECEL S.A.")  # "Claro"
"""

from __future__ import annotations

from src.scrapers.isp_registry import ISP_REGISTRY, ISPConfig, get_isp_config


def brand_to_legal_name(brand: str) -> str:
    """Convierte un nombre de marca en la razón social oficial.

    Args:
        brand: Nombre comercial del ISP (ej: 'Netlife').

    Returns:
        Razón social registrada, o el brand original como fallback.
    """
    for config in ISP_REGISTRY.values():
        if config.brand.lower() == brand.lower():
            return config.legal_name
    return brand


def legal_name_to_brand(legal_name: str) -> str:
    """Convierte una razón social en el nombre comercial del ISP.

    Args:
        legal_name: Razón social (ej: 'CONECEL S.A.').

    Returns:
        Nombre de marca si se encuentra, o el legal_name original.
    """
    for config in ISP_REGISTRY.values():
        if config.legal_name.lower() == legal_name.lower():
            return config.brand
    return legal_name


def get_company_info(isp_key: str) -> ISPConfig:
    """Alias de compatibilidad para código que usaba company_registry.get_company_info().

    Args:
        isp_key: Clave del ISP en minúsculas.

    Returns:
        ISPConfig con brand y legal_name accesibles directamente.

    Raises:
        KeyError: Si el ISP no está registrado.
    """
    return get_isp_config(isp_key)
