# src/utils/company_registry.py
"""
Static company registry mapping ISP brands to legal entities.

Maps commercial brand names to their legal company names as
registered in the Superintendencia de Compañías del Ecuador.
This mapping is used to populate the 'empresa' field in ISPPlan.

Data should be verified against: https://www.supercias.gob.ec

Typical usage example:
    >>> from src.utils.company_registry import get_empresa_name
    >>> get_empresa_name("netlife")
    'MEGADATOS S.A.'
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CompanyInfo:
    """Legal and commercial information for an ISP company.

    Attributes:
        empresa: Legal name in Superintendencia de Compañías.
        marca: Primary commercial brand name.
        ruc: Tax identification number (RUC).
        verified: Whether the data has been verified against Supercias.
    """

    empresa: str
    marca: str
    ruc: str
    verified: bool = False


# ─────────────────────────────────────────────────────────────────
# Registro oficial de empresas ISP en Ecuador
# Fuente: Superintendencia de Compañías del Ecuador
# ─────────────────────────────────────────────────────────────────

COMPANY_REGISTRY: Dict[str, CompanyInfo] = {
    "netlife": CompanyInfo(
        empresa="MEGADATOS S.A.",
        marca="Netlife",
        ruc="1791256115001",
        verified=True,  # ✅ Verificado
    ),
    "claro": CompanyInfo(
        empresa="CONECEL S.A.",
        marca="Claro",
        ruc="0992988061001",
        verified=True,  # ✅ Verificado
    ),
    "cnt": CompanyInfo(
        empresa="CORPORACION NACIONAL DE TELECOMUNICACIONES CNT E.P.",
        marca="CNT",
        ruc="1768152560001",
        verified=True,  # ✅ Verificado
    ),
    "xtrim": CompanyInfo(
        empresa="TV CABLE S.A.",  # ⚠️ VERIFICAR en supercias.gob.ec
        marca="Xtrim",
        ruc="PENDIENTE_VERIFICAR",
        verified=False,
    ),
    "ecuanet": CompanyInfo(
        empresa="ECUANET CIA. LTDA.",  # ⚠️ VERIFICAR
        marca="Ecuanet",
        ruc="PENDIENTE_VERIFICAR",
        verified=False,
    ),
    "puntonet": CompanyInfo(
        empresa="CELERITY NETWORKS S.A.",  # ⚠️ VERIFICAR
        marca="Puntonet",
        ruc="PENDIENTE_VERIFICAR",
        verified=False,
    ),
    "alfanet": CompanyInfo(
        empresa="ALFANET S.A.",  # ⚠️ VERIFICAR
        marca="Alfanet",
        ruc="PENDIENTE_VERIFICAR",
        verified=False,
    ),
    "fibramax": CompanyInfo(
        empresa="FIBRAMAX S.A.",  # ⚠️ VERIFICAR
        marca="Fibramax",
        ruc="PENDIENTE_VERIFICAR",
        verified=False,
    ),
}


def get_company_info(isp_key: str) -> CompanyInfo:
    """Retrieve company info by ISP key.

    Args:
        isp_key: Internal ISP identifier (lowercase, no spaces).

    Returns:
        CompanyInfo dataclass with legal and commercial data.

    Raises:
        KeyError: If isp_key is not in the registry.

    Example:
        >>> info = get_company_info("netlife")
        >>> info.empresa
        'MEGADATOS S.A.'
    """
    key = isp_key.lower().strip()
    if key not in COMPANY_REGISTRY:
        raise KeyError(
            f"ISP key '{key}' not found in registry. "
            f"Available keys: {list(COMPANY_REGISTRY.keys())}"
        )
    return COMPANY_REGISTRY[key]


def get_empresa_name(isp_key: str) -> str:
    """Get the legal company name for an ISP.

    Args:
        isp_key: Internal ISP identifier.

    Returns:
        Legal company name string.

    Example:
        >>> get_empresa_name("claro")
        'CONECEL S.A.'
    """
    return get_company_info(isp_key).empresa


def get_marca_name(isp_key: str) -> str:
    """Get the commercial brand name for an ISP.

    Args:
        isp_key: Internal ISP identifier.

    Returns:
        Commercial brand name string.

    Example:
        >>> get_marca_name("cnt")
        'CNT'
    """
    return get_company_info(isp_key).marca


def get_unverified_companies() -> List[str]:
    """Get list of ISP keys with unverified company data.

    Returns:
        List of ISP keys where verified=False.

    Example:
        >>> unverified = get_unverified_companies()
        >>> print(f"⚠️ {len(unverified)} companies need verification")
    """
    return [
        key for key, info in COMPANY_REGISTRY.items()
        if not info.verified
    ]
