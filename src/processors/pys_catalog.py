# src/processors/pys_catalog.py
"""
PYS (Productos y Servicios) normalization catalog for Benchmark 360.

Provides the canonical snake_case mapping for ALL additional services
found across Ecuador ISP websites. This is the single source of truth
for pys_adicionales_detalle key normalization.

Business rule (from hackathon spec):
    "Disney Plus" y "disney +" deberán ser un solo valor "disney_plus".
    Keys must be consistent across ALL ISPs in the final Parquet file.

Adding a new service:
    1. Add its raw variants to PYS_ALIAS_MAP pointing to the canonical key.
    2. Add the canonical key to PYS_CATEGORY_MAP with its category.
    3. Run: uv run pytest tests/unit/test_normalizer.py -v

Typical usage example:
    >>> from src.processors.pys_catalog import normalize_pys_key
    >>> normalize_pys_key("Disney +")
    'disney_plus'
    >>> normalize_pys_key("HBO MAX")
    'hbo_max'
"""

from __future__ import annotations

import re
import unicodedata

# ─────────────────────────────────────────────────────────────────
# Alias Map — raw variants → canonical snake_case key
# ─────────────────────────────────────────────────────────────────
# Keys: lowercase stripped variants found on ISP websites.
# Values: canonical snake_case identifier used in the Parquet schema.

PYS_ALIAS_MAP: dict[str, str] = {
    # ── Disney ───────────────────────────────────────────────────
    "disney plus":              "disney_plus",
    "disney+":                  "disney_plus",
    "disney +":                 "disney_plus",
    "disney plus premium":      "disney_plus",
    "disney plus standard":     "disney_plus",
    "disney plus basic":        "disney_plus",
    # ── Netflix ──────────────────────────────────────────────────
    "netflix":                  "netflix",
    "netflix standard":         "netflix",
    "netflix premium":          "netflix",
    "netflix basico":           "netflix",
    "netflix básico":           "netflix",
    "netflix with ads":         "netflix",
    # ── HBO / Max ─────────────────────────────────────────────────
    "hbo max":                  "hbo_max",
    "hbo":                      "hbo_max",
    "max":                      "hbo_max",
    "hbomax":                   "hbo_max",
    # ── Star+ ────────────────────────────────────────────────────
    "star+":                    "star_plus",
    "star plus":                "star_plus",
    "star +":                   "star_plus",
    # ── Amazon ───────────────────────────────────────────────────
    "amazon prime":             "amazon_prime",
    "amazon prime video":       "amazon_prime",
    "prime video":              "amazon_prime",
    "amazon":                   "amazon_prime",
    # ── Paramount ────────────────────────────────────────────────
    "paramount+":               "paramount_plus",
    "paramount plus":           "paramount_plus",
    "paramount +":              "paramount_plus",
    # ── Apple ────────────────────────────────────────────────────
    "apple tv+":                "apple_tv_plus",
    "apple tv plus":            "apple_tv_plus",
    "apple tv":                 "apple_tv_plus",
    # ── Crunchyroll ──────────────────────────────────────────────
    "crunchyroll":              "crunchyroll",
    "crunchyroll premium":      "crunchyroll",
    # ── Vix ──────────────────────────────────────────────────────
    "vix":                      "vix",
    "vix+":                     "vix_plus",
    "vix plus":                 "vix_plus",
    # ── YouTube ──────────────────────────────────────────────────
    "youtube premium":          "youtube_premium",
    "youtube":                  "youtube_premium",
    # ── Spotify ──────────────────────────────────────────────────
    "spotify":                  "spotify",
    "spotify premium":          "spotify",
    # ── Security — Kaspersky ─────────────────────────────────────
    "kaspersky":                "kaspersky",
    "kaspersky basic":          "kaspersky",
    "kaspersky standard":       "kaspersky",
    "kaspersky plus":           "kaspersky",
    "kaspersky premium":        "kaspersky",
    "netlife defense":          "netlife_defense",
    "defensa netlife":          "netlife_defense",
    # ── Security — Generic ───────────────────────────────────────
    "antivirus":                "antivirus",
    "seguridad digital":        "seguridad_digital",
    "control parental":         "control_parental",
    "parental control":         "control_parental",
    # ── Cloud / Productivity ─────────────────────────────────────
    "microsoft 365":            "microsoft_365",
    "office 365":               "microsoft_365",
    "microsoft office":         "microsoft_365",
    "google one":               "google_one",
    "google drive":             "google_one",
    "dropbox":                  "dropbox",
    # ── Gaming ───────────────────────────────────────────────────
    "xbox game pass":           "xbox_game_pass",
    "xbox":                     "xbox_game_pass",
    "game pass":                "xbox_game_pass",
    "playstation plus":         "playstation_plus",
    "ps plus":                  "playstation_plus",
    "nvidia geforce now":       "geforce_now",
    "geforce now":              "geforce_now",
    # ── Connectivity ─────────────────────────────────────────────
    "router":                   "router_incluido",
    "router incluido":          "router_incluido",
    "modem":                    "router_incluido",
    "wifi":                     "wifi_incluido",
    "wifi incluido":            "wifi_incluido",
    "ip fija":                  "ip_fija",
    "ip estatica":              "ip_fija",
    "ip estática":              "ip_fija",
    "static ip":                "ip_fija",
    # ── Support ──────────────────────────────────────────────────
    "soporte tecnico":          "soporte_tecnico",
    "soporte técnico":          "soporte_tecnico",
    "soporte 24/7":             "soporte_tecnico",
    "soporte 24 7":             "soporte_tecnico",
    "instalacion gratis":       "instalacion_gratis",
    "instalación gratis":       "instalacion_gratis",
    "instalacion gratuita":     "instalacion_gratis",
}

# ─────────────────────────────────────────────────────────────────
# Category Map — canonical key → category label
# ─────────────────────────────────────────────────────────────────

PYS_CATEGORY_MAP: dict[str, str] = {
    # streaming
    "disney_plus":          "streaming",
    "netflix":              "streaming",
    "hbo_max":              "streaming",
    "star_plus":            "streaming",
    "amazon_prime":         "streaming",
    "paramount_plus":       "streaming",
    "apple_tv_plus":        "streaming",
    "crunchyroll":          "streaming",
    "vix":                  "streaming",
    "vix_plus":             "streaming",
    "youtube_premium":      "streaming",
    # music
    "spotify":              "musica",
    # seguridad
    "kaspersky":            "seguridad",
    "netlife_defense":      "seguridad",
    "antivirus":            "seguridad",
    "seguridad_digital":    "seguridad",
    "control_parental":     "seguridad",
    # productividad
    "microsoft_365":        "productividad",
    "google_one":           "productividad",
    "dropbox":              "productividad",
    # gaming
    "xbox_game_pass":       "gaming",
    "playstation_plus":     "gaming",
    "geforce_now":          "gaming",
    # conectividad
    "router_incluido":      "conectividad",
    "wifi_incluido":        "conectividad",
    "ip_fija":              "conectividad",
    # soporte
    "soporte_tecnico":      "soporte",
    "instalacion_gratis":   "soporte",
}


# ─────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────


def normalize_pys_key(raw_name: str) -> str:
    """Convert any raw service name to its canonical snake_case key.

    Lookup strategy:
    1. Exact match in PYS_ALIAS_MAP (after lowercase + strip)
    2. Partial match: raw name contains a known alias
    3. Fallback: slugify the raw name to snake_case

    Args:
        raw_name: Raw service name as found on ISP website.
            Examples: "Disney +", "HBO MAX", "Kaspersky Basic"

    Returns:
        Canonical snake_case key string.
        Examples: "disney_plus", "hbo_max", "kaspersky"

    Example:
        >>> normalize_pys_key("Disney +")
        'disney_plus'
        >>> normalize_pys_key("Unknown Service XYZ")
        'unknown_service_xyz'
    """
    if not raw_name or not raw_name.strip():
        return "servicio_desconocido"

    cleaned = _clean_for_lookup(raw_name)

    # 1. Exact match
    if cleaned in PYS_ALIAS_MAP:
        return PYS_ALIAS_MAP[cleaned]

    # 2. Partial match — check if cleaned contains a known alias
    for alias, canonical in PYS_ALIAS_MAP.items():
        if alias in cleaned or cleaned in alias:
            return canonical

    # 3. Fallback: slugify
    return _slugify(raw_name)


def get_pys_category(canonical_key: str) -> str:
    """Get the category label for a canonical pys key.

    Args:
        canonical_key: Canonical snake_case key from normalize_pys_key().

    Returns:
        Category string. Falls back to "otro" if key is unknown.

    Example:
        >>> get_pys_category("disney_plus")
        'streaming'
        >>> get_pys_category("unknown_service")
        'otro'
    """
    return PYS_CATEGORY_MAP.get(canonical_key, "otro")


def normalize_pys_detalle(raw_detalle: dict) -> dict:
    """Normalize all keys in a pys_adicionales_detalle dict.

    Converts every raw service name key to its canonical snake_case
    equivalent while preserving and enriching the value dict.

    Args:
        raw_detalle: Dict mapping raw service names to detail dicts.
            Each value should have: tipo_plan, meses, categoria.

    Returns:
        New dict with canonical snake_case keys and enriched values
        where categoria is auto-filled if missing.

    Example:
        >>> normalize_pys_detalle({
        ...     "Disney +": {"tipo_plan": "premium", "meses": 9}
        ... })
        {'disney_plus': {'tipo_plan': 'disney_plus_premium',
                         'meses': 9, 'categoria': 'streaming'}}
    """
    normalized: dict = {}

    for raw_key, detail in raw_detalle.items():
        if not isinstance(detail, dict):
            continue

        canonical = normalize_pys_key(str(raw_key))

        # Build enriched detail
        detail_copy = dict(detail)

        # Auto-fill categoria if missing or generic
        if not detail_copy.get("categoria"):
            detail_copy["categoria"] = get_pys_category(canonical)

        # Normalize tipo_plan to snake_case if present
        if detail_copy.get("tipo_plan"):
            detail_copy["tipo_plan"] = _slugify(
                str(detail_copy["tipo_plan"])
            )

        normalized[canonical] = detail_copy

    return normalized


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────


def _clean_for_lookup(text: str) -> str:
    """Normalize text for alias map lookup.

    Strips accents, lowercases, removes extra whitespace.

    Args:
        text: Raw text to normalize for lookup.

    Returns:
        Cleaned lowercase string without diacritics.
    """
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def _slugify(text: str) -> str:
    """Convert any string to a safe snake_case slug.

    Args:
        text: Raw string to slugify.

    Returns:
        Lowercase underscore-separated string with no special chars.

    Example:
        >>> _slugify("HBO Max")
        'hbo_max'
        >>> _slugify("Apple TV+")
        'apple_tv'
    """
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    # Replace non-alphanumeric with underscores
    slugged = re.sub(r"[^a-z0-9]+", "_", lowered)
    # Strip leading/trailing underscores and collapse multiples
    return re.sub(r"_+", "_", slugged).strip("_")
