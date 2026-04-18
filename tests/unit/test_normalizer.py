# tests/unit/test_normalizer.py
"""
Unit tests for Phase 6: PlanNormalizer and PYS catalog.

Tests:
    PYS Catalog:
      - normalize_pys_key: exact matches, partial, fallback slugify
      - get_pys_category: known and unknown keys
      - normalize_pys_detalle: key normalization + Pydantic validation

    PlanNormalizer:
      - IVA divisor selection by date
      - _clean_numeric_fields: string prices, nulls, negatives
      - _remove_iva_from_prices: correct division, costo_instalacion kept
      - _build_pys_detalle: full pipeline with real catalog
      - normalize_all: happy path, vision merge, arma tu plan, validation errors
      - _recover_from_validation_error: partial recovery
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.processors.normalizer import (
    PlanNormalizer,
    _parse_float,
    _parse_int,
)
from src.processors.pys_catalog import (
    PYS_ALIAS_MAP,
    get_pys_category,
    normalize_pys_detalle,
    normalize_pys_key,
)
from src.processors.llm_processor import LLMExtractionResult
from src.processors.vision_processor import VisionExtractionResult
from src.utils.company_registry import CompanyInfo

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def netlife_info() -> CompanyInfo:
    """CompanyInfo fixture for Netlife tests."""
    return CompanyInfo(
        empresa="MEGADATOS S.A.",
        marca="Netlife",
        ruc="1791256115001",
        verified=True,
    )


@pytest.fixture
def extraction_dt_current() -> datetime:
    """Datetime after IVA change (15%)."""
    return datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def extraction_dt_legacy() -> datetime:
    """Datetime before IVA change (12%)."""
    return datetime(2023, 12, 1, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def minimal_raw_plan() -> dict:
    """Minimal raw plan dict that should always validate."""
    return {
        "nombre_plan": "Plan 300 Mbps",
        "velocidad_download_mbps": 345.0,   # with IVA 15%
        "velocidad_upload_mbps": 345.0,
        "precio_plan": 28.75,               # with IVA 15% → 25.0 without
        "pys_adicionales_detalle": {},
        "tecnologia": "fibra_optica",
        "comparticion": "1:1",
        "meses_contrato": 12,
    }


@pytest.fixture
def full_raw_plan() -> dict:
    """Complete raw plan with all optional fields populated."""
    return {
        "nombre_plan": "Plan Dúo 600",
        "velocidad_download_mbps": 690.0,
        "velocidad_upload_mbps": 690.0,
        "precio_plan": 40.25,
        "precio_plan_descuento": 32.2,
        "meses_descuento": 3,
        "costo_instalacion": 0.0,
        "comparticion": "1:1",
        "pys_adicionales_detalle": {
            "Disney +": {
                "tipo_plan": "disney plus premium",
                "meses": 9,
                "categoria": "streaming",
            },
            "HBO Max": {
                "tipo_plan": "hbo max standard",
                "meses": 6,
                "categoria": "streaming",
            },
        },
        "meses_contrato": 12,
        "facturas_gratis": 2,
        "tecnologia": "Fibra Óptica",
        "sectores": ["Norte", "Sur"],
        "parroquia": [],
        "canton": "Quito",
        "provincia": ["Pichincha"],
        "factura_anterior": False,
        "terminos_condiciones": "Aplican T&C",
        "beneficios_publicitados": "Internet simétrico, sin cortes",
    }


@pytest.fixture
def normalizer() -> PlanNormalizer:
    """Fresh PlanNormalizer instance."""
    return PlanNormalizer()


@pytest.fixture
def llm_result_single(minimal_raw_plan: dict) -> LLMExtractionResult:
    """LLMExtractionResult with one minimal plan."""
    return LLMExtractionResult(
        isp_key="netlife",
        raw_plans=[minimal_raw_plan],
    )


@pytest.fixture
def llm_result_empty() -> LLMExtractionResult:
    """LLMExtractionResult with no plans."""
    return LLMExtractionResult(isp_key="netlife", raw_plans=[])


# ─────────────────────────────────────────────────────────────────
# PYS Catalog — normalize_pys_key
# ─────────────────────────────────────────────────────────────────


class TestNormalizePysKey:
    """Validates service name → canonical snake_case mapping."""

    @pytest.mark.unit
    @pytest.mark.parametrize("raw,expected", [
        ("Disney Plus",     "disney_plus"),
        ("disney+",         "disney_plus"),
        ("Disney +",        "disney_plus"),
        ("DISNEY PLUS",     "disney_plus"),
        ("HBO Max",         "hbo_max"),
        ("HBO MAX",         "hbo_max"),
        ("hbo max",         "hbo_max"),
        ("Star+",           "star_plus"),
        ("Star Plus",       "star_plus"),
        ("Netflix",         "netflix"),
        ("NETFLIX",         "netflix"),
        ("Amazon Prime",    "amazon_prime"),
        ("Prime Video",     "amazon_prime"),
        ("Paramount+",      "paramount_plus"),
        ("Apple TV+",       "apple_tv_plus"),
        ("Kaspersky",       "kaspersky"),
        ("Kaspersky Basic", "kaspersky"),
        ("Spotify",         "spotify"),
        ("Microsoft 365",   "microsoft_365"),
        ("Office 365",      "microsoft_365"),
    ])
    def test_known_aliases(self, raw: str, expected: str) -> None:
        assert normalize_pys_key(raw) == expected

    @pytest.mark.unit
    def test_unknown_service_slugified(self) -> None:
        """Unknown services must be slugified to snake_case."""
        result = normalize_pys_key("Unknown Streaming Service")
        assert " " not in result
        assert result == result.lower()
        assert "_" in result or result.isalnum()

    @pytest.mark.unit
    def test_empty_string_returns_fallback(self) -> None:
        assert normalize_pys_key("") == "servicio_desconocido"

    @pytest.mark.unit
    def test_whitespace_only_returns_fallback(self) -> None:
        assert normalize_pys_key("   ") == "servicio_desconocido"

    @pytest.mark.unit
    def test_accented_name_handled(self) -> None:
        """Spanish accented names must be normalized correctly."""
        result = normalize_pys_key("Televisión en vivo")
        assert " " not in result
        assert result == result.lower()

    @pytest.mark.unit
    def test_all_alias_map_values_are_valid_snake_case(self) -> None:
        """Every canonical value in the alias map must be valid snake_case."""
        for raw_alias, canonical in PYS_ALIAS_MAP.items():
            assert canonical == canonical.lower(), (
                f"Canonical '{canonical}' for '{raw_alias}' must be lowercase"
            )
            assert " " not in canonical, (
                f"Canonical '{canonical}' must not contain spaces"
            )


# ─────────────────────────────────────────────────────────────────
# PYS Catalog — get_pys_category
# ─────────────────────────────────────────────────────────────────


class TestGetPysCategory:
    """Validates category lookup for canonical keys."""

    @pytest.mark.unit
    @pytest.mark.parametrize("key,expected_cat", [
        ("disney_plus",     "streaming"),
        ("netflix",         "streaming"),
        ("hbo_max",         "streaming"),
        ("star_plus",       "streaming"),
        ("spotify",         "musica"),
        ("kaspersky",       "seguridad"),
        ("netlife_defense", "seguridad"),
        ("microsoft_365",   "productividad"),
        ("xbox_game_pass",  "gaming"),
        ("router_incluido", "conectividad"),
        ("ip_fija",         "conectividad"),
        ("soporte_tecnico", "soporte"),
    ])
    def test_known_categories(self, key: str, expected_cat: str) -> None:
        assert get_pys_category(key) == expected_cat

    @pytest.mark.unit
    def test_unknown_key_returns_otro(self) -> None:
        assert get_pys_category("completely_unknown_service") == "otro"

    @pytest.mark.unit
    def test_empty_string_returns_otro(self) -> None:
        assert get_pys_category("") == "otro"


# ─────────────────────────────────────────────────────────────────
# PYS Catalog — normalize_pys_detalle
# ─────────────────────────────────────────────────────────────────


class TestNormalizePysDetalle:
    """Validates full detalle dict normalization."""

    @pytest.mark.unit
    def test_normalizes_keys_to_canonical(self) -> None:
        raw = {
            "Disney +": {"tipo_plan": "premium", "meses": 9, "categoria": "streaming"},
            "HBO Max":  {"tipo_plan": "standard", "meses": 6, "categoria": "streaming"},
        }
        result = normalize_pys_detalle(raw)
        assert "disney_plus" in result
        assert "hbo_max" in result
        assert "Disney +" not in result
        assert "HBO Max" not in result

    @pytest.mark.unit
    def test_autofills_missing_categoria(self) -> None:
        raw = {
            "Netflix": {"tipo_plan": "netflix_standard", "meses": 12}
        }
        result = normalize_pys_detalle(raw)
        assert result["netflix"]["categoria"] == "streaming"

    @pytest.mark.unit
    def test_normalizes_tipo_plan_to_snake_case(self) -> None:
        raw = {
            "Disney Plus": {
                "tipo_plan": "Disney Plus Premium",
                "meses": 9,
                "categoria": "streaming",
            }
        }
        result = normalize_pys_detalle(raw)
        tipo = result["disney_plus"]["tipo_plan"]
        assert " " not in tipo
        assert tipo == tipo.lower()

    @pytest.mark.unit
    def test_skips_non_dict_values(self) -> None:
        raw = {
            "netflix": "included",  # wrong type
            "spotify": {"tipo_plan": "premium", "meses": 6, "categoria": "musica"},
        }
        result = normalize_pys_detalle(raw)
        assert "spotify" in result
        # "netflix" had string value — should be skipped or empty
        assert "netflix" not in result or isinstance(result["netflix"], dict)

    @pytest.mark.unit
    def test_empty_input_returns_empty(self) -> None:
        assert normalize_pys_detalle({}) == {}


# ─────────────────────────────────────────────────────────────────
# PlanNormalizer — IVA divisor
# ─────────────────────────────────────────────────────────────────


class TestIvaDivisor:
    """Validates IVA divisor selection by extraction date."""

    @pytest.mark.unit
    def test_current_iva_after_april_2024(
        self, extraction_dt_current: datetime
    ) -> None:
        divisor = PlanNormalizer._get_iva_divisor(extraction_dt_current)
        assert divisor == 1.15

    @pytest.mark.unit
    def test_legacy_iva_before_april_2024(
        self, extraction_dt_legacy: datetime
    ) -> None:
        divisor = PlanNormalizer._get_iva_divisor(extraction_dt_legacy)
        assert divisor == 1.12

    @pytest.mark.unit
    def test_boundary_date_april_2024_uses_current(self) -> None:
        boundary = datetime(2024, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert PlanNormalizer._get_iva_divisor(boundary) == 1.15


# ─────────────────────────────────────────────────────────────────
# PlanNormalizer — _clean_numeric_fields
# ─────────────────────────────────────────────────────────────────


class TestCleanNumericFields:
    """Validates LLM string prices are parsed to floats correctly."""

    @pytest.mark.unit
    @pytest.mark.parametrize("raw_val,expected", [
        (25.0,       25.0),
        ("$25.00",   25.0),
        ("25,99",    25.99),
        ("$1.234,56", 1234.56),
        (0,          None),
        (-5.0,       None),
        (None,       None),
        ("",         None),
        ("N/A",      None),
        ("GRATIS",   None),
    ])
    def test_parse_float_variants(
        self, raw_val: object, expected: float | None
    ) -> None:
        assert _parse_float(raw_val) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("raw_val,expected", [
        (12,     12),
        (3.0,    3),
        ("12",   12),
        ("0",    0),
        (None,   None),
        (-1,     None),
        ("N/A",  None),
        (True,   None),  # booleans must not be treated as ints
    ])
    def test_parse_int_variants(
        self, raw_val: object, expected: int | None
    ) -> None:
        assert _parse_int(raw_val) == expected

    @pytest.mark.unit
    def test_clean_numeric_fields_processes_all_price_fields(self) -> None:
        raw = {
            "nombre_plan": "Test Plan",
            "velocidad_download_mbps": "300",
            "precio_plan": "$28.75",
            "precio_plan_descuento": "23.00",
            "meses_descuento": "3",
            "meses_contrato": "12",
        }
        result = PlanNormalizer._clean_numeric_fields(raw)
        assert result["velocidad_download_mbps"] == 300.0
        assert result["precio_plan"] == 28.75
        assert result["precio_plan_descuento"] == 23.0
        assert result["meses_descuento"] == 3
        assert result["meses_contrato"] == 12


# ─────────────────────────────────────────────────────────────────
# PlanNormalizer — _remove_iva_from_prices
# ─────────────────────────────────────────────────────────────────


class TestRemoveIvaFromPrices:
    """Validates IVA is correctly removed from price fields."""

    @pytest.mark.unit
    def test_removes_iva_15_from_precio_plan(self) -> None:
        cleaned = {"precio_plan": 28.75, "costo_instalacion": 15.0}
        result = PlanNormalizer._remove_iva_from_prices(cleaned, 1.15)
        assert abs(result["precio_plan"] - 25.0) < 0.01

    @pytest.mark.unit
    def test_costo_instalacion_kept_with_iva(self) -> None:
        """costo_instalacion stays WITH IVA per schema spec."""
        cleaned = {"precio_plan": 28.75, "costo_instalacion": 15.0}
        result = PlanNormalizer._remove_iva_from_prices(cleaned, 1.15)
        assert result["costo_instalacion"] == 15.0

    @pytest.mark.unit
    def test_none_prices_stay_none(self) -> None:
        cleaned = {
            "precio_plan": 28.75,
            "precio_plan_tarjeta": None,
            "precio_plan_descuento": None,
        }
        result = PlanNormalizer._remove_iva_from_prices(cleaned, 1.15)
        assert result["precio_plan_tarjeta"] is None
        assert result["precio_plan_descuento"] is None

    @pytest.mark.unit
    def test_removes_iva_12_legacy(self) -> None:
        cleaned = {"precio_plan": 22.4}
        result = PlanNormalizer._remove_iva_from_prices(cleaned, 1.12)
        assert abs(result["precio_plan"] - 20.0) < 0.01

    @pytest.mark.unit
    def test_rounds_to_2_decimals(self) -> None:
        cleaned = {"precio_plan": 100.0}
        result = PlanNormalizer._remove_iva_from_prices(cleaned, 1.15)
        price = result["precio_plan"]
        assert price == round(price, 2)


# ─────────────────────────────────────────────────────────────────
# PlanNormalizer — normalize_all (full pipeline)
# ─────────────────────────────────────────────────────────────────


class TestNormalizeAll:
    """Validates the full normalization pipeline end-to-end."""

    @pytest.mark.unit
    def test_happy_path_returns_isp_plan(
        self,
        normalizer: PlanNormalizer,
        llm_result_single: LLMExtractionResult,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
    ) -> None:
        from src.models.isp_plan import ISPPlan

        plans = normalizer.normalize_all(
            llm_result=llm_result_single,
            vision_result=None,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        assert len(plans) == 1
        assert isinstance(plans[0], ISPPlan)

    @pytest.mark.unit
    def test_isp_plan_fields_populated(
        self,
        normalizer: PlanNormalizer,
        llm_result_single: LLMExtractionResult,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
    ) -> None:
        plans = normalizer.normalize_all(
            llm_result=llm_result_single,
            vision_result=None,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        plan = plans[0]
        assert plan.empresa == "MEGADATOS S.A."
        assert plan.marca == "Netlife"
        assert plan.anio == 2024
        assert plan.mes == 6
        assert plan.dia == 15

    @pytest.mark.unit
    def test_empty_raw_plans_returns_empty_list(
        self,
        normalizer: PlanNormalizer,
        llm_result_empty: LLMExtractionResult,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
    ) -> None:
        plans = normalizer.normalize_all(
            llm_result=llm_result_empty,
            vision_result=None,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        assert plans == []

    @pytest.mark.unit
    def test_plan_without_nombre_plan_skipped(
        self,
        normalizer: PlanNormalizer,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
    ) -> None:
        llm_result = LLMExtractionResult(
            isp_key="netlife",
            raw_plans=[
                {"velocidad_download_mbps": 300.0, "precio_plan": 25.0},
            ],
        )
        plans = normalizer.normalize_all(
            llm_result=llm_result,
            vision_result=None,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        assert len(plans) == 0

    @pytest.mark.unit
    def test_vision_plans_merged_without_duplicates(
        self,
        normalizer: PlanNormalizer,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
        minimal_raw_plan: dict,
    ) -> None:
        """Vision plans with new names are added; duplicates skipped."""
        llm_result = LLMExtractionResult(
            isp_key="netlife",
            raw_plans=[minimal_raw_plan],
        )
        vision_result = VisionExtractionResult(
            isp_key="netlife",
            raw_plans=[
                minimal_raw_plan,  # duplicate — must be skipped
                {
                    "nombre_plan": "Plan 1 Gbps",
                    "velocidad_download_mbps": 1150.0,
                    "velocidad_upload_mbps": 1150.0,
                    "precio_plan": 51.75,
                    "pys_adicionales_detalle": {},
                },
            ],
        )
        plans = normalizer.normalize_all(
            llm_result=llm_result,
            vision_result=vision_result,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        assert len(plans) == 2
        names = {p.nombre_plan for p in plans}
        assert "Plan 300 Mbps" in names
        assert "Plan 1 Gbps" in names

    @pytest.mark.unit
    def test_arma_tu_plan_expanded(
        self,
        normalizer: PlanNormalizer,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
    ) -> None:
        """Arma tu Plan config must produce N rows via Cartesian product."""
        llm_result = LLMExtractionResult(
            isp_key="netlife",
            raw_plans=[],
            arma_tu_plan_config={
                "base_plan_name": "Arma tu Plan",
                "option_dimensions": [
                    {
                        "dimension_name": "velocidad",
                        "options": [
                            {
                                "label": "100 Mbps",
                                "velocidad_download_mbps": 115.0,
                                "precio_adicional": 17.25,
                                "pys_detalle": {},
                            },
                            {
                                "label": "300 Mbps",
                                "velocidad_download_mbps": 345.0,
                                "precio_adicional": 28.75,
                                "pys_detalle": {},
                            },
                        ],
                    },
                    {
                        "dimension_name": "streaming",
                        "options": [
                            {
                                "label": "Sin streaming",
                                "precio_adicional": 0.0,
                                "pys_detalle": {},
                            },
                            {
                                "label": "Disney+",
                                "precio_adicional": 0.0,
                                "pys_detalle": {
                                    "disney_plus": {
                                        "tipo_plan": "disney_plus_premium",
                                        "meses": 12,
                                        "categoria": "streaming",
                                    }
                                },
                            },
                        ],
                    },
                ],
                "common_fields": {
                    "tecnologia": "fibra_optica",
                    "meses_contrato": 12,
                    "costo_instalacion": 0.0,
                },
            },
        )
        plans = normalizer.normalize_all(
            llm_result=llm_result,
            vision_result=None,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        # 2 velocidades × 2 streamings = 4 combinations
        assert len(plans) == 4

    @pytest.mark.unit
    def test_pys_detalle_normalized_in_output(
        self,
        normalizer: PlanNormalizer,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
        full_raw_plan: dict,
    ) -> None:
        """pys_adicionales_detalle keys must be canonical snake_case."""
        llm_result = LLMExtractionResult(
            isp_key="netlife",
            raw_plans=[full_raw_plan],
        )
        plans = normalizer.normalize_all(
            llm_result=llm_result,
            vision_result=None,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        assert len(plans) == 1
        pys_keys = set(plans[0].pys_adicionales_detalle.keys())
        assert "disney_plus" in pys_keys
        assert "hbo_max" in pys_keys
        assert "Disney +" not in pys_keys
        assert "HBO Max" not in pys_keys

    @pytest.mark.unit
    def test_pys_count_synced_with_detalle(
        self,
        normalizer: PlanNormalizer,
        netlife_info: CompanyInfo,
        extraction_dt_current: datetime,
        full_raw_plan: dict,
    ) -> None:
        """pys_adicionales count must match len(pys_adicionales_detalle)."""
        llm_result = LLMExtractionResult(
            isp_key="netlife", raw_plans=[full_raw_plan]
        )
        plans = normalizer.normalize_all(
            llm_result=llm_result,
            vision_result=None,
            company_info=netlife_info,
            extraction_dt=extraction_dt_current,
        )
        plan = plans[0]
        assert plan.pys_adicionales == len(plan.pys_adicionales_detalle)

