# src/models/isp_plan.py
"""
ISP Plan data models using Pydantic V2.

Defines the canonical schema for internet service provider
plan data extracted from competitor websites. These models
serve as the single source of truth for data contracts
throughout the entire Benchmark 360 pipeline.

Typical usage example:
    >>> from src.models.isp_plan import ISPPlan, AdditionalServiceDetail
    >>> plan = ISPPlan(
    ...     fecha=datetime.now(),
    ...     empresa="MEGADATOS S.A.",
    ...     marca="Netlife",
    ...     nombre_plan="Plan Dúo 300",
    ...     velocidad_download_mbps=300.0,
    ...     velocidad_upload_mbps=300.0,
    ...     precio_plan=35.90,
    ... )
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Annotated, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ─────────────────────────────────────────────────────────────────
# Sub-modelo: Detalle de servicio adicional
# ─────────────────────────────────────────────────────────────────


class AdditionalServiceDetail(BaseModel):
    """Detail of a single additional service bundled with an ISP plan.

    Each key in the pys_adicionales_detalle dictionary maps
    to an instance of this model. Keys must be in snake_case.

    Attributes:
        tipo_plan: Exact plan tier name of the additional service.
            Example: "disney_plus_premium", "kaspersky_basic".
        meses: Number of months the benefit is included with the plan.
            None if the benefit is permanent during the contract.
        categoria: Category classification of the service.
            Known values: "streaming", "seguridad", "gaming",
            "conectividad", "productividad", "soporte", "otro".

    Example:
        >>> detail = AdditionalServiceDetail(
        ...     tipo_plan="disney_plus_premium",
        ...     meses=9,
        ...     categoria="streaming",
        ... )
    """

    tipo_plan: str = Field(
        ...,
        description="Exact plan tier name of the additional service",
        min_length=1,
        examples=["disney_plus_premium", "netflix_standard", "kaspersky_basic"],
    )
    meses: int | None = Field(
        default=None,
        description="Months the benefit is included. None = permanent",
        ge=0,
        le=120,  # Max 10 años — sanity check
    )
    categoria: str = Field(
        ...,
        description="Service category in snake_case",
        min_length=1,
        examples=["streaming", "seguridad", "gaming", "conectividad"],
    )

    model_config = {
        "str_strip_whitespace": True,  # Elimina espacios automáticamente
        "str_to_lower": False,         # NO convertir a minúsculas (preservar tipo_plan)
    }


# ─────────────────────────────────────────────────────────────────
# Modelo principal: Plan de ISP
# ─────────────────────────────────────────────────────────────────


class ISPPlan(BaseModel):
    """Canonical schema for a single ISP internet plan offering.

    Represents one internet plan from a competitor ISP, extracted
    and normalized from their official website. Enforces strict
    type validation, business rules, and data consistency.

    The model automatically:
    - Syncs anio/mes/dia from the fecha field
    - Calculates descuento from precio_plan and precio_plan_descuento
    - Counts pys_adicionales from the detalle dictionary length

    Attributes:
        fecha: Exact extraction datetime with timezone info.
        anio: Year extracted from fecha (auto-populated).
        mes: Month extracted from fecha (auto-populated).
        dia: Day extracted from fecha (auto-populated).
        empresa: Legal company name as registered in Superintendencia
            de Compañías del Ecuador.
        marca: Commercial brand name. One company may have multiple brands.
        nombre_plan: Official name of the internet plan.
        velocidad_download_mbps: Download speed in Mbps.
        velocidad_upload_mbps: Upload speed in Mbps.
        precio_plan: Base monthly price without IVA, no discounts.
        precio_plan_tarjeta: Price when paying with credit card.
        precio_plan_debito: Price when paying with debit/savings account.
        precio_plan_efectivo: Price when paying with cash.
        precio_plan_descuento: Discounted price (during promo period).
        descuento: Discount ratio (0.0 to 1.0). Auto-calculated.
        meses_descuento: Number of months discount applies.
        costo_instalacion: Total installation cost WITH IVA.
        comparticion: Internet sharing ratio (e.g., "1:1", "1:8").
        pys_adicionales: Count of bundled services. Auto-calculated.
        pys_adicionales_detalle: Dict mapping snake_case service names
            to their detail objects.
        meses_contrato: Minimum contract duration in months.
        facturas_gratis: Number of free monthly invoices.
        tecnologia: Network technology in snake_case.
        sectores: List of geographic sectors with special benefits.
        parroquia: List of parishes with special benefits.
        canton: Canton/municipality with benefits.
        provincia: List of provinces with special benefits.
        factura_anterior: Whether previous ISP invoice is required.
        terminos_condiciones: Full terms and conditions text.
        beneficios_publicitados: Advertised benefits as found on site.
        is_hallucination: Flag indicating if the plan is likely an LLM hallucination.
        hallucination_reason: Detailed reason if the plan is flagged as hallucination.
    """

    # ── 1. Temporal ───────────────────────────────────────────────
    fecha: datetime = Field(
        ...,
        description="Exact extraction datetime (UTC recommended)",
    )
    anio: int = Field(
        default=0,  # Se sobreescribe en model_validator
        ge=2020,
        le=2100,
        description="Year — auto-populated from fecha",
    )
    mes: int = Field(
        default=0,
        ge=1,
        le=12,
        description="Month — auto-populated from fecha",
    )
    dia: int = Field(
        default=0,
        ge=1,
        le=31,
        description="Day — auto-populated from fecha",
    )

    # ── 2. Identificación Empresa ─────────────────────────────────
    empresa: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Legal name from Superintendencia de Compañías",
        examples=["MEGADATOS S.A.", "CONECEL S.A."],
    )
    marca: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Commercial brand name",
        examples=["Netlife", "Claro", "CNT"],
    )

    # ── 3. Plan Base ──────────────────────────────────────────────
    nombre_plan: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Official plan name as shown on website",
    )
    velocidad_download_mbps: float = Field(
        ...,
        gt=0,
        le=100000,  # Sanity check: max 100 Gbps
        description="Download speed in Mbps",
    )
    velocidad_upload_mbps: float = Field(
        ...,
        gt=0,
        le=100000,
        description="Upload speed in Mbps",
    )

    # ── 4. Precios ────────────────────────────────────────────────
    precio_plan: float = Field(
        ...,
        gt=0,
        le=10000,
        description="Base monthly price WITHOUT IVA, no discounts",
    )
    precio_plan_tarjeta: float | None = Field(
        default=None,
        gt=0,
        le=10000,
        description="Price WITH credit card payment, WITHOUT IVA",
    )
    precio_plan_debito: float | None = Field(
        default=None,
        gt=0,
        le=10000,
        description="Price WITH debit/savings account, WITHOUT IVA",
    )
    precio_plan_efectivo: float | None = Field(
        default=None,
        gt=0,
        le=10000,
        description="Price WITH cash payment, WITHOUT IVA",
    )
    precio_plan_descuento: float | None = Field(
        default=None,
        gt=0,
        le=10000,
        description="Discounted monthly price WITHOUT IVA",
    )
    descuento: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Discount ratio 0.0-1.0. Formula: (base-desc)/base",
    )
    meses_descuento: int | None = Field(
        default=None,
        ge=0,
        le=120,
        description="Number of months the discount applies",
    )

    # ── 5. Instalación ────────────────────────────────────────────
    costo_instalacion: float | None = Field(
        default=None,
        ge=0,
        le=10000,
        description="Total installation cost INCLUDING IVA",
    )

    # ── 6. Servicio ───────────────────────────────────────────────
    comparticion: str | None = Field(
        default=None,
        description="Sharing ratio (e.g., '1:1', '1:4', '1:8')",
        examples=["1:1", "1:4", "1:8", "dedicado"],
    )
    pys_adicionales: int = Field(
        default=0,
        ge=0,
        description="Count of bundled services — auto-synced with detalle",
    )
    pys_adicionales_detalle: dict[str, AdditionalServiceDetail] = Field(
        default_factory=dict,
        description="Snake_case keyed dict of bundled service details",
    )

    # ── 7. Contrato ───────────────────────────────────────────────
    meses_contrato: int | None = Field(
        default=None,
        ge=0,
        le=120,
        description="Minimum contract duration in months",
    )
    facturas_gratis: int | None = Field(
        default=None,
        ge=0,
        le=24,
        description="Number of free monthly invoices offered",
    )

    # ── 8. Tecnología ─────────────────────────────────────────────
    tecnologia: str | None = Field(
        default=None,
        description="Network technology in snake_case",
        examples=["fibra_optica", "fttp", "hfc", "cobre", "wimax"],
    )

    # ── 9. Geografía ──────────────────────────────────────────────
    sectores: list[str] = Field(
        default_factory=list,
        description="Geographic sectors with special benefits",
    )
    parroquia: list[str] = Field(
        default_factory=list,
        description="Parishes with special benefits",
    )
    canton: str | None = Field(
        default=None,
        description="Canton/municipality with benefits",
    )
    provincia: list[str] = Field(
        default_factory=list,
        description="Provinces with special benefits",
    )

    # ── 10. Condiciones ───────────────────────────────────────────
    factura_anterior: bool = Field(
        default=False,
        description="Whether previous ISP invoice is required",
    )
    terminos_condiciones: str | None = Field(
        default=None,
        description="Full terms and conditions text as found on site",
    )
    beneficios_publicitados: str | None = Field(
        default=None,
        description="Advertised benefits string as found on site",
    )
    is_hallucination: bool = Field(
        default=False,
        description="Flag indicating if the plan is likely an LLM hallucination",
    )
    hallucination_reason: str | None = Field(
        default=None,
        description="Detailed reason if the plan is flagged as hallucination",
    )

    # ── Model Config ──────────────────────────────────────────────
    model_config = {
        "str_strip_whitespace": True,
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "fecha": "2024-01-15T10:30:00",
                    "empresa": "MEGADATOS S.A.",
                    "marca": "Netlife",
                    "nombre_plan": "Plan Dúo 300",
                    "velocidad_download_mbps": 300.0,
                    "velocidad_upload_mbps": 300.0,
                    "precio_plan": 35.90,
                }
            ]
        },
    }

    # ── Validators ────────────────────────────────────────────────

    @model_validator(mode="after")
    def sync_temporal_fields(self) -> "ISPPlan":
        """Synchronize anio, mes, dia fields from fecha.

        Ensures temporal fields are always consistent with
        the fecha datetime field. Called automatically after
        model initialization.

        Returns:
            Self with synced temporal fields.
        """
        self.anio = self.fecha.year
        self.mes = self.fecha.month
        self.dia = self.fecha.day
        return self

    @model_validator(mode="after")
    def calculate_and_validate_discount(self) -> "ISPPlan":
        """Auto-calculate or validate the discount ratio.

        If precio_plan_descuento is provided:
        - If descuento is None: auto-calculates it
        - If descuento is provided: validates it matches formula

        Formula: descuento = (precio_plan - precio_plan_descuento) / precio_plan

        Returns:
            Self with validated/calculated descuento.

        Raises:
            ValueError: If provided descuento doesn't match formula.
        """
        if self.precio_plan_descuento is not None and self.precio_plan > 0:
            expected = (
                (self.precio_plan - self.precio_plan_descuento)
                / self.precio_plan
            )
            expected = round(expected, 4)

            if self.descuento is None:
                # Auto-calculate
                self.descuento = expected
            else:
                # Validate provided value mathematically (using math.isclose)
                if not math.isclose(self.descuento, expected, abs_tol=0.001):
                    raise ValueError(
                        f"descuento mismatch: provided={self.descuento:.4f}, "
                        f"calculated={expected:.4f} "
                        f"(formula: ({self.precio_plan} - "
                        f"{self.precio_plan_descuento}) / {self.precio_plan})"
                    )
        return self

    @model_validator(mode="after")
    def sync_pys_count_with_detalle(self) -> "ISPPlan":
        """Synchronize pys_adicionales count with detalle dict length.

        Ensures the count field always matches the actual number
        of services in the detail dictionary.

        Returns:
            Self with synced pys_adicionales count.
        """
        self.pys_adicionales = len(self.pys_adicionales_detalle)
        return self

    @field_validator("tecnologia", mode="before")
    @classmethod
    def normalize_tecnologia_field(cls, v: str | None) -> str | None:
        """Normalize technology name to snake_case standard.

        Args:
            v: Raw technology name from web scraping.

        Returns:
            Normalized snake_case technology name, or None.
        """
        if v is None:
            return None

        normalized = v.lower().strip()

        # Mapping de tecnologías conocidas
        tech_mapping = {
            "fibra": "fibra_optica",
            "fibre": "fibra_optica",
            "fiber": "fibra_optica",
            "fttp": "fttp",
            "ftth": "fttp",
            "cobre": "cobre",
            "dsl": "cobre",
            "adsl": "cobre",
            "vdsl": "cobre",
            "hfc": "hfc",
            "coaxial": "hfc",
            "cable": "hfc",
            "wimax": "wimax",
            "4g": "lte_4g",
            "5g": "nr_5g",
            "satelital": "satelital",
            "satellite": "satelital",
        }

        for key, value in tech_mapping.items():
            if key in normalized:
                return value

        # Si no se encuentra en el mapping, convertir a snake_case genérico
        return normalized.replace(" ", "_").replace("-", "_")

    @field_validator("pys_adicionales_detalle", mode="before")
    @classmethod
    def validate_pys_keys_snake_case(
        cls, v: dict[str, AdditionalServiceDetail] | None
    ) -> dict[str, AdditionalServiceDetail] | None:
        """Validate all service keys are in proper snake_case format.

        Args:
            v: Raw dictionary of additional services.

        Returns:
            Dictionary with validated snake_case keys.

        Raises:
            ValueError: If any key contains uppercase letters or spaces.
        """
        if not isinstance(v, dict):
            return v

        for key in v:
            if key != key.lower():
                raise ValueError(
                    f"Service key '{key}' must be lowercase snake_case. "
                    f"Use '{key.lower().replace(' ', '_')}' instead."
                )
            if " " in key:
                raise ValueError(
                    f"Service key '{key}' contains spaces. "
                    f"Use '{key.replace(' ', '_')}' instead."
                )

        return v

    def to_parquet_row(self) -> dict:
        """Convert model to a flat dict ready for Parquet serialization.

        Serializes complex fields (dicts, lists) to JSON strings
        for Parquet compatibility.

        Returns:
            Flat dictionary with all fields ready for pd.DataFrame.
        """
        # Exclude unset handles edge cases, but model_dump provides standard dict
        row = self.model_dump()

        # Serialize complex types for flat Parquet schemas
        # Note: While Parquet supports structs, flattening complex schemas
        # to JSON strings is safer for dynamic nested structures in DataFrames.
        row["pys_adicionales_detalle"] = json.dumps(
            row["pys_adicionales_detalle"],
            ensure_ascii=False,
            default=str,
        )
        row["sectores"] = json.dumps(row["sectores"], ensure_ascii=False)
        row["parroquia"] = json.dumps(row["parroquia"], ensure_ascii=False)
        row["provincia"] = json.dumps(row["provincia"], ensure_ascii=False)

        return row
