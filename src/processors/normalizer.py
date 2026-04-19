# src/processors/normalizer.py
"""
Normalizer — Converts raw LLM plan dicts into validated ISPPlan objects.

This is the final transformation step before data reaches the Parquet
writer. It bridges the unstructured LLM output and the strict Pydantic
V2 schema, applying:

  - IVA removal (÷ 1.15 or ÷ 1.12 depending on date heuristic)
  - pys_adicionales_detalle snake_case normalization via PYS catalog
  - tecnologia normalization via ISPPlan field_validator
  - ArmaTuPlan Cartesian expansion via ArmaTuPlanHandler
  - discount auto-calculation via ISPPlan model_validator
  - Graceful handling of partial/malformed LLM output

Input (Phase 5 output):
    LLMExtractionResult or VisionExtractionResult with raw_plans

Output (Phase 7 input):
    list[ISPPlan] — fully validated Pydantic V2 objects

Typical usage example:
    >>> normalizer = PlanNormalizer()
    >>> plans = normalizer.normalize_all(
    ...     llm_result=llm_result,
    ...     vision_result=vision_result,
    ...     company_info=company_info,
    ...     extraction_dt=datetime.now(tz=timezone.utc),
    ... )
    >>> print(f"Validated {len(plans)} ISPPlan objects")
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.models.isp_plan import AdditionalServiceDetail, ISPPlan
from src.processors.arma_tu_plan_handler import ArmaTuPlanHandler
from src.processors.llm_processor import LLMExtractionResult
from src.processors.pys_catalog import normalize_pys_detalle
from src.processors.vision_processor import VisionExtractionResult
from src.utils.company_registry import CompanyInfo
from src.processors.hallucination_detector import HallucinationDetector

# ─────────────────────────────────────────────────────────────────
# IVA Constants
# ─────────────────────────────────────────────────────────────────

_IVA_CURRENT: float = 1.15     # Ecuador IVA since April 2024
_IVA_LEGACY: float = 1.12      # Ecuador IVA before April 2024
_IVA_CHANGE_DATE: datetime = datetime(2024, 4, 1, tzinfo=timezone.utc)

# ─────────────────────────────────────────────────────────────────
# PlanNormalizer
# ─────────────────────────────────────────────────────────────────


class PlanNormalizer:
    """Transforms raw LLM plan dicts into validated ISPPlan objects.

    Handles the messy reality of LLM output: missing fields, wrong
    types, prices with IVA included, inconsistent service names.

    Args:
        arma_tu_plan_handler: Handler for Cartesian plan expansion.
            Defaults to a fresh ArmaTuPlanHandler instance.

    Example:
        >>> normalizer = PlanNormalizer()
        >>> plans = normalizer.normalize_all(
        ...     llm_result=llm_result,
        ...     vision_result=vision_result,
        ...     company_info=company_info,
        ...     extraction_dt=datetime.now(tz=timezone.utc),
        ... )
    """

    def __init__(
        self,
        arma_tu_plan_handler: ArmaTuPlanHandler | None = None,
    ) -> None:
        self._arma_handler = arma_tu_plan_handler or ArmaTuPlanHandler()

    def normalize_all(
        self,
        llm_result: LLMExtractionResult,
        vision_result: VisionExtractionResult | None,
        company_info: CompanyInfo,
        extraction_dt: datetime,
        terminos_condiciones_raw: str = "",
    ) -> list[ISPPlan]:
        """Normalize and validate all plans from Phase 5 outputs.

        Merges LLM text results + Vision results, deduplicates by
        plan name, expands any Arma tu Plan configs, and validates
        each plan against the ISPPlan Pydantic schema.

        Args:
            llm_result: Result from LLMProcessor.extract_plans().
            vision_result: Result from VisionProcessor (may be None
                if no screenshots were available).
            company_info: CompanyInfo with empresa and marca names.
            extraction_dt: UTC datetime of this extraction run.

        Returns:
            List of validated ISPPlan objects ready for Parquet.
            Malformed plans are logged and skipped (not raised).
        """
        raw_plans: list[dict] = list(llm_result.raw_plans)

        # Merge vision results (avoid exact duplicates by plan name)
        if vision_result and vision_result.raw_plans:
            existing_names = {
                p.get("nombre_plan", "").strip().lower()
                for p in raw_plans
            }
            for vplan in vision_result.raw_plans:
                name = vplan.get("nombre_plan", "").strip().lower()
                if name and name not in existing_names:
                    raw_plans.append(vplan)
                    existing_names.add(name)

            logger.info(
                "[{}] Merged vision plans: {} text + {} vision = {} total",
                llm_result.isp_key,
                len(llm_result.raw_plans),
                len(vision_result.raw_plans),
                len(raw_plans),
            )

        # Expand Arma tu Plan configurations
        arma_config = (
            llm_result.arma_tu_plan_config
            or (vision_result.arma_tu_plan_config if vision_result else None)
        )
        if arma_config:
            expanded = self._arma_handler.expand(arma_config)
            raw_plans.extend(expanded)
            logger.info(
                "[{}] Arma tu Plan expanded {} combinations",
                llm_result.isp_key,
                len(expanded),
            )

        # Normalize and validate each plan
        validated: list[ISPPlan] = []
        iva_divisor = self._get_iva_divisor(extraction_dt)

        for idx, raw in enumerate(raw_plans):
            try:
                plan = self._normalize_single(
                    raw=raw,
                    company_info=company_info,
                    extraction_dt=extraction_dt,
                    iva_divisor=iva_divisor,
                    isp_key=llm_result.isp_key,
                    terminos_condiciones_raw=terminos_condiciones_raw,
                )
                if plan:
                    validated.append(plan)
            except Exception as exc:
                logger.warning(
                    "[{}] Plan #{} skipped — {}: {}",
                    llm_result.isp_key,
                    idx,
                    type(exc).__name__,
                    exc,
                )

        logger.info(
            "[{}] Normalization complete: {}/{} plans validated",
            llm_result.isp_key,
            len(validated),
            len(raw_plans),
        )
        return validated

    def _normalize_single(
        self,
        raw: dict,
        company_info: CompanyInfo,
        extraction_dt: datetime,
        iva_divisor: float,
        isp_key: str,
        terminos_condiciones_raw: str = "",
    ) -> ISPPlan | None:
        """Normalize and validate a single raw plan dict.

        Args:
            raw: Raw plan dict from LLM output.
            company_info: Company legal and brand information.
            extraction_dt: Extraction datetime for fecha field.
            iva_divisor: IVA multiplier (1.15 or 1.12) to remove.
            isp_key: ISP key for log context.

        Returns:
            Validated ISPPlan or None if plan cannot be recovered.

        Raises:
            ValidationError: Propagated from Pydantic if plan is
                structurally invalid after all recovery attempts.
        """
        if not raw.get("nombre_plan"):
            logger.debug("[{}] Skipping plan with no nombre_plan", isp_key)
            return None

        # ── 0. Validación de Alucinaciones Semánticas ───────────────
        if HallucinationDetector.detect(raw, isp_key):
            reason = HallucinationDetector.get_reason(raw, isp_key)
            logger.warning(
                "ALUCINACIÓN DETECTADA | ISP: {} | Plan: {} | Razón: {}",
                isp_key,
                raw.get("nombre_plan", "unknown"),
                reason
            )
            raw["is_hallucination"] = True
            raw["hallucination_reason"] = reason

        # ── 1. Clean numeric fields ───────────────────────────────
        cleaned = self._clean_numeric_fields(raw)

        # ── 2. Apply IVA removal to price fields ──────────────────
        cleaned = self._remove_iva_from_prices(cleaned, iva_divisor)

        # ── 3. Normalize pys_adicionales_detalle ──────────────────
        raw_pys = cleaned.get("pys_adicionales_detalle", {})
        if isinstance(raw_pys, dict):
            cleaned["pys_adicionales_detalle"] = self._build_pys_detalle(
                raw_pys
            )
        else:
            cleaned["pys_adicionales_detalle"] = {}

        # ── 4. Ensure upload speed ────────────────────────────────
        if not cleaned.get("velocidad_upload_mbps"):
            cleaned["velocidad_upload_mbps"] = cleaned.get(
                "velocidad_download_mbps"
            )

        # ── 5. Coerce list fields ─────────────────────────────────
        for list_field in ("sectores", "parroquia", "provincia"):
            val = cleaned.get(list_field)
            if not isinstance(val, list):
                cleaned[list_field] = [str(val)] if val else []

        # ── 6. Build and validate ISPPlan ─────────────────────────
        # FIX CRÍTICO: El LLM a veces extrae terminos_condiciones del
        # HTML de planes. Hay que eliminarlo del dict ANTES de construir
        # ISPPlan, porque lo inyectamos explícitamente abajo con el texto
        # oficial del TCHTMLScraper. Sin este pop → TypeError duplicado.
        cleaned.pop("terminos_condiciones", None)

        # Determinar T&C a usar (TCHTMLScraper tiene prioridad absoluta)
        tc_text = terminos_condiciones_raw[:5000] if terminos_condiciones_raw else None

        try:
            plan = ISPPlan(
                fecha=extraction_dt,
                empresa=company_info.empresa,
                marca=company_info.marca,
                terminos_condiciones=tc_text,
                **{
                    k: v
                    for k, v in cleaned.items()
                    if k in ISPPlan.model_fields
                },
            )
            return plan

        except ValidationError as exc:
            # Attempt recovery: remove fields that fail validation
            logger.debug(
                "[{}] ValidationError on '{}' — attempting field recovery: {}",
                isp_key,
                cleaned.get("nombre_plan"),
                exc.error_count(),
            )
            return self._recover_from_validation_error(
                exc=exc,
                cleaned=cleaned,
                company_info=company_info,
                extraction_dt=extraction_dt,
                isp_key=isp_key,
                terminos_condiciones_raw=terminos_condiciones_raw,
            )

    def _recover_from_validation_error(
        self,
        exc: ValidationError,
        cleaned: dict,
        company_info: CompanyInfo,
        extraction_dt: datetime,
        isp_key: str,
        terminos_condiciones_raw: str = "",
    ) -> ISPPlan | None:
        """Attempt to build ISPPlan by nullifying invalid fields.

        For each validation error, sets the offending field to None
        and retries. This ensures partial data is captured rather than
        discarding the entire plan record.

        Args:
            exc: The ValidationError from the first attempt.
            cleaned: The cleaned plan dict that failed validation.
            company_info: Company info for ISPPlan fields.
            extraction_dt: Extraction datetime.
            isp_key: ISP key for log context.

        Returns:
            Recovered ISPPlan with nullified invalid fields, or None
            if even the recovery attempt fails.
        """
        recovery = dict(cleaned)

        for error in exc.errors():
            field_path = error.get("loc", ())
            if field_path:
                top_field = field_path[0]
                if top_field in ISPPlan.model_fields:
                    recovery[top_field] = None
                    logger.debug(
                        "[{}] Recovery: nullifying field '{}'",
                        isp_key,
                        top_field,
                    )

        try:
            return ISPPlan(
                fecha=extraction_dt,
                empresa=company_info.empresa,
                marca=company_info.marca,
                terminos_condiciones=terminos_condiciones_raw or None,
                **{
                    k: v
                    for k, v in recovery.items()
                    if k in ISPPlan.model_fields
                },
            )
        except ValidationError as final_exc:
            logger.warning(
                "[{}] Recovery failed for '{}' — {} errors, skipping",
                isp_key,
                cleaned.get("nombre_plan"),
                final_exc.error_count(),
            )
            return None

    # ── Static Helpers ────────────────────────────────────────────

    @staticmethod
    def _get_iva_divisor(extraction_dt: datetime) -> float:
        """Return the correct IVA divisor based on extraction date.

        Ecuador changed IVA from 12% to 15% on April 1, 2024.
        Plans extracted before that date use the legacy 1.12 divisor.

        Args:
            extraction_dt: UTC datetime of the extraction run.

        Returns:
            1.15 for current IVA, 1.12 for legacy IVA.
        """
        if extraction_dt >= _IVA_CHANGE_DATE:
            return _IVA_CURRENT
        return _IVA_LEGACY

    @staticmethod
    def _clean_numeric_fields(raw: dict) -> dict:
        """Parse and coerce numeric price/speed fields from LLM output.

        LLMs sometimes return prices as strings like "$25.00" or
        "25,00". This method extracts the numeric value safely.

        Args:
            raw: Raw plan dict from LLM.

        Returns:
            Dict with numeric fields coerced to float or None.
        """
        numeric_fields = (
            "velocidad_download_mbps",
            "velocidad_upload_mbps",
            "precio_plan",
            "precio_plan_tarjeta",
            "precio_plan_debito",
            "precio_plan_efectivo",
            "precio_plan_descuento",
            "costo_instalacion",
            "descuento",
        )
        int_fields = (
            "meses_descuento",
            "meses_contrato",
            "facturas_gratis",
        )
        cleaned = dict(raw)

        for field in numeric_fields:
            val = cleaned.get(field)
            cleaned[field] = _parse_float(val)

        for field in int_fields:
            val = cleaned.get(field)
            cleaned[field] = _parse_int(val)

        return cleaned

    @staticmethod
    def _remove_iva_from_prices(
        cleaned: dict, iva_divisor: float
    ) -> dict:
        """Remove IVA from all price fields that appear to include it.

        Heuristic: if a price field has a value and the LLM hasn't
        already removed IVA (detected via the field name pattern),
        divide by iva_divisor and round to 2 decimal places.

        Note: costo_instalacion is kept WITH IVA per the schema spec.

        Args:
            cleaned: Plan dict with numeric price fields.
            iva_divisor: IVA multiplier to divide out (1.15 or 1.12).

        Returns:
            Dict with IVA removed from price fields (except instalacion).
        """
        price_fields = (
            "precio_plan",
            "precio_plan_tarjeta",
            "precio_plan_debito",
            "precio_plan_efectivo",
            "precio_plan_descuento",
        )
        result = dict(cleaned)
        for field in price_fields:
            val = result.get(field)
            if val is not None and val > 0:
                result[field] = round(val / iva_divisor, 2)
        return result

    @staticmethod
    def _build_pys_detalle(raw_pys: dict) -> dict[str, dict]:
        """Normalize and validate pys_adicionales_detalle dict.

        Converts raw service names to canonical snake_case keys and
        validates each value against AdditionalServiceDetail schema.

        Args:
            raw_pys: Raw dict from LLM with potentially messy keys.

        Returns:
            Dict with canonical keys and validated detail dicts.
            Invalid entries are skipped with a debug log.
        """
        # First normalize all keys via PYS catalog
        normalized_pys = normalize_pys_detalle(raw_pys)

        # Then validate each value with Pydantic
        validated_pys: dict[str, dict] = {}

        for key, detail in normalized_pys.items():
            if not isinstance(detail, dict):
                continue
            try:
                service = AdditionalServiceDetail(**detail)
                validated_pys[key] = service.model_dump()
            except (ValidationError, TypeError) as exc:
                logger.debug(
                    "pys_adicionales: skipping '{}' — {}", key, exc
                )

        return validated_pys


# ─────────────────────────────────────────────────────────────────
# Private parsing helpers
# ─────────────────────────────────────────────────────────────────


def _parse_float(val: Any) -> float | None:
    """Safely parse a value to float, handling string formats.

    Args:
        val: Value from LLM output (str, int, float, or None).

    Returns:
        Float value or None if unparseable.

    Example:
        >>> _parse_float("$25.00")
        25.0
        >>> _parse_float("25,99")
        25.99
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    if isinstance(val, str):
        # Remove currency symbols, spaces, and normalize decimal separator
        cleaned = re.sub(r"[^\d.,]", "", val.strip())
        # Handle comma as decimal separator (e.g., "25,99")
        if "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        elif "," in cleaned and "." in cleaned:
            last_comma = cleaned.rfind(",")
            last_dot = cleaned.rfind(".")
            if last_comma > last_dot:
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        try:
            result = float(cleaned)
            return result if result > 0 else None
        except ValueError:
            return None
    return None


def _parse_int(val: Any) -> int | None:
    """Safely parse a value to int.

    Args:
        val: Value from LLM output (str, int, float, or None).

    Returns:
        Integer value or None if unparseable.

    Example:
        >>> _parse_int("12")
        12
        >>> _parse_int(3.0)
        3
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val) if val >= 0 else None
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d]", "", val.strip())
        try:
            return int(cleaned) if cleaned else None
        except ValueError:
            return None
    return None
