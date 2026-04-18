# src/processors/arma_tu_plan_handler.py
"""
ArmaTuPlanHandler — Cartesian product expansion of configurable plans.

Some ISPs (Netlife, Xtrim) offer "Arma tu Plan": a configurable builder
where customers pick from independent dimensions (speed, streaming bundle,
security addon). Each unique combination is a purchasable plan and needs
its own row in the Parquet output.

This handler converts the arma_tu_plan_config dict from the LLM into
N individual raw plan dicts via itertools.product.

Business rule (from hackathon spec):
    "Manejo exitoso de planes tipo Arma tu plan mediante la generación
    de registros por cada combinación posible."

Typical usage example:
    >>> handler = ArmaTuPlanHandler()
    >>> plans = handler.expand(arma_tu_plan_config)
    >>> print(f"Generated {len(plans)} combinations")
    # 3 speeds x 4 bundles = 12 rows
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ExpandedPlan:
    """Single plan record from one configurable plan combination.

    Attributes:
        nombre_plan: Generated name from base + selected option labels.
        velocidad_download_mbps: Download speed for this combination.
        velocidad_upload_mbps: Upload speed for this combination.
        precio_plan: Accumulated price across selected options (no IVA).
        pys_adicionales_detalle: Merged services from chosen options.
        combination_labels: Human-readable selected option labels.
        common_fields: Shared fields (tecnologia, meses_contrato, etc.).
    """

    nombre_plan: str
    velocidad_download_mbps: float | None
    velocidad_upload_mbps: float | None
    precio_plan: float | None
    pys_adicionales_detalle: dict
    combination_labels: list[str]
    common_fields: dict = field(default_factory=dict)

    def to_raw_plan_dict(self) -> dict:
        """Serialize to raw plan dict for Phase 6 normalizer input.

        Returns:
            Flat dict matching the LLM extraction output plan schema.
        """
        base: dict = {
            "nombre_plan": self.nombre_plan,
            "velocidad_download_mbps": self.velocidad_download_mbps,
            "velocidad_upload_mbps": self.velocidad_upload_mbps,
            "precio_plan": self.precio_plan,
            "pys_adicionales_detalle": self.pys_adicionales_detalle,
            "_arma_tu_plan_combination": " + ".join(self.combination_labels),
        }
        for key, value in self.common_fields.items():
            if key not in base:
                base[key] = value
        return base


class ArmaTuPlanHandler:
    """Expands arma_tu_plan_config into individual plan records.

    Uses itertools.product to generate the full Cartesian product of
    all option dimensions, yielding one plan dict per combination.

    Example:
        Dimensions:
          velocidad:  [100 Mbps ($15), 300 Mbps ($25), 600 Mbps ($35)]
          streaming:  [sin_streaming ($0), disney_plus ($5)]

        Result: 3 x 2 = 6 ExpandedPlan records.
    """

    def expand(self, arma_tu_plan_config: dict) -> list[dict]:
        """Expand a configurable plan config into all valid combinations.

        Args:
            arma_tu_plan_config: The arma_tu_plan_config extracted by LLM.
                Expected keys: base_plan_name, option_dimensions,
                common_fields.

        Returns:
            List of raw plan dicts (one per combination) ready for the
            Phase 6 normalizer. Returns empty list if config is invalid.
        """
        if not arma_tu_plan_config:
            return []

        base_name: str = arma_tu_plan_config.get(
            "base_plan_name", "Arma tu Plan"
        )
        dimensions: list[dict] = arma_tu_plan_config.get(
            "option_dimensions", []
        )
        common_fields: dict = arma_tu_plan_config.get(
            "common_fields", {}
        )

        if not dimensions:
            logger.warning(
                "ArmaTuPlanHandler: empty option_dimensions in config"
            )
            return []

        option_lists: list[list[dict]] = []
        dimension_names: list[str] = []

        for dim in dimensions:
            options = dim.get("options", [])
            if options:
                option_lists.append(options)
                dimension_names.append(
                    dim.get("dimension_name", "dimension")
                )

        if not option_lists:
            return []

        combinations = list(itertools.product(*option_lists))

        logger.info(
            "ArmaTuPlanHandler: {} dim(s) → {} combo(s) [{}]",
            len(option_lists),
            len(combinations),
            " × ".join(str(len(o)) for o in option_lists),
        )

        return [
            self._build_combination(
                base_name=base_name,
                combo=combo,
                dimension_names=dimension_names,
                common_fields=common_fields,
            ).to_raw_plan_dict()
            for combo in combinations
        ]

    @staticmethod
    def _build_combination(
        base_name: str,
        combo: tuple[dict, ...],
        dimension_names: list[str],
        common_fields: dict,
    ) -> ExpandedPlan:
        """Build one ExpandedPlan from a single combination tuple.

        Args:
            base_name: Base plan name prefix string.
            combo: Tuple of one selected option dict per dimension.
            dimension_names: Dimension names for label generation.
            common_fields: Shared fields to merge into the plan.

        Returns:
            ExpandedPlan with aggregated speed, price, and services.
        """
        labels: list[str] = []
        download_mbps: float | None = None
        upload_mbps: float | None = None
        total_price: float = 0.0
        merged_pys: dict = {}

        for dim_name, option in zip(dimension_names, combo):
            label = option.get("label", dim_name)
            labels.append(label)

            if download_mbps is None:
                val = option.get("velocidad_download_mbps")
                if val is not None:
                    download_mbps = float(val)

            if upload_mbps is None:
                val = option.get("velocidad_upload_mbps")
                if val is not None:
                    upload_mbps = float(val)

            total_price += float(option.get("precio_adicional", 0.0) or 0.0)

            pys = option.get("pys_detalle", {}) or {}
            merged_pys.update(pys)

        nombre = f"{base_name} — {' + '.join(labels)}"

        return ExpandedPlan(
            nombre_plan=nombre,
            velocidad_download_mbps=download_mbps,
            velocidad_upload_mbps=upload_mbps,
            precio_plan=round(total_price, 2) if total_price else None,
            pys_adicionales_detalle=merged_pys,
            combination_labels=labels,
            common_fields=common_fields,
        )
