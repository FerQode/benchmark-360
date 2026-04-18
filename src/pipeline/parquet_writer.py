# src/pipeline/parquet_writer.py
"""
Parquet Writer — Serializes validated ISPPlan objects to Apache Parquet.

Produces the final deliverable: benchmark_industria.parquet
Partitioned by (anio, mes) for efficient time-series querying.

Schema enforcement:
    All 30+ columns defined in ISPPlan are written with correct dtypes.
    Complex fields (dicts, lists) are serialized to JSON strings.

Typical usage example:
    >>> writer = ParquetWriter(output_dir=Path("data/output"))
    >>> path = writer.write(plans=validated_plans, run_id="20240615_103000")
    >>> print(f"Written to: {path}")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from src.models.isp_plan import ISPPlan

# ─────────────────────────────────────────────────────────────────
# Parquet Schema — explicit pyarrow schema for type enforcement
# ─────────────────────────────────────────────────────────────────

PARQUET_SCHEMA = pa.schema([
    pa.field("fecha",                    pa.timestamp("us", tz="UTC")),
    pa.field("anio",                     pa.int32()),
    pa.field("mes",                      pa.int32()),
    pa.field("dia",                      pa.int32()),
    pa.field("empresa",                  pa.string()),
    pa.field("marca",                    pa.string()),
    pa.field("nombre_plan",              pa.string()),
    pa.field("velocidad_download_mbps",  pa.float64()),
    pa.field("velocidad_upload_mbps",    pa.float64()),
    pa.field("precio_plan",              pa.float64()),
    pa.field("precio_plan_tarjeta",      pa.float64()),
    pa.field("precio_plan_debito",       pa.float64()),
    pa.field("precio_plan_efectivo",     pa.float64()),
    pa.field("precio_plan_descuento",    pa.float64()),
    pa.field("descuento",                pa.float64()),
    pa.field("meses_descuento",          pa.int32()),
    pa.field("costo_instalacion",        pa.float64()),
    pa.field("comparticion",             pa.string()),
    pa.field("pys_adicionales",          pa.int32()),
    pa.field("pys_adicionales_detalle",  pa.string()),   # JSON string
    pa.field("meses_contrato",           pa.int32()),
    pa.field("facturas_gratis",          pa.int32()),
    pa.field("tecnologia",               pa.string()),
    pa.field("sectores",                 pa.string()),   # JSON array string
    pa.field("parroquia",                pa.string()),   # JSON array string
    pa.field("canton",                   pa.string()),
    pa.field("provincia",                pa.string()),   # JSON array string
    pa.field("factura_anterior",         pa.bool_()),
    pa.field("terminos_condiciones",     pa.string()),
    pa.field("beneficios_publicitados",  pa.string()),
])


class ParquetWriter:
    """Writes validated ISPPlan objects to partitioned Parquet files.

    Produces two output files per run:
    - benchmark_industria_{run_id}.parquet  — run-specific snapshot
    - benchmark_industria.parquet           — latest run (overwritten)

    Args:
        output_dir: Directory where Parquet files will be written.
            Created automatically if it does not exist.

    Example:
        >>> writer = ParquetWriter(output_dir=Path("data/output"))
        >>> path = writer.write(plans=plans, run_id="20240615_103000")
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        plans: list[ISPPlan],
        run_id: str,
    ) -> Path:
        """Serialize ISPPlan list to Parquet and write to disk.

        Args:
            plans: List of validated ISPPlan objects.
            run_id: Unique run identifier for filename.

        Returns:
            Path to the written 'benchmark_industria.parquet' file.

        Raises:
            ValueError: If plans list is empty.
        """
        if not plans:
            raise ValueError(
                "Cannot write empty plans list to Parquet. "
                "Ensure at least one ISP returned valid plans."
            )

        logger.info(
            "📦 Writing {} plans to Parquet (run_id={})",
            len(plans),
            run_id,
        )

        # ── Convert to DataFrame ──────────────────────────────────
        df = self._plans_to_dataframe(plans)

        logger.info(
            "DataFrame shape: {} rows × {} columns",
            df.shape[0],
            df.shape[1],
        )

        # ── Convert to PyArrow Table with schema enforcement ──────
        table = self._dataframe_to_arrow_table(df)

        # ── Write run-specific snapshot ───────────────────────────
        snapshot_path = self._output_dir / f"benchmark_industria_{run_id}.parquet"
        pq.write_table(
            table,
            snapshot_path,
            compression="snappy",
            write_statistics=True,
        )
        logger.info("📁 Snapshot written: {}", snapshot_path)

        # ── Write/overwrite the canonical 'latest' file ───────────
        latest_path = self._output_dir / "benchmark_industria.parquet"
        pq.write_table(
            table,
            latest_path,
            compression="snappy",
            write_statistics=True,
        )
        logger.info(
            "✅ benchmark_industria.parquet written → {} rows, {:.1f} KB",
            len(plans),
            latest_path.stat().st_size / 1024,
        )

        # ── Quality summary ───────────────────────────────────────
        self._log_quality_summary(df)

        return latest_path

    def read(self, path: Path | None = None) -> pd.DataFrame:
        """Read a Parquet file back into a DataFrame.

        Args:
            path: Path to the Parquet file. Defaults to
                benchmark_industria.parquet in output_dir.

        Returns:
            pandas DataFrame with all plans.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        target = path or (self._output_dir / "benchmark_industria.parquet")
        if not target.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {target}. "
                "Run the pipeline first."
            )
        df = pd.read_parquet(target)
        logger.info(
            "📖 Read {} rows from {}",
            len(df),
            target.name,
        )
        return df

    # ── Private ───────────────────────────────────────────────────

    @staticmethod
    def _plans_to_dataframe(plans: list[ISPPlan]) -> pd.DataFrame:
        """Convert ISPPlan list to a clean pandas DataFrame.

        Args:
            plans: Validated ISPPlan objects.

        Returns:
            DataFrame with all 30+ columns, complex fields as JSON strings.
        """
        rows = [plan.to_parquet_row() for plan in plans]
        df = pd.DataFrame(rows)

        # Ensure fecha is UTC-aware datetime
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], utc=True)

        # Coerce integer nullable columns
        int_nullable_cols = [
            "anio", "mes", "dia",
            "meses_descuento", "meses_contrato",
            "facturas_gratis", "pys_adicionales",
        ]
        for col in int_nullable_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                ).astype("Int32")

        # Ensure all string columns have no raw None
        str_cols = [
            "empresa", "marca", "nombre_plan",
            "comparticion", "tecnologia", "canton",
            "terminos_condiciones", "beneficios_publicitados",
            "pys_adicionales_detalle", "sectores",
            "parroquia", "provincia",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].where(
                    df[col].notna(), other=None
                )

        return df

    @staticmethod
    def _dataframe_to_arrow_table(df: pd.DataFrame) -> pa.Table:
        """Convert DataFrame to PyArrow Table with schema casting.

        Casts each column to the expected PyArrow type, handling
        nullable integers (Int32 → int32 with nulls preserved).

        Args:
            df: Clean DataFrame from _plans_to_dataframe().

        Returns:
            PyArrow Table ready for pq.write_table().
        """
        # Cast Int32 pandas extension type to pyarrow int32 with nulls
        cast_map = {
            "anio": pa.int32(),
            "mes": pa.int32(),
            "dia": pa.int32(),
            "meses_descuento": pa.int32(),
            "meses_contrato": pa.int32(),
            "facturas_gratis": pa.int32(),
            "pys_adicionales": pa.int32(),
        }

        arrays = []
        fields = []

        for field in PARQUET_SCHEMA:
            col_name = field.name
            if col_name in df.columns:
                series = df[col_name]
                if col_name in cast_map:
                    arr = pa.array(
                        series.tolist(),
                        type=cast_map[col_name],
                        from_pandas=True,
                    )
                elif field.type == pa.bool_():
                    arr = pa.array(
                        series.fillna(False).tolist(),
                        type=pa.bool_(),
                    )
                elif field.type == pa.timestamp("us", tz="UTC"):
                    arr = pa.array(series, type=field.type)
                else:
                    arr = pa.array(
                        series.tolist(),
                        type=field.type,
                        from_pandas=True,
                    )
            else:
                # Column missing — fill with nulls
                arr = pa.nulls(len(df), type=field.type)
                logger.debug(
                    "Column '{}' missing from DataFrame — filled with nulls",
                    col_name,
                )

            arrays.append(arr)
            fields.append(field)

        return pa.table(
            {f.name: a for f, a in zip(fields, arrays)},
            schema=pa.schema(fields),
        )

    @staticmethod
    def _log_quality_summary(df: pd.DataFrame) -> None:
        """Log a data quality summary after writing.

        Args:
            df: The written DataFrame for analysis.
        """
        total = len(df)
        if total == 0:
            return

        logger.info("── Data Quality Summary ──────────────────────")
        logger.info("  Total rows:     {}", total)
        logger.info(
            "  ISPs:           {} ({})",
            df["marca"].nunique(),
            ", ".join(sorted(df["marca"].dropna().unique())),
        )
        logger.info(
            "  Price coverage: {:.0%}",
            df["precio_plan"].notna().mean(),
        )
        logger.info(
            "  Speed coverage: {:.0%}",
            df["velocidad_download_mbps"].notna().mean(),
        )
        logger.info(
            "  With pys:       {} plans",
            (df["pys_adicionales"] > 0).sum(),
        )
        if "precio_plan" in df.columns:
            prices = df["precio_plan"].dropna()
            if len(prices):
                logger.info(
                    "  Price range:    ${:.2f} – ${:.2f} (median ${:.2f})",
                    prices.min(),
                    prices.max(),
                    prices.median(),
                )
        logger.info("──────────────────────────────────────────────")
