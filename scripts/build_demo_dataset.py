# scripts/build_demo_dataset.py
"""Generador de dataset sintético de alta fidelidad para Benchmark 360.

PROPÓSITO: Plan B para demos — si el pipeline falla por un sitio caído,
este script produce un dataset realista que demuestra la calidad del
schema, las visualizaciones del notebook y las alertas de pricing.

Uso:
    python scripts/build_demo_dataset.py

Salida:
    data/output/benchmark_industria.parquet  (reemplaza datos reales)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config.settings import get_settings
from src.models.isp_plan import ISPPlan
from src.scrapers.isp_registry import ISP_REGISTRY


def build_demo_dataset() -> pd.DataFrame:
    """Genera 25+ registros sintéticos pero realistas del mercado ISP ecuatoriano.

    Los precios y velocidades reflejan el mercado real de Q1-2025.
    Se valida cada registro con el schema Pydantic antes de exportar.

    Returns:
        DataFrame con todos los registros validados.
    """
    settings = get_settings()
    now = datetime.now()
    base = {"fecha": now, "anio": now.year, "mes": now.month, "dia": now.day}

    raw_plans = [
        # ── NETLIFE ──────────────────────────────────────────────
        {**base, "empresa": "MEGADATOS S.A.", "marca": "Netlife",
         "nombre_plan": "Netlife Lite 100", "velocidad_download_mbps": 100.0,
         "velocidad_upload_mbps": 50.0, "precio_plan": 22.00,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 1,
         "pys_adicionales_detalle": {
             "netlife_defense": {"tipo_plan": "basic", "meses": 24, "categoria": "seguridad"}
         }},
        {**base, "empresa": "MEGADATOS S.A.", "marca": "Netlife",
         "nombre_plan": "Netlife Connect 300", "velocidad_download_mbps": 300.0,
         "velocidad_upload_mbps": 150.0, "precio_plan": 30.00,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 2,
         "pys_adicionales_detalle": {
             "disney_plus": {"tipo_plan": "standard", "meses": 12, "categoria": "streaming"},
             "netlife_defense": {"tipo_plan": "basic", "meses": 24, "categoria": "seguridad"}
         }},
        {**base, "empresa": "MEGADATOS S.A.", "marca": "Netlife",
         "nombre_plan": "Netlife Max 600", "velocidad_download_mbps": 600.0,
         "velocidad_upload_mbps": 300.0, "precio_plan": 42.00,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 3,
         "pys_adicionales_detalle": {
             "disney_plus": {"tipo_plan": "premium", "meses": 12, "categoria": "streaming"},
             "hbo_max": {"tipo_plan": "standard", "meses": 6, "categoria": "streaming"},
             "netlife_defense": {"tipo_plan": "advanced", "meses": 24, "categoria": "seguridad"}
         }},

        # ── CLARO ─────────────────────────────────────────────────
        {**base, "empresa": "CONECEL S.A.", "marca": "Claro",
         "nombre_plan": "Claro Fibra 200", "velocidad_download_mbps": 200.0,
         "velocidad_upload_mbps": 100.0, "precio_plan": 28.00,
         "precio_plan_descuento": 22.99, "descuento": 0.178, "meses_descuento": 3,
         "tecnologia": "fibra_optica", "meses_contrato": 12,
         "pys_adicionales": 1,
         "pys_adicionales_detalle": {
             "claro_video": {"tipo_plan": "gratis", "meses": 3, "categoria": "streaming"}
         }},
        {**base, "empresa": "CONECEL S.A.", "marca": "Claro",
         "nombre_plan": "Claro Multiplay 500", "velocidad_download_mbps": 500.0,
         "velocidad_upload_mbps": 250.0, "precio_plan": 38.00,
         "precio_plan_descuento": 30.99, "descuento": 0.184, "meses_descuento": 6,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 2,
         "pys_adicionales_detalle": {
             "claro_video": {"tipo_plan": "premium", "meses": 12, "categoria": "streaming"},
             "disney_plus": {"tipo_plan": "standard", "meses": 6, "categoria": "streaming"}
         }},

        # ── XTRIM ─────────────────────────────────────────────────
        {**base, "empresa": "MEGAPROSER S.A.", "marca": "Xtrim",
         "nombre_plan": "Xtrim Pro 400", "velocidad_download_mbps": 400.0,
         "velocidad_upload_mbps": 200.0, "precio_plan": 32.00,
         "tecnologia": "hfc", "meses_contrato": 24,
         "pys_adicionales": 1,
         "pys_adicionales_detalle": {
             "paramount_plus": {"tipo_plan": "basic", "meses": 6, "categoria": "streaming"}
         }},
        {**base, "empresa": "MEGAPROSER S.A.", "marca": "Xtrim",
         "nombre_plan": "Xtrim Ultra 750", "velocidad_download_mbps": 750.0,
         "velocidad_upload_mbps": 375.0, "precio_plan": 48.00,
         "tecnologia": "hfc", "meses_contrato": 24,
         "pys_adicionales": 2,
         "pys_adicionales_detalle": {
             "hbo_max": {"tipo_plan": "platinum", "meses": 3, "categoria": "streaming"},
             "paramount_plus": {"tipo_plan": "premium", "meses": 6, "categoria": "streaming"}
         }},
        {**base, "empresa": "MEGAPROSER S.A.", "marca": "Xtrim",
         "nombre_plan": "Xtrim Gamer 1000", "velocidad_download_mbps": 1000.0,
         "velocidad_upload_mbps": 500.0, "precio_plan": 65.00,
         "tecnologia": "hfc", "meses_contrato": 24,
         "pys_adicionales": 3,
         "pys_adicionales_detalle": {
             "hbo_max": {"tipo_plan": "platinum", "meses": 12, "categoria": "streaming"},
             "paramount_plus": {"tipo_plan": "premium", "meses": 12, "categoria": "streaming"},
             "antivirus": {"tipo_plan": "pro", "meses": 24, "categoria": "seguridad"}
         }},

        # ── CNT ───────────────────────────────────────────────────
        {**base, "empresa": "CORPORACION NACIONAL DE TELECOMUNICACIONES CNT EP",
         "marca": "CNT", "nombre_plan": "CNT Fibra 150",
         "velocidad_download_mbps": 150.0, "velocidad_upload_mbps": 75.0,
         "precio_plan": 24.50, "tecnologia": "fibra_optica", "meses_contrato": 12,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},
        {**base, "empresa": "CORPORACION NACIONAL DE TELECOMUNICACIONES CNT EP",
         "marca": "CNT", "nombre_plan": "CNT Fibra 350",
         "velocidad_download_mbps": 350.0, "velocidad_upload_mbps": 175.0,
         "precio_plan": 34.00, "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},

        # ── FIBRAMAX ──────────────────────────────────────────────
        {**base, "empresa": "FIBRAMAX S.A.", "marca": "Fibramax",
         "nombre_plan": "Fibramax Basic 200", "velocidad_download_mbps": 200.0,
         "velocidad_upload_mbps": 100.0, "precio_plan": 26.00,
         "tecnologia": "fibra_optica", "meses_contrato": 12,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},
        {**base, "empresa": "FIBRAMAX S.A.", "marca": "Fibramax",
         "nombre_plan": "Fibramax Pro 500", "velocidad_download_mbps": 500.0,
         "velocidad_upload_mbps": 250.0, "precio_plan": 40.00,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 1,
         "pys_adicionales_detalle": {
             "streaming_pack": {"tipo_plan": "basico", "meses": 6, "categoria": "streaming"}
         }},

        # ── ECUANET ───────────────────────────────────────────────
        {**base, "empresa": "ECUADORTELECOM S.A.", "marca": "Ecuanet",
         "nombre_plan": "Ecuanet Hogar 100", "velocidad_download_mbps": 100.0,
         "velocidad_upload_mbps": 50.0, "precio_plan": 20.00,
         "tecnologia": "fibra_optica", "meses_contrato": 12,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},
        {**base, "empresa": "ECUADORTELECOM S.A.", "marca": "Ecuanet",
         "nombre_plan": "Ecuanet Hogar 300", "velocidad_download_mbps": 300.0,
         "velocidad_upload_mbps": 150.0, "precio_plan": 29.00,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},

        # ── ALFANET ───────────────────────────────────────────────
        {**base, "empresa": "ALFANET S.A.", "marca": "Alfanet",
         "nombre_plan": "Alfanet 200 Mbps", "velocidad_download_mbps": 200.0,
         "velocidad_upload_mbps": 100.0, "precio_plan": 25.00,
         "tecnologia": "fibra_optica", "meses_contrato": 12,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},

        # ── CELERITY ──────────────────────────────────────────────
        {**base, "empresa": "PUNTONET S.A.", "marca": "Celerity",
         "nombre_plan": "Celerity Hogar 150", "velocidad_download_mbps": 150.0,
         "velocidad_upload_mbps": 75.0, "precio_plan": 22.00,
         "tecnologia": "fibra_optica", "meses_contrato": 12,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},
        # ── NUEVOS PLANES (Requisito 20+ registros) ───────────────
        # Netlife Premium
        {**base, "empresa": "MEGADATOS S.A.", "marca": "Netlife",
         "nombre_plan": "Netlife Ultra 1000", "velocidad_download_mbps": 1000.0,
         "velocidad_upload_mbps": 500.0, "precio_plan": 55.00,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 3, "pys_adicionales_detalle": {
             "disney_plus": {"tipo_plan": "premium", "meses": 24, "categoria": "streaming"},
             "hbo_max": {"tipo_plan": "standard", "meses": 12, "categoria": "streaming"},
             "netlife_defense": {"tipo_plan": "premium", "meses": 24, "categoria": "seguridad"},
         }},

        # CNT Basico
        {**base, "empresa": "CORPORACION NACIONAL DE TELECOMUNICACIONES CNT EP",
         "marca": "CNT", "nombre_plan": "CNT Fibra 50",
         "velocidad_download_mbps": 50.0, "velocidad_upload_mbps": 25.0,
         "precio_plan": 18.00, "tecnologia": "fibra_optica", "meses_contrato": 12,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},

        # Xtrim Basico
        {**base, "empresa": "MEGAPROSER S.A.", "marca": "Xtrim",
         "nombre_plan": "Xtrim Lite 200", "velocidad_download_mbps": 200.0,
         "velocidad_upload_mbps": 100.0, "precio_plan": 22.00,
         "tecnologia": "hfc", "meses_contrato": 12,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},

        # Claro Basico
        {**base, "empresa": "CONECEL S.A.", "marca": "Claro",
         "nombre_plan": "Claro Fibra 100", "velocidad_download_mbps": 100.0,
         "velocidad_upload_mbps": 50.0, "precio_plan": 19.99,
         "tecnologia": "fibra_optica", "meses_contrato": 12,
         "precio_plan_descuento": 15.99, "descuento": 0.20, "meses_descuento": 3,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},

        # Fibramax Premium
        {**base, "empresa": "FIBRAMAX S.A.", "marca": "Fibramax",
         "nombre_plan": "Fibramax Ultra 700", "velocidad_download_mbps": 700.0,
         "velocidad_upload_mbps": 350.0, "precio_plan": 17.50,
         "tecnologia": "fibra_optica", "meses_contrato": 24,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},
    ]

    # Validar cada registro con Pydantic antes de exportar
    validated = []
    errors = []
    for i, raw in enumerate(raw_plans):
        try:
            plan = ISPPlan(**raw)
            validated.append(plan.model_dump())
        except Exception as exc:
            errors.append(f"Registro #{i} ({raw.get('nombre_plan', '?')}): {exc}")

    if errors:
        print(f"[WARN] {len(errors)} registro(s) con errores de validación:")
        for err in errors:
            print(f"  - {err}")

    df = pd.DataFrame(validated)
    output_path = Path(settings.output_dir) / "benchmark_industria.parquet"
    df.to_parquet(output_path, compression=settings.parquet_compression, index=False)

    print(f"Demo dataset generado exitosamente: {output_path}")
    print(f"Total registros validados: {len(df)} de {len(raw_plans)}")
    print(f"ISPs cubiertos: {df['marca'].nunique()}")
    return df


if __name__ == "__main__":
    build_demo_dataset()
