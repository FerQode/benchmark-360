# scripts/enrich_and_export.py
"""Capa de integración ML sobre el Parquet generado.

Flujo:
    1. Cargar benchmark_industria.parquet
    2. StrategicLabeler  → 10 columnas estratégicas
    3. MarketClusterer   → cluster_id + cluster_name + 3 gráficos
    4. CompetitiveAlerts → alertas ejecutivas
    5. Guardar Parquet enriquecido + CSV de alertas

Uso:
    python scripts/enrich_and_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.processors.strategic_labels import StrategicLabeler
from src.processors.market_clustering import MarketClusterer
from src.processors.competitive_alerts import CompetitiveAlertEngine
from src.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    parquet_path = Path(settings.output_dir) / "benchmark_industria.parquet"

    if not parquet_path.exists():
        print("❌ Parquet no encontrado. Ejecuta primero:")
        print("   python scripts/build_demo_dataset.py")
        sys.exit(1)

    # -- 1. Cargar -------------------------------------------------------
    df = pd.read_parquet(parquet_path)
    print(f"[OK] Parquet cargado: {len(df)} registros, {len(df.columns)} columnas")
    print(f"     ISPs: {', '.join(sorted(df['marca'].unique()))}")

    # -- 2. StrategicLabeler --------------------------------------------
    print("\n[1/3] Aplicando StrategicLabeler...")
    labeler = StrategicLabeler()
    df = labeler.enrich(df)
    print(f"     +10 columnas estrategicas -> total: {len(df.columns)}")

    # -- 3. MarketClusterer ---------------------------------------------
    print("\n[2/3] Aplicando KMeans + KNN clustering...")
    clusterer = MarketClusterer(n_clusters=4, random_state=42)
    df = clusterer.fit_and_label(df)

    print("\n     Segmentos descubiertos:")
    for name, grp in df[df["cluster_id"] != -1].groupby("cluster_name"):
        safe_name = name.encode("ascii", "ignore").decode()
        print(
            f"     {safe_name}: {len(grp)} planes | "
            f"Vel: {grp['velocidad_download_mbps'].mean():.0f} Mbps | "
            f"Precio: ${grp['precio_plan'].mean():.2f}"
        )

    print("\n[3/3] Generando graficos ML...")
    charts_dir = str(Path(settings.output_dir) / "charts")
    chart_paths = clusterer.plot_clusters(df, output_dir=charts_dir)
    print(f"     Graficos generados: {len(chart_paths)}")
    for p in chart_paths:
        print(f"     -> {p}")

    # -- 4. CompetitiveAlerts -------------------------------------------
    print("\n[ALERTAS] Generando alertas competitivas...")
    engine = CompetitiveAlertEngine(own_brand="Netlife")
    alerts = engine.generate_alerts(df)
    engine.print_executive_summary(alerts)

    alerts_df = engine.to_dataframe(alerts)
    alerts_path = Path(settings.output_dir) / "competitive_alerts.csv"
    alerts_df.to_csv(alerts_path, index=False, encoding="utf-8-sig")
    print(f"\n     Alertas exportadas: {alerts_path}")

    # -- 5. Guardar Parquet enriquecido ---------------------------------
    df.to_parquet(parquet_path, compression=settings.parquet_compression, index=False)
    print(f"\n[DONE] Parquet final guardado:")
    print(f"       Ruta:      {parquet_path}")
    print(f"       Registros: {len(df)}")
    print(f"       Columnas:  {len(df.columns)}")
    print(f"       ISPs:      {df['marca'].nunique()}")

    ml_cols = [
        "mbps_por_dolar", "segmento_velocidad", "segmento_precio",
        "streaming_score", "tiene_streaming", "value_tier",
        "ranking_precio", "es_lider_precio", "cluster_name",
    ]
    present = [c for c in ml_cols if c in df.columns]
    print(f"\n[ML] Columnas ML: {len(present)}/{len(ml_cols)} activas")
    for c in present:
        n_unique = df[c].nunique()
        print(f"     OK {c}: {n_unique} valores unicos")

    print("\n[FIN] Integracion ML completada exitosamente.")


if __name__ == "__main__":
    main()
