# tests/test_ml_processors.py
"""Tests para los procesadores ML: StrategicLabeler, MarketClusterer, CompetitiveAlertEngine."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.processors.strategic_labels import StrategicLabeler
from src.processors.competitive_alerts import (
    CompetitiveAlertEngine,
    AlertSeverity,
)
from src.processors.market_clustering import MarketClusterer


# ── Fixture: DataFrame mínimo de prueba ───────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame con datos de ISPs para pruebas de los procesadores ML."""
    return pd.DataFrame([
        # Netlife — líder en segmento medio
        {"marca": "Netlife", "nombre_plan": "Netlife 200", "precio_plan": 24.99,
         "velocidad_download_mbps": 200, "velocidad_upload_mbps": 100,
         "pys_adicionales": 1, "pys_adicionales_detalle": {"Disney+": {"categoria": "streaming", "meses": 6}}},
        # Netlife — segmento alto
        {"marca": "Netlife", "nombre_plan": "Netlife 500", "precio_plan": 39.99,
         "velocidad_download_mbps": 500, "velocidad_upload_mbps": 250,
         "pys_adicionales": 2, "pys_adicionales_detalle": {"Netflix": {"categoria": "streaming", "meses": 12}}},
        # Claro — competidor barato en medio
        {"marca": "Claro", "nombre_plan": "Claro 200", "precio_plan": 19.99,
         "velocidad_download_mbps": 200, "velocidad_upload_mbps": 100,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},
        # Claro — segmento premium
        {"marca": "Claro", "nombre_plan": "Claro 1G", "precio_plan": 65.00,
         "velocidad_download_mbps": 1000, "velocidad_upload_mbps": 500,
         "pys_adicionales": 3, "pys_adicionales_detalle": {}},
        # CNT — económico básico
        {"marca": "CNT", "nombre_plan": "CNT 50", "precio_plan": 14.99,
         "velocidad_download_mbps": 50, "velocidad_upload_mbps": 25,
         "pys_adicionales": 0, "pys_adicionales_detalle": {}},
        # Xtrim — compite en medio
        {"marca": "Xtrim", "nombre_plan": "Xtrim 300", "precio_plan": 27.99,
         "velocidad_download_mbps": 300, "velocidad_upload_mbps": 150,
         "pys_adicionales": 1, "pys_adicionales_detalle": {"Amazon Prime": {"categoria": "streaming", "meses": 3}}},
    ])


@pytest.fixture
def enriched_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame ya enriquecido con StrategicLabeler."""
    return StrategicLabeler().enrich(sample_df)


# ── Tests: StrategicLabeler ───────────────────────────────────────────────────

class TestStrategicLabeler:

    def test_enrich_adds_mbps_por_dolar(self, sample_df):
        df = StrategicLabeler().enrich(sample_df)
        assert "mbps_por_dolar" in df.columns
        # Netlife 200: 200 / 24.99 ≈ 8.00
        row = df[df["nombre_plan"] == "Netlife 200"].iloc[0]
        assert row["mbps_por_dolar"] == pytest.approx(200 / 24.99, rel=0.01)

    def test_enrich_adds_costo_por_100mbps(self, sample_df):
        df = StrategicLabeler().enrich(sample_df)
        assert "costo_por_100mbps" in df.columns

    def test_speed_segments_correct(self, sample_df):
        df = StrategicLabeler().enrich(sample_df)
        assert df[df["nombre_plan"] == "CNT 50"].iloc[0]["segmento_velocidad"] == "basico"
        assert df[df["nombre_plan"] == "Netlife 200"].iloc[0]["segmento_velocidad"] == "medio"
        assert df[df["nombre_plan"] == "Netlife 500"].iloc[0]["segmento_velocidad"] == "alto"
        assert df[df["nombre_plan"] == "Claro 1G"].iloc[0]["segmento_velocidad"] == "premium"

    def test_price_segments_correct(self, sample_df):
        df = StrategicLabeler().enrich(sample_df)
        assert df[df["nombre_plan"] == "CNT 50"].iloc[0]["segmento_precio"] == "economico"
        assert df[df["nombre_plan"] == "Claro 1G"].iloc[0]["segmento_precio"] == "enterprise"

    def test_streaming_score_nonzero_for_plans_with_streaming(self, sample_df):
        df = StrategicLabeler().enrich(sample_df)
        assert df[df["nombre_plan"] == "Netlife 200"].iloc[0]["streaming_score"] > 0
        assert df[df["nombre_plan"] == "Netlife 200"].iloc[0]["tiene_streaming"] == True

    def test_streaming_score_zero_for_no_streaming(self, sample_df):
        df = StrategicLabeler().enrich(sample_df)
        assert df[df["nombre_plan"] == "CNT 50"].iloc[0]["streaming_score"] == 0
        assert df[df["nombre_plan"] == "CNT 50"].iloc[0]["tiene_streaming"] == False

    def test_es_lider_precio_correctness(self, enriched_df):
        # Claro 200 ($19.99) debe ser líder en segmento "medio" (vs Netlife 200 $24.99)
        claro_medio = enriched_df[enriched_df["nombre_plan"] == "Claro 200"].iloc[0]
        assert claro_medio["es_lider_precio"] == True

    def test_diferencia_vs_lider_is_zero_for_leader(self, enriched_df):
        # El líder debe tener diferencia 0
        leaders = enriched_df[enriched_df["es_lider_precio"] == True]
        assert (leaders["diferencia_vs_lider"] == 0.0).all()

    def test_value_tier_assigned_to_all(self, enriched_df):
        assert enriched_df["value_tier"].notna().all()
        assert set(enriched_df["value_tier"].unique()).issubset(
            {"best_value", "good_value", "overpriced"}
        )

    def test_all_10_strategic_columns_added(self, enriched_df):
        expected = {
            "mbps_por_dolar", "costo_por_100mbps",
            "segmento_velocidad", "segmento_precio",
            "streaming_score", "tiene_streaming",
            "ranking_precio", "es_lider_precio",
            "diferencia_vs_lider", "value_tier",
        }
        assert expected.issubset(set(enriched_df.columns))


# ── Tests: CompetitiveAlertEngine ────────────────────────────────────────────

class TestCompetitiveAlertEngine:

    def test_generate_alerts_returns_list(self, enriched_df):
        engine = CompetitiveAlertEngine(own_brand="Netlife")
        alerts = engine.generate_alerts(enriched_df)
        assert isinstance(alerts, list)
        assert len(alerts) > 0

    def test_critical_alert_when_competitor_cheaper(self, enriched_df):
        engine = CompetitiveAlertEngine(own_brand="Netlife", price_threshold=0.05)
        alerts = engine.generate_alerts(enriched_df)
        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        # Claro 200 ($19.99) es ~20% más barato que Netlife 200 ($24.99)
        assert len(critical) > 0

    def test_alerts_sorted_by_severity(self, enriched_df):
        engine = CompetitiveAlertEngine(own_brand="Netlife")
        alerts = engine.generate_alerts(enriched_df)
        order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.OPPORTUNITY: 2,
            AlertSeverity.STRENGTH: 3,
        }
        severities = [order[a.severity] for a in alerts]
        assert severities == sorted(severities)

    def test_to_dataframe_has_correct_columns(self, enriched_df):
        engine = CompetitiveAlertEngine(own_brand="Netlife")
        alerts = engine.generate_alerts(enriched_df)
        df = engine.to_dataframe(alerts)
        assert not df.empty
        for col in ["severidad", "titulo", "segmento", "competidor", "recomendacion"]:
            assert col in df.columns

    def test_empty_alerts_returns_empty_dataframe(self):
        engine = CompetitiveAlertEngine()
        df = engine.to_dataframe([])
        assert df.empty


# ── Tests: MarketClusterer ────────────────────────────────────────────────────

class TestMarketClusterer:

    def test_fit_and_label_adds_cluster_columns(self, enriched_df):
        clusterer = MarketClusterer(n_clusters=3)
        result = clusterer.fit_and_label(enriched_df)
        assert "cluster_id" in result.columns
        assert "cluster_name" in result.columns

    def test_cluster_ids_in_expected_range(self, enriched_df):
        n = 3
        clusterer = MarketClusterer(n_clusters=n)
        result = clusterer.fit_and_label(enriched_df)
        valid = result[result["cluster_id"] != -1]
        assert valid["cluster_id"].between(0, n - 1).all()

    def test_n_unique_clusters_matches_config(self, enriched_df):
        n = 3
        clusterer = MarketClusterer(n_clusters=n)
        result = clusterer.fit_and_label(enriched_df)
        n_clusters = result[result["cluster_id"] != -1]["cluster_id"].nunique()
        assert n_clusters == n

    def test_classify_new_plan_after_training(self, enriched_df):
        clusterer = MarketClusterer(n_clusters=3)
        clusterer.fit_and_label(enriched_df)
        result = clusterer.classify_new_plan({
            "velocidad_download_mbps": 250,
            "velocidad_upload_mbps": 125,
            "precio_plan": 25,
            "pys_adicionales": 1,
            "mbps_por_dolar": 10.0,
        })
        assert "cluster_id" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_new_plan_raises_if_not_fitted(self):
        clusterer = MarketClusterer()
        with pytest.raises(RuntimeError, match="Modelo no entrenado"):
            clusterer.classify_new_plan({})

    def test_get_cluster_centroids_shape(self, enriched_df):
        n = 3
        clusterer = MarketClusterer(n_clusters=n)
        clusterer.fit_and_label(enriched_df)
        centroids = clusterer.get_cluster_centroids()
        assert centroids.shape == (n, len(MarketClusterer.FEATURE_COLUMNS))

    def test_missing_features_returns_sin_clasificar(self, sample_df):
        """Sin enriquecimiento previo (sin mbps_por_dolar), debe degradar gracefully."""
        clusterer = MarketClusterer(n_clusters=3)
        result = clusterer.fit_and_label(sample_df)  # Sin StrategicLabeler previo
        assert "cluster_name" in result.columns
        # Todos deben ser sin_clasificar porque falta mbps_por_dolar
        assert (result["cluster_name"] == "sin_clasificar").all()
