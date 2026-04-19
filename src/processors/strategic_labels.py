# src/processors/strategic_labels.py
"""Motor de Etiquetas Estratégicas para segmentación de mercado ISP.

Transforma datos crudos de planes ISP en inteligencia accionable
mediante la creación de variables derivadas y etiquetas de negocio.

Inspirado en los frameworks de segmentación de McKinsey y BCG
aplicados al mercado de telecomunicaciones ecuatoriano.

Uso:
    from src.processors.strategic_labels import StrategicLabeler
    labeler = StrategicLabeler()
    enriched_df = labeler.enrich(df)
    print(enriched_df[["marca", "mbps_por_dolar", "segmento_velocidad", "value_tier"]])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import logger


class StrategicLabeler:
    """Enriquece el DataFrame con variables estratégicas derivadas.

    Genera etiquetas de segmentación, métricas de valor y
    clasificaciones de mercado que facilitan la toma de decisiones
    ejecutivas sin necesidad de análisis adicional.

    Variables generadas:
        - mbps_por_dolar: Mbps de descarga por cada $1 mensual.
        - costo_por_100mbps: Costo de cada bloque de 100 Mbps.
        - segmento_velocidad: basico | medio | alto | premium.
        - segmento_precio: economico | estandar | premium | enterprise.
        - streaming_score: 0-6+ score de valor de streaming incluido.
        - tiene_streaming: bool — tiene al menos un servicio streaming.
        - ranking_precio: posición dentro de su segmento (1 = más barato).
        - es_lider_precio: True si es el plan más barato del segmento.
        - diferencia_vs_lider: USD de diferencia contra el líder.
        - value_tier: best_value | good_value | overpriced.
    """

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas las transformaciones estratégicas al DataFrame.

        Args:
            df: DataFrame con los datos crudos del pipeline.
                Columnas requeridas: precio_plan, velocidad_download_mbps,
                velocidad_upload_mbps, pys_adicionales_detalle.

        Returns:
            DataFrame enriquecido con 10 columnas estratégicas adicionales.
        """
        log = logger.bind(isp="labeler", phase="enrichment")
        df = df.copy()
        n_before = len(df.columns)

        df = self._add_value_metrics(df)
        df = self._add_speed_segments(df)
        df = self._add_price_segments(df)
        df = self._add_streaming_score(df)
        df = self._add_competitive_position(df)
        df = self._add_value_tier(df)

        n_added = len(df.columns) - n_before
        log.info(
            "Enriquecimiento completado: {} registros, +{} columnas estratégicas",
            len(df),
            n_added,
        )
        return df

    # ── Métricas de Valor por Dinero ──────────────────────────────

    def _add_value_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas de eficiencia precio/velocidad."""
        df["mbps_por_dolar"] = np.where(
            df["precio_plan"] > 0,
            (df["velocidad_download_mbps"] / df["precio_plan"]).round(2),
            0.0,
        )
        df["costo_por_100mbps"] = np.where(
            df["velocidad_download_mbps"] > 0,
            (df["precio_plan"] / (df["velocidad_download_mbps"] / 100)).round(2),
            0.0,
        )
        return df

    # ── Segmentación por Velocidad ────────────────────────────────

    def _add_speed_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clasifica planes por bloque de velocidad de descarga.

        Segmentos:
            basico: ≤100 Mbps — familias pequeñas, uso básico.
            medio: 101–300 Mbps — teletrabajo, streaming HD.
            alto: 301–500 Mbps — gaming, múltiples dispositivos.
            premium: >500 Mbps — power users, hogares premium.
        """
        conditions = [
            df["velocidad_download_mbps"] <= 100,
            df["velocidad_download_mbps"] <= 300,
            df["velocidad_download_mbps"] <= 500,
            df["velocidad_download_mbps"] > 500,
        ]
        choices = ["basico", "medio", "alto", "premium"]
        df["segmento_velocidad"] = np.select(conditions, choices, default="sin_datos")
        return df

    # ── Segmentación por Precio ───────────────────────────────────

    def _add_price_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clasifica planes por rango de precio mensual (USD sin IVA).

        Segmentos:
            economico: ≤$20.
            estandar: $20.01–$35 (mercado masivo).
            premium: $35.01–$50 (valor agregado).
            enterprise: >$50 (segmento corporativo/gamer).
        """
        conditions = [
            df["precio_plan"] <= 20,
            df["precio_plan"] <= 35,
            df["precio_plan"] <= 50,
            df["precio_plan"] > 50,
        ]
        choices = ["economico", "estandar", "premium", "enterprise"]
        df["segmento_precio"] = np.select(conditions, choices, default="sin_datos")
        return df

    # ── Score de Streaming ────────────────────────────────────────

    def _add_streaming_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula score de valor de streaming incluido en el plan.

        Score por servicio:
            - ≤6 meses: 1 punto.
            - >6 meses: 2 puntos.
        """
        def _calc(detail: object) -> int:
            if not isinstance(detail, dict):
                return 0
            score = 0
            for _, info in detail.items():
                if not isinstance(info, dict):
                    continue
                cat = str(info.get("categoria", "")).lower()
                if cat == "streaming":
                    meses = info.get("meses", 0)
                    score += 1 if int(meses) <= 6 else 2
            return score

        df["streaming_score"] = df["pys_adicionales_detalle"].apply(_calc)
        df["tiene_streaming"] = df["streaming_score"] > 0
        return df

    # ── Posición Competitiva Relativa ─────────────────────────────

    def _add_competitive_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determina la posición competitiva dentro de cada segmento de velocidad."""
        df["ranking_precio"] = (
            df.groupby("segmento_velocidad")["precio_plan"]
            .rank(method="min")
            .astype(int)
        )
        precio_lider = df.groupby("segmento_velocidad")["precio_plan"].transform("min")
        df["es_lider_precio"] = df["precio_plan"] == precio_lider
        df["diferencia_vs_lider"] = (df["precio_plan"] - precio_lider).round(2)
        return df

    # ── Value Tier Final ──────────────────────────────────────────

    def _add_value_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clasificación compuesta de valor (precio + velocidad + streaming).

        Tiers:
            best_value: Percentil 70+ en Mbps/$ Y tiene streaming.
            good_value: Percentil 30+ en Mbps/$.
            overpriced: Percentil <30 en Mbps/$.
        """
        p70 = df["mbps_por_dolar"].quantile(0.70)
        p30 = df["mbps_por_dolar"].quantile(0.30)

        conditions = [
            (df["mbps_por_dolar"] >= p70) & (df["streaming_score"] >= 1),
            df["mbps_por_dolar"] >= p30,
            df["mbps_por_dolar"] < p30,
        ]
        choices = ["best_value", "good_value", "overpriced"]
        df["value_tier"] = np.select(conditions, choices, default="good_value")
        return df
