# src/processors/competitive_alerts.py
"""Motor de Alertas Competitivas para toma de decisiones ejecutiva.

Genera alertas priorizadas que identifican amenazas y oportunidades
en el mercado ISP ecuatoriano. Diseñado para que un Product Manager
pueda tomar decisiones en segundos al abrir el reporte.

Tipos de alerta:
    🔴 CRÍTICA:      Competidor significativamente más barato en nuestro segmento.
    🟠 ADVERTENCIA:  Competidor ofrece más streaming/beneficios.
    🟡 OPORTUNIDAD:  Segmento desatendido donde Netlife podría entrar.
    🟢 FORTALEZA:    Segmento donde Netlife es líder en valor.

Uso:
    from src.processors.competitive_alerts import CompetitiveAlertEngine
    engine = CompetitiveAlertEngine(own_brand="Netlife")
    alerts = engine.generate_alerts(df)
    engine.print_executive_summary(alerts)
    alerts_df = engine.to_dataframe(alerts)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from src.utils.logger import logger


class AlertSeverity(str, Enum):
    """Nivel de severidad de una alerta competitiva."""

    CRITICAL = "🔴 CRÍTICA"
    WARNING = "🟠 ADVERTENCIA"
    OPPORTUNITY = "🟡 OPORTUNIDAD"
    STRENGTH = "🟢 FORTALEZA"


@dataclass
class CompetitiveAlert:
    """Alerta competitiva individual.

    Attributes:
        severity: Nivel de urgencia de la alerta.
        title: Título descriptivo breve.
        description: Explicación detallada con datos concretos.
        segment: Segmento de mercado afectado.
        competitor: ISP que genera la alerta.
        metric: Métrica cuantitativa principal.
        recommendation: Acción sugerida para la marca propia.
    """

    severity: AlertSeverity
    title: str
    description: str
    segment: str
    competitor: str
    metric: str
    recommendation: str


class CompetitiveAlertEngine:
    """Motor que analiza el mercado y genera alertas accionables.

    Attributes:
        own_brand: Marca propia para comparar (default: "Netlife").
        price_threshold: Diferencia % para activar alerta CRÍTICA (default: 10%).
    """

    def __init__(
        self,
        own_brand: str = "Netlife",
        price_threshold: float = 0.10,
    ) -> None:
        self.own_brand = own_brand
        self.price_threshold = price_threshold
        self._log = logger.bind(isp=own_brand, phase="competitive_alerts")

    def generate_alerts(self, df: pd.DataFrame) -> list[CompetitiveAlert]:
        """Analiza el DataFrame completo y genera alertas priorizadas.

        Args:
            df: DataFrame enriquecido con etiquetas de StrategicLabeler.

        Returns:
            Lista de CompetitiveAlert ordenadas por severidad (críticas primero).
        """
        alerts: list[CompetitiveAlert] = []
        alerts.extend(self._check_price_threats(df))
        alerts.extend(self._check_streaming_wars(df))
        alerts.extend(self._check_unserved_segments(df))
        alerts.extend(self._check_strengths(df))

        _order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.OPPORTUNITY: 2,
            AlertSeverity.STRENGTH: 3,
        }
        alerts.sort(key=lambda a: _order.get(a.severity, 99))
        self._log.info(
            "Alertas generadas: {} total ({} críticas, {} advertencias, {} oportunidades, {} fortalezas)",
            len(alerts),
            sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL),
            sum(1 for a in alerts if a.severity == AlertSeverity.WARNING),
            sum(1 for a in alerts if a.severity == AlertSeverity.OPPORTUNITY),
            sum(1 for a in alerts if a.severity == AlertSeverity.STRENGTH),
        )
        return alerts

    def _check_price_threats(self, df: pd.DataFrame) -> list[CompetitiveAlert]:
        """Detecta competidores más baratos en cada segmento de velocidad."""
        alerts: list[CompetitiveAlert] = []
        if "segmento_velocidad" not in df.columns:
            return alerts

        own = df[df["marca"] == self.own_brand]
        competitors = df[df["marca"] != self.own_brand]

        for seg in df["segmento_velocidad"].dropna().unique():
            own_seg = own[own["segmento_velocidad"] == seg]
            comp_seg = competitors[competitors["segmento_velocidad"] == seg]

            if own_seg.empty or comp_seg.empty:
                continue

            own_min = own_seg["precio_plan"].min()
            cheapest = comp_seg.loc[comp_seg["precio_plan"].idxmin()]
            comp_price = cheapest["precio_plan"]

            if comp_price >= own_min:
                continue

            diff_pct = (own_min - comp_price) / own_min
            diff_usd = own_min - comp_price
            severity = (
                AlertSeverity.CRITICAL if diff_pct > self.price_threshold
                else AlertSeverity.WARNING
            )

            alerts.append(CompetitiveAlert(
                severity=severity,
                title=(
                    f"{cheapest['marca']} es ${diff_usd:.2f} más barato "
                    f"en segmento '{seg}'"
                ),
                description=(
                    f"{cheapest['marca']} ofrece '{cheapest['nombre_plan']}' "
                    f"a ${comp_price:.2f} vs nuestro mínimo ${own_min:.2f} "
                    f"({diff_pct:.1%} más barato)"
                ),
                segment=seg,
                competitor=str(cheapest["marca"]),
                metric=f"-${diff_usd:.2f} ({diff_pct:.1%})",
                recommendation=(
                    f"Evaluar ajuste de precio en '{seg}' o reforzar "
                    f"propuesta con servicios adicionales."
                ),
            ))

        return alerts

    def _check_streaming_wars(self, df: pd.DataFrame) -> list[CompetitiveAlert]:
        """Detecta competidores con mejor oferta de streaming incluido."""
        alerts: list[CompetitiveAlert] = []
        if "streaming_score" not in df.columns:
            return alerts

        own_max = df[df["marca"] == self.own_brand]["streaming_score"].max()

        for marca in df["marca"].unique():
            if marca == self.own_brand:
                continue
            comp_max = df[df["marca"] == marca]["streaming_score"].max()
            if comp_max > own_max:
                alerts.append(CompetitiveAlert(
                    severity=AlertSeverity.WARNING,
                    title=f"{marca} supera a {self.own_brand} en streaming incluido",
                    description=(
                        f"{marca} tiene planes con streaming score de {comp_max} "
                        f"vs nuestro máximo de {own_max}."
                    ),
                    segment="all",
                    competitor=marca,
                    metric=f"Score: {comp_max} vs {own_max}",
                    recommendation=(
                        "Negociar bundles adicionales de streaming para "
                        "planes de gama media y alta."
                    ),
                ))

        return alerts

    def _check_unserved_segments(self, df: pd.DataFrame) -> list[CompetitiveAlert]:
        """Identifica segmentos de velocidad donde la marca no tiene presencia."""
        alerts: list[CompetitiveAlert] = []
        if "segmento_velocidad" not in df.columns:
            return alerts

        all_segs = set(df["segmento_velocidad"].dropna().unique())
        own_segs = set(
            df[df["marca"] == self.own_brand]["segmento_velocidad"]
            .dropna()
            .unique()
        )

        for seg in (all_segs - own_segs):
            n_comp = df[df["segmento_velocidad"] == seg]["marca"].nunique()
            alerts.append(CompetitiveAlert(
                severity=AlertSeverity.OPPORTUNITY,
                title=f"Segmento '{seg}' sin presencia de {self.own_brand}",
                description=(
                    f"Hay {n_comp} competidores en '{seg}' pero "
                    f"{self.own_brand} no tiene planes aquí."
                ),
                segment=seg,
                competitor=f"{n_comp} ISPs",
                metric=f"{n_comp} competidores activos",
                recommendation=(
                    f"Evaluar lanzamiento de plan en '{seg}' "
                    f"para capturar demanda desatendida."
                ),
            ))

        return alerts

    def _check_strengths(self, df: pd.DataFrame) -> list[CompetitiveAlert]:
        """Identifica segmentos donde la marca propia es líder en precio."""
        alerts: list[CompetitiveAlert] = []
        if "segmento_velocidad" not in df.columns:
            return alerts

        own = df[df["marca"] == self.own_brand]

        for seg in own["segmento_velocidad"].dropna().unique():
            seg_df = df[df["segmento_velocidad"] == seg]
            if seg_df.empty:
                continue
            cheapest = seg_df.loc[seg_df["precio_plan"].idxmin()]
            if cheapest["marca"] != self.own_brand:
                continue

            alerts.append(CompetitiveAlert(
                severity=AlertSeverity.STRENGTH,
                title=f"{self.own_brand} es líder en precio — segmento '{seg}'",
                description=(
                    f"'{cheapest['nombre_plan']}' a ${cheapest['precio_plan']:.2f} "
                    f"es el plan más competitivo del segmento."
                ),
                segment=seg,
                competitor="ninguno",
                metric=f"${cheapest['precio_plan']:.2f} (líder)",
                recommendation="Mantener posición y comunicar liderazgo en campañas.",
            ))

        return alerts

    def to_dataframe(self, alerts: list[CompetitiveAlert]) -> pd.DataFrame:
        """Convierte alertas en DataFrame para visualización en notebook.

        Args:
            alerts: Lista de CompetitiveAlert generadas.

        Returns:
            DataFrame con 7 columnas descriptivas de cada alerta.
        """
        if not alerts:
            return pd.DataFrame(
                columns=["severidad", "titulo", "segmento", "competidor",
                         "metrica", "recomendacion", "descripcion"]
            )
        return pd.DataFrame([
            {
                "severidad": a.severity.value,
                "titulo": a.title,
                "segmento": a.segment,
                "competidor": a.competitor,
                "metrica": a.metric,
                "recomendacion": a.recommendation,
                "descripcion": a.description,
            }
            for a in alerts
        ])

    def print_executive_summary(self, alerts: list[CompetitiveAlert]) -> None:
        """Imprime un resumen ejecutivo formateado para consola/notebook."""
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"  INTELIGENCIA COMPETITIVA -- {self.own_brand.upper()}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(separator)

        by_sev = {s: [a for a in alerts if a.severity == s] for s in AlertSeverity}
        counts = " | ".join(
            f"{s.name}: {len(v)}" for s, v in by_sev.items()
        )
        print(f"\n  {counts}\n")

        for alert in alerts:
            print(f"  [{alert.severity.name}]  {alert.title}")
            print(f"    Metrica: {alert.metric}")
            print(f"    > {alert.recommendation}\n")

        print(separator)
