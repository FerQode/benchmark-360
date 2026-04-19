# src/processors/market_clustering.py
"""Motor de Clustering de Mercado ISP usando KMeans + KNN.

Aplica Machine Learning no supervisado para descubrir segmentos
naturales del mercado que no son visibles con reglas manuales.

KMeans: descubre clusters naturales en el espacio precio/velocidad.
KNN: clasifica nuevos planes en los clusters ya entrenados.

Stack: scikit-learn (sklearn) + matplotlib + seaborn.

Uso:
    from src.processors.market_clustering import MarketClusterer
    clusterer = MarketClusterer(n_clusters=4)
    df_clustered = clusterer.fit_and_label(df)
    paths = clusterer.plot_clusters(df_clustered)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import logger

# scikit-learn — importado aquí para lazy-load en contextos sin ML
try:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class MarketClusterer:
    """Segmentación de mercado ISP mediante clustering ML.

    Attributes:
        n_clusters: Número de segmentos a descubrir (default: 4).
        n_neighbors: K para KNN classifier (default: 3).
        FEATURE_COLUMNS: Variables usadas para clustering.
    """

    FEATURE_COLUMNS: list[str] = [
        "velocidad_download_mbps",
        "velocidad_upload_mbps",
        "precio_plan",
        "pys_adicionales",
        "mbps_por_dolar",
    ]

    _TIER_NAMES: list[str] = [
        "🟢 Económico Básico",
        "🔵 Masivo Competitivo",
        "🟡 Alto Valor",
        "🔴 Premium Bundle",
    ]

    def __init__(
        self,
        n_clusters: int = 4,
        n_neighbors: int = 3,
        random_state: int = 42,
    ) -> None:
        """Inicializa el motor de clustering.

        Args:
            n_clusters: Número de segmentos de mercado a descubrir.
            n_neighbors: K para el clasificador KNN.
            random_state: Semilla para reproducibilidad.

        Raises:
            ImportError: Si scikit-learn no está instalado.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn no está instalado. "
                "Ejecuta: pip install scikit-learn"
            )

        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self._scaler = StandardScaler()
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self._knn: KNeighborsClassifier | None = None
        self._is_fitted = False
        self._cluster_name_map: dict[int, str] = {}
        self._log = logger.bind(isp="clusterer", phase="ml_clustering")

    def fit_and_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Entrena KMeans y asigna etiquetas de cluster a cada plan.

        Pipeline:
        1. Validar que existen features necesarias.
        2. Escalar con StandardScaler.
        3. KMeans → descubrir clusters naturales.
        4. KNN → entrenar clasificador para nuevos planes.
        5. Asignar nombres estratégicos por centroide.

        Args:
            df: DataFrame (debe incluir las columnas de FEATURE_COLUMNS).
                Ejecuta StrategicLabeler.enrich() primero para mbps_por_dolar.

        Returns:
            DataFrame + columnas 'cluster_id' y 'cluster_name'.
        """
        df = df.copy()

        # Verificar features disponibles
        missing = [c for c in self.FEATURE_COLUMNS if c not in df.columns]
        if missing:
            self._log.warning(
                "Columnas faltantes para clustering: {}. "
                "Ejecuta StrategicLabeler.enrich() primero.",
                missing,
            )
            df["cluster_id"] = -1
            df["cluster_name"] = "sin_clasificar"
            return df

        valid_mask = df[self.FEATURE_COLUMNS].notna().all(axis=1)
        n_valid = valid_mask.sum()

        if n_valid < self.n_clusters:
            self._log.warning(
                "Solo {} registros válidos para {} clusters. Omitiendo ML.",
                n_valid,
                self.n_clusters,
            )
            df["cluster_id"] = -1
            df["cluster_name"] = "sin_clasificar"
            return df

        X = df.loc[valid_mask, self.FEATURE_COLUMNS].values
        X_scaled = self._scaler.fit_transform(X)

        # KMeans
        cluster_ids = self._kmeans.fit_predict(X_scaled)
        self._is_fitted = True

        # KNN sobre los clusters descubiertos
        k = min(self.n_neighbors, n_valid)
        self._knn = KNeighborsClassifier(n_neighbors=k)
        self._knn.fit(X_scaled, cluster_ids)

        # Asignar al DataFrame
        df.loc[valid_mask, "cluster_id"] = cluster_ids.astype(int)
        df.loc[~valid_mask, "cluster_id"] = -1
        df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)

        # Nombres estratégicos por centroide ordenados por precio
        self._cluster_name_map = self._build_name_map()
        df["cluster_name"] = (
            df["cluster_id"].map(self._cluster_name_map).fillna("sin_clasificar")
        )

        self._log.info(
            "Clustering completado: {} segmentos, {} planes clasificados",
            self.n_clusters,
            n_valid,
        )
        self._log_summary(df)
        return df

    def classify_new_plan(self, features: dict) -> dict:
        """Clasifica un plan nuevo en un cluster existente via KNN.

        Caso de uso real: Cuando un competidor lanza un plan nuevo,
        el sistema lo clasifica automáticamente sin re-entrenar.

        Args:
            features: Dict con las variables del plan.
                Ejemplo: {"velocidad_download_mbps": 300, "precio_plan": 28, ...}

        Returns:
            Dict con cluster_id, cluster_name y confidence.

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado aún.
        """
        if not self._is_fitted or self._knn is None:
            raise RuntimeError(
                "Modelo no entrenado. Ejecuta fit_and_label() primero."
            )

        X_new = [[features.get(c, 0.0) for c in self.FEATURE_COLUMNS]]
        X_scaled = self._scaler.transform(X_new)

        cid = int(self._knn.predict(X_scaled)[0])
        proba = self._knn.predict_proba(X_scaled)[0]

        return {
            "cluster_id": cid,
            "cluster_name": self._cluster_name_map.get(cid, "desconocido"),
            "confidence": round(float(max(proba)), 3),
        }

    def get_cluster_centroids(self) -> pd.DataFrame:
        """Retorna centroides de los clusters en escala original.

        Returns:
            DataFrame con un centroide por fila, o DataFrame vacío si no entrenado.
        """
        if not self._is_fitted:
            return pd.DataFrame()

        raw = self._scaler.inverse_transform(self._kmeans.cluster_centers_)
        return pd.DataFrame(raw, columns=self.FEATURE_COLUMNS).round(2)

    def _build_name_map(self) -> dict[int, str]:
        """Asigna nombres estratégicos a los clusters ordenando por precio."""
        centroids = self.get_cluster_centroids()
        if centroids.empty:
            return {}

        sorted_ids = centroids["precio_plan"].sort_values().index.tolist()
        return {
            cluster_idx: (
                self._TIER_NAMES[rank]
                if rank < len(self._TIER_NAMES)
                else f"Segmento {cluster_idx}"
            )
            for rank, cluster_idx in enumerate(sorted_ids)
        }

    def _log_summary(self, df: pd.DataFrame) -> None:
        for cid in sorted(df["cluster_id"].unique()):
            if cid == -1:
                continue
            sub = df[df["cluster_id"] == cid]
            self._log.info(
                "  {}: {} planes | Vel: {:.0f} Mbps | Precio: ${:.2f} | "
                "{} marcas",
                sub["cluster_name"].iloc[0],
                len(sub),
                sub["velocidad_download_mbps"].mean(),
                sub["precio_plan"].mean(),
                sub["marca"].nunique(),
            )

    # ── Visualizaciones ───────────────────────────────────────────

    def plot_clusters(
        self,
        df: pd.DataFrame,
        output_dir: str = "data/output/charts",
    ) -> list[str]:
        """Genera 3 gráficos de clusters para el notebook y pitch.

        Gráficos:
        1. Scatter: Precio vs Velocidad coloreado por cluster.
        2. Bar: Distribución de ISPs por cluster.
        3. Heatmap: Precio promedio por ISP y segmento de velocidad.

        Args:
            df: DataFrame con cluster_id y cluster_name.
            output_dir: Directorio de salida para las imágenes.

        Returns:
            Lista de paths absolutos de las imágenes generadas.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns

        matplotlib.use("Agg")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files: list[str] = []

        valid = df[df["cluster_id"] != -1].copy()
        _COLORS = {
            "🟢 Económico Básico": "#2ecc71",
            "🔵 Masivo Competitivo": "#3498db",
            "🟡 Alto Valor": "#f39c12",
            "🔴 Premium Bundle": "#e74c3c",
        }

        # ── Gráfico 1: Scatter Precio vs Velocidad ─────────────
        fig, ax = plt.subplots(figsize=(13, 7))
        for name, group in valid.groupby("cluster_name"):
            color = _COLORS.get(str(name), "#95a5a6")
            ax.scatter(
                group["precio_plan"],
                group["velocidad_download_mbps"],
                c=color,
                label=name,
                s=130,
                alpha=0.85,
                edgecolors="white",
                linewidth=1.5,
                zorder=3,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    row["marca"],
                    (row["precio_plan"], row["velocidad_download_mbps"]),
                    xytext=(7, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    alpha=0.75,
                )

        ax.set_xlabel("Precio Mensual (USD sin IVA)", fontsize=12)
        ax.set_ylabel("Velocidad de Descarga (Mbps)", fontsize=12)
        ax.set_title(
            "Mapa de Mercado ISP Ecuador — Segmentación ML (KMeans + KNN)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        p1 = str(out / "market_clusters_scatter.png")
        fig.tight_layout()
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(p1)
        self._log.info("Gráfico 1 generado: {}", p1)

        # ── Gráfico 2: Distribución por Cluster y Marca ────────
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        cluster_brand = (
            valid.groupby(["cluster_name", "marca"])
            .size()
            .unstack(fill_value=0)
        )
        cluster_brand.plot(kind="bar", stacked=True, ax=ax2,
                           colormap="Set2", edgecolor="white", width=0.7)
        ax2.set_title("Distribución de Planes por Segmento y Marca",
                      fontsize=13, fontweight="bold")
        ax2.set_xlabel("Segmento de Mercado", fontsize=11)
        ax2.set_ylabel("Cantidad de Planes", fontsize=11)
        ax2.legend(title="Marca", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.tick_params(axis="x", rotation=20)
        ax2.spines[["top", "right"]].set_visible(False)

        p2 = str(out / "cluster_distribution.png")
        fig2.tight_layout()
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        files.append(p2)
        self._log.info("Gráfico 2 generado: {}", p2)

        # ── Gráfico 3: Heatmap Competitivo ─────────────────────
        if "segmento_velocidad" in valid.columns:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            pivot = valid.pivot_table(
                index="marca",
                columns="segmento_velocidad",
                values="precio_plan",
                aggfunc="mean",
            ).reindex(columns=["basico", "medio", "alto", "premium"])

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn_r",
                ax=ax3,
                linewidths=0.5,
                cbar_kws={"label": "Precio Promedio (USD)"},
                annot_kws={"size": 10},
            )
            ax3.set_title(
                "Heatmap Competitivo: Precio Promedio por ISP y Segmento",
                fontsize=13,
                fontweight="bold",
            )
            ax3.set_xlabel("Segmento de Velocidad", fontsize=11)
            ax3.set_ylabel("ISP", fontsize=11)

            p3 = str(out / "competitive_heatmap.png")
            fig3.tight_layout()
            fig3.savefig(p3, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            files.append(p3)
            self._log.info("Gráfico 3 generado: {}", p3)

        return files
