"""
scripts/generate_notebook.py
Genera benchmark_industria_notebook.ipynb programáticamente.
Run: python scripts/generate_notebook.py
"""
import json
from pathlib import Path


def md(source: str) -> dict:
    """Crea una celda Markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    """Crea una celda de código."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


CELLS: list[dict] = []

# ══ Celda 1: Portada ══════════════════════════════════════════
CELLS.append(md("""
# Benchmark 360: Inteligencia Competitiva Automatizada
## Mercado de Internet Fijo - Ecuador 2026

---
**Cliente:** Megadatos S.A. / Netlife  
**Equipo:** FerQode Dev Team  
**Fecha:** Abril 2026

---

### El Problema en Numeros
> En Ecuador existen **+1,400 ISPs** compitiendo en un mercado de precio altamente
> volatil. El equipo de Pricing de Netlife tardaba **semanas** en detectar cambios
> en la oferta de competidores clave como Claro, Xtrim o CNT.

### La Solucion
Un **Data Pipeline Enterprise** que ejecuta diariamente y entrega en menos de
**24 horas** el 100% de la oferta comercial vigente de los principales ISPs
competidores, incluyendo precios en imagenes y terminos y condiciones completos.
"""))

# ══ Celda 2: Arquitectura ════════════════════════════════════
CELLS.append(md("""
## Arquitectura del Pipeline

```
INTERNET (paginas web de 8 ISPs)
          |
          v
+-------------------------------------------------------+
|              CAPA 1: EXTRACCION                       |
|  ISPStrategy --> BaseISPScraper --> CookieHandler      |
|  TCHTMLScraper --> Terminos y Condiciones (sin IA)     |
+-------------------------------------------------------+
          |                    |
     Texto HTML (37KB)   7 Tiles PNG
          |                    |
          v                    v
+-------------------------------------------------------+
|              CAPA 2: INTELIGENCIA ARTIFICIAL          |
|  LLMProcessor            VisionProcessor              |
|  gemini-2.5-flash   ->   gemini-3.1-pro (Tier 1)     |
|  flash-lite          ->   pixtral-large  (Tier 2)     |
|  gpt-4o-mini         ->   flash-lite     (Tier 3)     |
|                      ->   gpt-4o-mini    (Tier 4)     |
|  + Guardrails (Anti Prompt Injection)                 |
+-------------------------------------------------------+
          |
          v
+-------------------------------------------------------+
|              CAPA 3: VALIDACION Y STORAGE             |
|  Pydantic V2 --> PlanNormalizer --> Parquet            |
|  (30 campos)    (snake_case, IVA, dedup)  (por fecha) |
+-------------------------------------------------------+
          |
          v
+-------------------------------------------------------+
|              CAPA 4: MACHINE LEARNING                 |
|  StrategicLabeler --> MarketClusterer --> Alertas      |
|  (10 KPIs)          (KMeans + KNN)    (4 niveles)     |
+-------------------------------------------------------+
```

**Stack:** Python 3.12 | uv | Playwright | google-genai | Mistral | Pydantic V2 | scikit-learn | Pandas | PyArrow
"""))

# ══ Celda 3: Setup ═══════════════════════════════════════════
CELLS.append(code("""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json, ast, sys, os

# Agregar raiz del proyecto al path
sys.path.insert(0, os.getcwd())

plt.rcParams.update({
    "figure.figsize": (14, 7),
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

ISP_COLORS = {
    "Netlife":  "#E31837",
    "Claro":    "#DA291C",
    "Xtrim":    "#00A859",
    "CNT":      "#005BAA",
    "Ecuanet":  "#FF6B00",
    "Fibramax": "#8B00FF",
    "Alfanet":  "#00B4D8",
    "Celerity": "#2D6A4F",
}

Path("data/output/charts").mkdir(parents=True, exist_ok=True)
print("Librerias cargadas correctamente")
print(f"Fecha de analisis: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
"""))

# ══ Celda 4: Carga del Parquet ═══════════════════════════════
CELLS.append(code("""
PARQUET_PATH = Path("data/output/benchmark_industria.parquet")

def safe_parse_dict(v):
    \"\"\"Parsea diccionarios de forma segura.\"\"\"
    if pd.isna(v) or str(v) in ("", "{}", "None"):
        return {}
    try:
        return json.loads(v)
    except Exception:
        try:
            return ast.literal_eval(str(v))
        except Exception:
            return {}

def safe_parse_list(v):
    \"\"\"Parsea listas de forma segura.\"\"\"
    if pd.isna(v) or str(v) in ("", "[]", "None"):
        return []
    try:
        r = ast.literal_eval(str(v))
        return r if isinstance(r, list) else []
    except Exception:
        return []

df = pd.read_parquet(PARQUET_PATH)
df["pys_adicionales_detalle"] = df["pys_adicionales_detalle"].apply(safe_parse_dict)
if "sectores" in df.columns:
    df["sectores"] = df["sectores"].apply(safe_parse_list)
df["fecha"] = pd.to_datetime(df["fecha"])
df["precio_plan"] = pd.to_numeric(df["precio_plan"], errors="coerce")
df["velocidad_download_mbps"] = pd.to_numeric(df["velocidad_download_mbps"], errors="coerce")

print("=" * 60)
print("  BENCHMARK 360 - Dataset Cargado")
print("=" * 60)
print(f"  Registros:          {len(df)}")
print(f"  ISPs:               {df['marca'].nunique()} ({', '.join(sorted(df['marca'].unique()))})")
print(f"  Columnas:           {len(df.columns)}")
print(f"  Fecha extraccion:   {df['fecha'].max().strftime('%Y-%m-%d %H:%M')}")
print(f"  Rango precios:      ${df['precio_plan'].min():.2f} - ${df['precio_plan'].max():.2f}")
print(f"  Rango velocidades:  {df['velocidad_download_mbps'].min():.0f} - {df['velocidad_download_mbps'].max():.0f} Mbps")
print("=" * 60)

df[["marca", "nombre_plan", "velocidad_download_mbps", "precio_plan",
    "tecnologia", "pys_adicionales"]].head(10)
"""))

# ══ Celda 5: MD Calidad ══════════════════════════════════════
CELLS.append(md("""
## Seccion 1: Validacion de Calidad de Datos

Antes de cualquier analisis auditamos completitud y consistencia del dataset.
El pipeline usa **Pydantic V2** para garantizar que cada campo cumple las reglas
de negocio definidas en el schema (30+ campos, tipos estrictos, rangos validos).

**Metricas clave:**
- Completitud por columna (% valores no-nulos)
- Consistencia del calculo de descuentos (precio_plan vs precio_plan_descuento)
- Unicidad de planes por ISP (deduplicacion por nombre)
"""))

# ══ Celda 6: Auditoria calidad ══════════════════════════════
CELLS.append(code("""
columnas_criticas = [
    "marca", "nombre_plan", "velocidad_download_mbps",
    "precio_plan", "tecnologia", "terminos_condiciones",
    "pys_adicionales", "descuento",
]

# Filtrar solo las columnas que existen en el dataset
columnas_presentes = [c for c in columnas_criticas if c in df.columns]
completitud = (df.notna().mean() * 100).sort_values(ascending=False).round(1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Grafico 1: Completitud de campos criticos
comp_criticas = completitud[columnas_presentes].sort_values()
colors_bar = ["#E31837" if v < 80 else "#00A859" for v in comp_criticas]
comp_criticas.plot(kind="barh", ax=axes[0], color=colors_bar, edgecolor="white")
axes[0].set_title("Completitud - Campos Criticos de Negocio", fontweight="bold")
axes[0].set_xlabel("% de registros con dato")
axes[0].axvline(x=80, color="red", linestyle="--", alpha=0.7, label="Meta >80%")
axes[0].legend()
for i, v in enumerate(comp_criticas):
    axes[0].text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)

# Grafico 2: Planes por ISP
registros_isp = df["marca"].value_counts()
registros_isp.plot(
    kind="bar", ax=axes[1],
    color=[ISP_COLORS.get(isp, "#666") for isp in registros_isp.index],
    edgecolor="white",
)
axes[1].set_title("Planes Extraidos por ISP", fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)
for i, v in enumerate(registros_isp):
    axes[1].text(i, v + 0.1, str(v), ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Benchmark 360 - Auditoria de Calidad del Dataset",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/output/charts/01_calidad_datos.png", bbox_inches="tight", dpi=150)
plt.show()

score = completitud[columnas_presentes].mean()
print(f"\\nScore de Completitud Global: {score:.1f}%")
print(f"Criterio >80%: {'CUMPLIDO' if score >= 80 else 'EN PROGRESO'}")
"""))

# ══ Celda 7: MD Precios ══════════════════════════════════════
CELLS.append(md("""
## Seccion 2: Analisis Competitivo de Precios

El corazon del caso de negocio. El equipo de Pricing de Netlife necesita saber:

1. **Quien es el mas barato?** - Amenaza directa al volumen de clientes
2. **Quien da mas por el mismo precio?** - Amenaza a la percepcion de valor
3. **Quien usa descuentos agresivos?** - Estrategia de captacion temporal

> **Insight para Netlife:** Un descuento del 30% en los primeros 3 meses
> equivale a bajar el precio real un 7.5% anual. El pipeline detecta esto automaticamente.
"""))

# ══ Celda 8: Analisis precios ════════════════════════════════
CELLS.append(code("""
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Grafico 1: Precio promedio por ISP
precio_por_isp = df.groupby("marca")["precio_plan"].agg(["mean", "min", "max"]).sort_values("mean")
ax1 = axes[0, 0]
bars = ax1.barh(
    precio_por_isp.index, precio_por_isp["mean"],
    color=[ISP_COLORS.get(isp, "#999") for isp in precio_por_isp.index],
    edgecolor="white", height=0.6,
)
ax1.errorbar(
    precio_por_isp["mean"], range(len(precio_por_isp)),
    xerr=[
        precio_por_isp["mean"] - precio_por_isp["min"],
        precio_por_isp["max"] - precio_por_isp["mean"],
    ],
    fmt="none", color="#333", capsize=4, linewidth=2,
)
ax1.set_title("Precio Promedio por ISP (sin IVA)\\n[barra=promedio, linea=rango]")
ax1.set_xlabel("Precio mensual USD (sin IVA)")
for bar, val in zip(bars, precio_por_isp["mean"]):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
             f"${val:.2f}", va="center", fontsize=9, fontweight="bold")

# Grafico 2: Violin plot de distribucion de precios
ax2 = axes[0, 1]
isps_ord = df.groupby("marca")["precio_plan"].median().sort_values().index.tolist()
data_violin = [df[df["marca"] == isp]["precio_plan"].dropna().values for isp in isps_ord]
if all(len(d) > 0 for d in data_violin):
    parts = ax2.violinplot(data_violin, positions=range(len(isps_ord)), showmedians=True)
    for pc, isp in zip(parts["bodies"], isps_ord):
        pc.set_facecolor(ISP_COLORS.get(isp, "#999"))
        pc.set_alpha(0.7)
ax2.set_xticks(range(len(isps_ord)))
ax2.set_xticklabels(isps_ord, rotation=45, ha="right")
ax2.set_title("Distribucion de Precios por ISP")
ax2.set_ylabel("Precio mensual USD (sin IVA)")

# Grafico 3: Heatmap precio por velocidad
ax3 = axes[1, 0]
df["velocidad_bin"] = pd.cut(
    df["velocidad_download_mbps"],
    bins=[0, 200, 400, 600, 800, 1000, float("inf")],
    labels=["<=200", "201-400", "401-600", "601-800", "801-1000", ">1000"],
)
pivot = df.pivot_table(
    values="precio_plan", index="marca", columns="velocidad_bin", aggfunc="mean"
)
sns.heatmap(pivot, ax=ax3, cmap="RdYlGn_r", annot=True, fmt=".1f",
            cbar_kws={"label": "Precio USD"}, linewidths=0.5)
ax3.set_title("Heatmap: Precio por ISP y Velocidad\\n[verde=barato, rojo=caro]")
ax3.set_xlabel("Rango velocidad (Mbps)")

# Grafico 4: Descuentos promocionales
ax4 = axes[1, 1]
if "descuento" in df.columns:
    df_d = df[df["descuento"].notna() & (df["descuento"] > 0)].copy()
    df_d["descuento_pct"] = df_d["descuento"] * 100
    if len(df_d) > 0:
        desc_isp = df_d.groupby("marca").agg(
            desc_prom=("descuento_pct", "mean"),
            n_planes=("nombre_plan", "count"),
        ).sort_values("desc_prom", ascending=False)
        bars4 = ax4.bar(
            desc_isp.index, desc_isp["desc_prom"],
            color=[ISP_COLORS.get(i, "#999") for i in desc_isp.index],
            edgecolor="white",
        )
        ax4.set_title("Descuentos Promocionales por ISP")
        ax4.set_ylabel("Descuento promedio (%)")
        ax4.tick_params(axis="x", rotation=45)
        for bar, (idx, row) in zip(bars4, desc_isp.iterrows()):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{row['desc_prom']:.1f}%", ha="center", fontsize=9)
    else:
        ax4.text(0.5, 0.5, "Sin datos de descuento",
                 ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Descuentos Promocionales")
else:
    ax4.text(0.5, 0.5, "Columna descuento no disponible",
             ha="center", va="center", transform=ax4.transAxes)

plt.suptitle("Benchmark 360 - Analisis Competitivo de Precios",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("data/output/charts/02_analisis_precios.png", bbox_inches="tight", dpi=150)
plt.show()
"""))

# ══ Celda 9: Indice de Valor ═════════════════════════════════
CELLS.append(code("""
df_valor = df[
    df["precio_plan"].notna()
    & df["velocidad_download_mbps"].notna()
    & (df["precio_plan"] > 0)
].copy()
df_valor["mbps_por_dolar"] = (
    df_valor["velocidad_download_mbps"] / df_valor["precio_plan"]
).round(2)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Scatter: Velocidad vs Precio
ax1 = axes[0]
for isp, grp in df_valor.groupby("marca"):
    ax1.scatter(
        grp["precio_plan"], grp["velocidad_download_mbps"],
        c=ISP_COLORS.get(isp, "#999"), s=120, alpha=0.8, label=isp,
        edgecolors="white", linewidth=1.5, zorder=3,
    )
    for _, row in grp.iterrows():
        nm = str(row["nombre_plan"])[:14]
        ax1.annotate(nm, (row["precio_plan"], row["velocidad_download_mbps"]),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=6.5, alpha=0.8)

avg_p = df_valor["precio_plan"].mean()
avg_v = df_valor["velocidad_download_mbps"].mean()
ax1.axvline(avg_p, color="gray", linestyle="--", alpha=0.5,
            label=f"Precio prom ${avg_p:.1f}")
ax1.axhline(avg_v, color="gray", linestyle=":", alpha=0.5,
            label=f"Vel prom {avg_v:.0f} Mbps")
ax1.text(df_valor["precio_plan"].min() + 1,
         df_valor["velocidad_download_mbps"].max() * 0.93,
         "MEJOR VALOR\\n(Rapido y Barato)", fontsize=8, color="green", alpha=0.8)
ax1.text(df_valor["precio_plan"].max() * 0.70,
         df_valor["velocidad_download_mbps"].min() + 50,
         "PEOR VALOR\\n(Lento y Caro)", fontsize=8, color="red", alpha=0.8)
ax1.set_title("Mapa Competitivo: Velocidad vs Precio")
ax1.set_xlabel("Precio mensual USD (sin IVA)")
ax1.set_ylabel("Velocidad de descarga (Mbps)")
ax1.legend(loc="lower right", fontsize=8, ncol=2)

# Bar: Mbps por dolar
ax2 = axes[1]
indice = df_valor.groupby("marca")["mbps_por_dolar"].mean().sort_values(ascending=False)
mkt_avg = indice.mean()
bars = ax2.bar(
    indice.index, indice.values,
    color=[ISP_COLORS.get(i, "#999") for i in indice.index],
    edgecolor="white", width=0.6,
)
ax2.axhline(mkt_avg, color="black", linestyle="--", alpha=0.6,
            label=f"Promedio mercado: {mkt_avg:.1f} Mbps/$")
ax2.set_title("Indice de Valor Competitivo\\n[Mbps por dolar - mayor es mejor]")
ax2.set_ylabel("Mbps por dolar (USD)")
ax2.tick_params(axis="x", rotation=45)
ax2.legend()
for bar, val in zip(bars, indice.values):
    color = "green" if val > mkt_avg else "red"
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", fontsize=11, fontweight="bold", color=color)

plt.suptitle("Benchmark 360 - Indice de Valor (Mbps/$)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("data/output/charts/03_indice_valor.png", bbox_inches="tight", dpi=150)
plt.show()

print("\\nRANKING: Mejor Valor (Mbps por Dolar)")
print("-" * 45)
for rank, (isp, val) in enumerate(indice.items(), 1):
    prefix = f"{rank}."
    diff = val - mkt_avg
    direction = "+" if diff > 0 else ""
    print(f"  {prefix} {isp:<12} {val:>6.1f} Mbps/$  ({direction}{diff:.1f} vs promedio)")
"""))

# ══ Celda 10: Servicios OTT ═════════════════════════════════
CELLS.append(code("""
todos_servicios = []
for _, row in df.iterrows():
    detalle = row.get("pys_adicionales_detalle", {})
    if isinstance(detalle, dict):
        for svc, info in detalle.items():
            categoria = info.get("categoria", "otros") if isinstance(info, dict) else "otros"
            meses = info.get("meses") if isinstance(info, dict) else None
            todos_servicios.append({
                "isp": row["marca"],
                "servicio": svc,
                "categoria": categoria,
                "meses": meses,
            })

df_svc = pd.DataFrame(todos_servicios)

if len(df_svc) == 0:
    print("Sin datos de servicios adicionales en el dataset.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_svc = df_svc["servicio"].value_counts().head(8)
    top_svc.plot(kind="barh", ax=axes[0], color="#E31837", edgecolor="white")
    axes[0].set_title("Top Servicios Adicionales mas Ofrecidos")
    axes[0].set_xlabel("Numero de planes que lo incluyen")

    cat_counts = df_svc["categoria"].value_counts()
    axes[1].pie(
        cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%",
        colors=plt.cm.Set3.colors[:len(cat_counts)], startangle=90,
    )
    axes[1].set_title("Distribucion por Categoria")

    plt.suptitle("Benchmark 360 - Servicios Adicionales (OTT/Streaming)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/output/charts/04_servicios_ott.png", bbox_inches="tight", dpi=150)
    plt.show()

    print("\\nPresencia de Servicios por ISP:")
    pivot_svc = df_svc.pivot_table(
        values="meses", index="isp", columns="servicio", aggfunc="first"
    ).notna()
    print(pivot_svc.replace({True: "Incluido", False: ""}).to_string())
"""))

# ══ Celda 11: Resumen ejecutivo ══════════════════════════════
CELLS.append(code("""
resumen = df.groupby("marca").agg(
    planes=("nombre_plan", "count"),
    precio_min=("precio_plan", "min"),
    precio_max=("precio_plan", "max"),
    precio_prom=("precio_plan", "mean"),
    vel_max=("velocidad_download_mbps", "max"),
    vel_prom=("velocidad_download_mbps", "mean"),
    con_descuento=("descuento", lambda x: (x > 0).sum() if x.notna().any() else 0),
    desc_max=("descuento", lambda x: round(x.max() * 100, 1) if x.notna().any() and x.max() > 0 else 0),
    ott=("pys_adicionales", "sum"),
    con_tc=("terminos_condiciones", lambda x: x.notna().sum()),
).round(2)

print("=" * 85)
print("  RESUMEN EJECUTIVO - BENCHMARK COMPETITIVO ISPs ECUADOR")
print(f"  Extraccion: {df['fecha'].max().strftime('%Y-%m-%d %H:%M')}")
print("=" * 85)

disp = resumen.copy()
for c in ["precio_min", "precio_max", "precio_prom"]:
    disp[c] = disp[c].apply(lambda x: f"${x:.2f}")
disp["vel_max"] = disp["vel_max"].apply(lambda x: f"{x:.0f} Mbps")
disp["desc_max"] = disp["desc_max"].apply(lambda x: f"{x:.1f}%")
disp.columns = [
    "Planes", "Min", "Max", "Prom", "Vel Max", "Vel Prom",
    "c/Dscto", "Dscto Max", "OTT", "Con T&C",
]
print(disp.to_string())

print("\\nALERTAS COMPETITIVAS DETECTADAS:")
precio_netlife = (
    df[df["marca"] == "Netlife"]["precio_plan"].mean()
    if "Netlife" in df["marca"].values else None
)
for isp, row in resumen.iterrows():
    if isp == "Netlife":
        continue
    if precio_netlife and row["precio_prom"] < precio_netlife * 0.85:
        diff = (precio_netlife - row["precio_prom"]) / precio_netlife * 100
        print(f"  [ALERTA] {isp} tiene precios {diff:.1f}% mas bajos que Netlife")
    if row["desc_max"] > 20:
        print(f"  [ALERTA] {isp} ofrece descuentos de hasta {row['desc_max']}% - campana agresiva")
    if row["vel_max"] > 800:
        print(f"  [INFO]   {isp} ofrece hasta {row['vel_max']} - presion en segmento premium")
"""))

# ══ Celda 12: MD ML Segmentacion ════════════════════════════
CELLS.append(md("""
## Seccion 3: Segmentacion de Mercado con Machine Learning

### KMeans + KNN: Del Analisis Humano a la Inteligencia Artificial

Aplicamos **KMeans** para descubrir los segmentos naturales del mercado
(sin sesgos previos) y **KNN** para clasificar automaticamente cualquier
plan nuevo que lance un competidor.

| Segmento | Descripcion | Perfil tipico |
|---|---|---|
| Economico Basico | Planes entry-level, precio bajo | Familias price-sensitive |
| Masivo Competitivo | El campo de batalla principal | Teletrabajo, streaming HD |
| Alto Valor | Velocidades altas, precio justificado | Gamers, multi-dispositivo |
| Premium Bundle | Gigabit + paquetes completos | Power users, empresas |

> *Un analista humano tomaria semanas en identificar estos patrones. 
> Nuestro modelo lo hace en milisegundos.*
"""))

# ══ Celda 13: ML Clustering ═════════════════════════════════
CELLS.append(code("""
from src.processors.strategic_labels import StrategicLabeler
from src.processors.market_clustering import MarketClusterer
from src.processors.competitive_alerts import CompetitiveAlertEngine
from IPython.display import Image, display as ipy_display

# Paso 1: Enriquecer con etiquetas estrategicas (10 KPIs)
if "mbps_por_dolar" not in df.columns:
    labeler = StrategicLabeler()
    df = labeler.enrich(df)
    print("StrategicLabeler: +10 columnas estrategicas agregadas")

# Paso 2: Clustering ML
clusterer = MarketClusterer(n_clusters=4, random_state=42)
df = clusterer.fit_and_label(df)

print("\\nSegmentos de Mercado Descubiertos por ML:")
print("=" * 55)
for cluster in sorted(df[df["cluster_id"] != -1]["cluster_name"].unique()):
    sub = df[df["cluster_name"] == cluster]
    print(f"\\n  {cluster}:")
    print(f"    Planes: {len(sub)} | Marcas: {', '.join(sorted(sub['marca'].unique()))}")
    print(f"    Velocidad media: {sub['velocidad_download_mbps'].mean():.0f} Mbps")
    print(f"    Precio promedio: ${sub['precio_plan'].mean():.2f}")
    print(f"    Mbps/dolar:      {sub['mbps_por_dolar'].mean():.1f}")

# Generar graficos
chart_paths = clusterer.plot_clusters(df, output_dir="data/output/charts")
scatter = "data/output/charts/market_clusters_scatter.png"
if os.path.exists(scatter):
    print("\\nMapa de Clusters del Mercado:")
    ipy_display(Image(filename=scatter))
"""))

# ══ Celda 14: KNN Clasificacion ══════════════════════════════
CELLS.append(code("""
# KNN: Clasificar un plan hipotetico nuevo
# Caso: "Claro lanza un plan de 400 Mbps a $32. En que segmento cae?"

nuevo_plan = {
    "velocidad_download_mbps": 400,
    "velocidad_upload_mbps": 200,
    "precio_plan": 32.0,
    "pys_adicionales": 1,
    "mbps_por_dolar": 12.5,
}

resultado = clusterer.classify_new_plan(nuevo_plan)

print("KNN: Clasificacion de Plan Hipotetico")
print("=" * 55)
print(f"  Plan simulado:      Claro 400 Mbps / $32 mensual")
print(f"  Segmento detectado: {resultado['cluster_name']}")
print(f"  Confianza:          {resultado['confidence']:.1%}")
print(f"")
print(f"  Interpretacion: Este plan competiria directamente")
print(f"  con los planes de Netlife en el segmento")
print(f"  '{resultado['cluster_name']}'.")
print(f"")
print(f"  Accion sugerida: Revisar propuesta de Netlife en ese segmento.")
"""))

# ══ Celda 15: Alertas Competitivas ══════════════════════════
CELLS.append(code("""
# Motor de Alertas Competitivas
engine = CompetitiveAlertEngine(own_brand="Netlife", price_threshold=0.10)
alerts = engine.generate_alerts(df)

# Resumen ejecutivo
engine.print_executive_summary(alerts)

# Tabla de alertas
alerts_df = engine.to_dataframe(alerts)
if not alerts_df.empty:
    display(
        alerts_df[["severidad", "titulo", "metrica", "recomendacion"]]
        .style.set_caption("Alertas Competitivas - Benchmark 360")
    )

# Mostrar heatmap competitivo
heatmap = "data/output/charts/competitive_heatmap.png"
if os.path.exists(heatmap):
    print("\\nHeatmap Competitivo por Segmento:")
    ipy_display(Image(filename=heatmap))
"""))

# ══ Celda 16: Conclusiones ══════════════════════════════════
CELLS.append(md("""
## Seccion 4: Conclusiones y Recomendaciones Estrategicas

### ROI del Sistema Benchmark 360

| Concepto | Antes (Manual) | Con Benchmark 360 |
|---|---|---|
| Tiempo de monitoreo | ~8 horas/semana | **< 10 minutos/dia** |
| Frecuencia de actualizacion | Semanal o quincenal | **Diaria automatizada** |
| Cobertura de ISPs | 3-4 ISPs principales | **8 ISPs (100% del mercado clave)** |
| Costo operativo API | N/A | **~$0.50/dia con cache** |
| Deteccion de cambios | 2-3 semanas | **< 24 horas** |
| Formato de datos | Excel manual | **Parquet analitico + ML** |

### Hallazgos Principales

**1. Presion en el segmento de precio bajo (<$25/mes)**  
Xtrim, Fibramax y Alfanet concentran su oferta en velocidades altas a precios 
agresivos, capturando el segmento price-sensitive con Mbps/$ superior al promedio.

**Recomendacion:** Evaluar un plan de entrada $15-20 con minimo 200 Mbps.

**2. El streaming como diferenciador clave**  
Xtrim y Claro incluyen Disney+, Max y otros OTT como beneficio temporal,
aumentando la percepcion de valor sin reducir el precio base.

**Recomendacion:** Agregar al menos 1 servicio OTT por plan premium.

**3. Oportunidad en el segmento premium (>500 Mbps)**  
Ningun ISP se posiciona claramente como "el premium de Ecuador".

**Recomendacion:** Oportunidad para Netlife de ocupar ese posicionamiento 
con SLA garantizado + soporte 24/7 + OTT bundle.

### Proximos Pasos
1. Ampliar a 50+ ISPs usando el ISP Registry Pattern
2. Dashboard en tiempo real conectado al Parquet (Tableau/Power BI)
3. Alertas automaticas via Slack/Teams cuando un competidor cambia precios >5%
4. Analisis de series de tiempo para detectar patrones estacionales

---
**Generado por Benchmark 360** | Pipeline Enterprise | Hackathon Netlife 2026  
*De datos a decisiones en minutos.*
"""))


# ══ Generar el .ipynb ════════════════════════════════════════

def generate_notebook() -> None:
    """Genera el archivo .ipynb con todas las celdas definidas."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "cells": CELLS,
    }

    out = Path("benchmark_industria_notebook.ipynb")
    out.write_text(
        json.dumps(nb, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size_kb = out.stat().st_size / 1024
    print(f"Notebook generado: {out} ({size_kb:.1f} KB, {len(CELLS)} celdas)")


if __name__ == "__main__":
    generate_notebook()
