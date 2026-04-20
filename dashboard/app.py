import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Asegurar importaciones
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.processors.strategic_labels import StrategicLabeler
from src.processors.market_clustering import MarketClusterer
from src.processors.competitive_alerts import CompetitiveAlertEngine

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Netlife | Executive Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. SEGURIDAD C-LEVEL (Autenticación)
# ==========================================
def check_password():
    """Retorna True si el usuario ingresó la contraseña correcta."""
    def password_entered():
        if st.session_state["password"] == "netlife2026": # Credencial Ejecutiva (Demo)
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h2 style='text-align: center; color: #E31837;'>🔒 Acceso Restringido - Nivel Ejecutivo</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Ingrese la contraseña de seguridad (Pass: netlife2026):", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("<h2 style='text-align: center; color: #E31837;'>🔒 Acceso Restringido - Nivel Ejecutivo</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Ingrese la contraseña de seguridad:", type="password", on_change=password_entered, key="password")
            st.error("🚨 Credenciales incorrectas. Intento registrado en el pipeline de seguridad.")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 3. ESTÉTICA CORPORATIVA NETLIFE
# ==========================================
st.markdown("""
<style>
/* Estética corporativa limpia y enfocada en Netlife */
.netlife-title { color: #E31837; font-weight: 800; font-size: 2.2rem; margin-bottom: 0px;}
.netlife-subtitle { color: #555555; font-weight: 400; font-size: 1.1rem; }
.alert-CRITICAL { border-left: 5px solid #E31837; padding: 10px; background-color: rgba(227, 24, 55, 0.1); margin-bottom: 10px; color: black;}
.alert-WARNING { border-left: 5px solid #FFA421; padding: 10px; background-color: rgba(255, 164, 33, 0.1); margin-bottom: 10px; color: black;}
.alert-OPPORTUNITY { border-left: 5px solid #00A859; padding: 10px; background-color: rgba(0, 168, 89, 0.1); margin-bottom: 10px; color: black;}
.alert-STRENGTH { border-left: 5px solid #005BAA; padding: 10px; background-color: rgba(0, 91, 170, 0.1); margin-bottom: 10px; color: black;}
.status-green { color: #00A859; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="netlife-title">🔴 NETLIFE | Inteligencia Competitiva</div>', unsafe_allow_html=True)
st.markdown('<div class="netlife-subtitle">Dashboard Ejecutivo Seguro - C-Level Decision Making</div>', unsafe_allow_html=True)

# ==========================================
# 4. DATA LOADING & ML (Caché Segura)
# ==========================================
@st.cache_resource(ttl=3600)
def load_and_process_data():
    parquet_path = Path("data/output/benchmark_industria.parquet")
    if not parquet_path.exists():
        return pd.DataFrame(), None
    
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df, None

    df["precio_plan"] = pd.to_numeric(df["precio_plan"], errors="coerce")
    df["velocidad_download_mbps"] = pd.to_numeric(df["velocidad_download_mbps"], errors="coerce")

    # ML Pipeline
    labeler = StrategicLabeler()
    df = labeler.enrich(df)
    clusterer = MarketClusterer(n_clusters=4, random_state=42)
    df = clusterer.fit_and_label(df)
    return df, clusterer

with st.spinner("Desencriptando y cargando data pipeline..."):
    df, clusterer_model = load_and_process_data()

if df.empty:
    st.error("🚨 Fallo de Origen de Datos: El pipeline de scraping no ha generado el Parquet.")
    st.stop()

MARCA_OBJETIVO = "Netlife"

# Calcular última actualización en Hora de Ecuador
if 'fecha' in df.columns:
    fecha_max = pd.to_datetime(df['fecha']).max()
else:
    mtime = Path("data/output/benchmark_industria.parquet").stat().st_mtime
    fecha_max = pd.to_datetime(mtime, unit='s', utc=True)

if fecha_max.tzinfo is None:
    fecha_max = fecha_max.tz_localize('UTC')
fecha_ecuador = fecha_max.tz_convert('America/Guayaquil')
fecha_str = fecha_ecuador.strftime('%Y-%m-%d %H:%M:%S')

# Contenedor superior: Resiliencia y Última Actualización
col_status, col_time = st.columns([2, 1])

with col_status:
    if MARCA_OBJETIVO in df['marca'].values:
        st.markdown(f"📡 **Estatus de Extracción:** <span class='status-green'>100% Operativo - Datos asegurados.</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"📡 **Estatus de Extracción:** <span style='color:red;font-weight:bold;'>ALERTA - Usando caché de respaldo.</span>", unsafe_allow_html=True)

with col_time:
    st.markdown(f"<div style='text-align: right; color: #555;'><small>🕒 <b>Último Scrapeo:</b><br>{fecha_str} (UTC-5 Ecuador)</small></div>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 5. ESTRUCTURA DE PESTAÑAS C-LEVEL
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "⚔️ Cara-a-Cara (Netlife vs Competencia)", 
    "🚨 Alertas Estratégicas", 
    "🔮 Simulador de Posicionamiento (KNN)"
])

# ----------------------------------------------------
# PESTAÑA 1: Cara a Cara (Directo al punto)
# ----------------------------------------------------
with tab1:
    st.subheader(f"Comparativa Directa: {MARCA_OBJETIVO} en el Mercado")
    
    netlife_data = df[df['marca'] == MARCA_OBJETIVO]
    comp_data = df[df['marca'] != MARCA_OBJETIVO]
    
    if not netlife_data.empty:
        # Calcular el competidor más barato por segmento
        st.markdown("#### Nuestra Oferta vs La Mejor Oferta Rival (Por Segmento de Velocidad)")
        
        comparison = []
        for seg in sorted(df['segmento_velocidad'].dropna().unique()):
            n_seg = netlife_data[netlife_data['segmento_velocidad'] == seg]
            c_seg = comp_data[comp_data['segmento_velocidad'] == seg]
            
            if not n_seg.empty and not c_seg.empty:
                n_precio = n_seg['precio_plan'].min()
                c_min = c_seg.loc[c_seg['precio_plan'].idxmin()]
                diff = n_precio - c_min['precio_plan']
                status = "🔴 Caros" if diff > 0 else "🟢 Competitivos"
                
                comparison.append({
                    "Segmento": seg,
                    f"Precio {MARCA_OBJETIVO}": f"${n_precio:.2f}",
                    "Rival Más Fuerte": c_min['marca'],
                    "Precio Rival": f"${c_min['precio_plan']:.2f}",
                    "Diferencia": f"${diff:.2f}",
                    "Status": status
                })
        
        comp_df = pd.DataFrame(comparison)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Gráfica Corporativa de Valor
        st.markdown("#### Índice de Valor Entregado (Mbps / Dólar)")
        fig_val = px.bar(
            df.groupby('marca')['mbps_por_dolar'].mean().reset_index().sort_values('mbps_por_dolar', ascending=False),
            x='marca', y='mbps_por_dolar',
            color='marca',
            color_discrete_map={MARCA_OBJETIVO: '#E31837'},
            title="Quién da más velocidad por el mismo dinero (Más alto es mejor)"
        )
        st.plotly_chart(fig_val, use_container_width=True)

# ----------------------------------------------------
# PESTAÑA 2: Alertas Estratégicas
# ----------------------------------------------------
with tab2:
    st.subheader("Panel de Alertas Competitivas")
    engine = CompetitiveAlertEngine(own_brand=MARCA_OBJETIVO, price_threshold=0.10)
    alerts = engine.generate_alerts(df)
    
    if not alerts:
        st.success(f"No se detectaron amenazas en contra de {MARCA_OBJETIVO}.")
    else:
        for alert in alerts:
            if "CRÍTICA" in alert.severity.value:
                css_class = "alert-CRITICAL"
                icon = "🔴"
            elif "ADVERTENCIA" in alert.severity.value:
                css_class = "alert-WARNING"
                icon = "🟠"
            elif "OPPORTUNIDAD" in alert.severity.value:
                css_class = "alert-OPPORTUNITY"
                icon = "🟢" # Green for opportunity
            else:
                css_class = "alert-STRENGTH"
                icon = "🔵"
                
            st.markdown(f"""
            <div class="{css_class}">
                <h4 style="margin-top:0; color: #333;">{icon} {alert.title}</h4>
                <p style="color: #444;"><strong>Detalle:</strong> {alert.description}</p>
                <p style="color: #444;"><strong>Métrica:</strong> <code>{alert.metric}</code> | <strong>Rival:</strong> {alert.competitor} | <strong>Segmento:</strong> {alert.segment}</p>
                <p style="color: #111;">💡 <strong>Acción Recomendada:</strong> <em>{alert.recommendation}</em></p>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------
# PESTAÑA 3: Simulador (K-Nearest Neighbors)
# ----------------------------------------------------
with tab3:
    st.subheader("Simulador de Estrategia de Pricing (IA)")
    st.markdown("Calculadora predictiva. Simula el lanzamiento de un nuevo plan y el motor KNN predecirá automáticamente en qué cuadrante del mercado caerá.")
    
    if clusterer_model:
        with st.form("knn_simulator"):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                sim_speed = st.number_input("Velocidad Descarga (Mbps)", min_value=10, max_value=2000, value=300, step=50)
            with col_s2:
                sim_price = st.number_input("Precio Objetivo ($)", min_value=5.0, max_value=200.0, value=30.0, step=1.0)
            with col_s3:
                sim_streaming = st.number_input("Apps de Streaming Incluidas", min_value=0, max_value=5, value=1)
                
            submitted = st.form_submit_button("Correr Predicción KNN 🚀")
            
        if submitted:
            sim_mbps_usd = sim_speed / sim_price if sim_price > 0 else 0
            nuevo_plan_data = {
                "velocidad_download_mbps": sim_speed,
                "velocidad_upload_mbps": sim_speed / 2,
                "precio_plan": sim_price,
                "pys_adicionales": sim_streaming,
                "mbps_por_dolar": sim_mbps_usd,
            }
            
            resultado = clusterer_model.classify_new_plan(nuevo_plan_data)
            
            st.success(f"**Predicción Completada.** Confianza del algoritmo: {resultado['confidence']:.1%}")
            r1, r2 = st.columns(2)
            r1.metric("Segmento de Mercado Impactado", resultado['cluster_name'])
            r2.metric("Valor del Plan", f"{sim_mbps_usd:.1f} Mbps/$")
            
            st.info(f"Si lanzas este plan a **${sim_price}**, entrarás a competir con los siguientes rivales del segmento **{resultado['cluster_name']}**:")
            
            planes_similares = df[(df['cluster_name'] == resultado['cluster_name']) & (df['marca'] != MARCA_OBJETIVO)]
            if not planes_similares.empty:
                st.dataframe(planes_similares[['marca', 'nombre_plan', 'precio_plan', 'velocidad_download_mbps']].sort_values('precio_plan'), hide_index=True)
            else:
                st.write("Serías el líder solitario en este segmento.")
