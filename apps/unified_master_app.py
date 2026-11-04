"""
================================================================================
RAINWISE AI — UNIFIED DISASTER INTELLIGENCE & XAI COMMAND CENTER
================================================================================
Integrates:
  1. Real-Time Telemetry (River Gauges, Satellite ERA5 Precipitation, Weather)
  2. Rapid ML Baseline (Logistic Regression Balanced Classifier)
  3. Sequential 2-Stage DL Pipeline (TabTransformer -> FloodDNN)
  4. Explainable AI with PyTorch Captum (Integrated Gradients & Axiomatic Attribution)
  5. Gujarat Municipal Benchmarks (Surat, Ahmedabad, Vadodara, Saputara, Rajkot)
================================================================================
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"
import sys
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# Path setup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.utils.realtime_data import get_all_realtime, get_pipeline_status
from model_trainingDL.models import TabTransformer
from xai_dl.explainable_flood_model import ExplainableFloodModel, FloodDNN
from xai_dl.visualize_attributions import plot_attribution_bars, plot_waterfall

# ==============================================================================
# 1. PAGE CONFIG & HIGH-END CYBERPUNK STYLING
# ==============================================================================
st.set_page_config(
    page_title="RAINWISE AI — Master Command Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0b1120;
        color: #f8fafc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 20px;
        font-weight: 600;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
    }
    .card-box {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-yellow {
        background: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #2563eb 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        height: 3em;
        font-size: 15px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.90;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🌊 RAINWISE AI — Master Disaster Intelligence Center")
st.markdown("##### Real-Time Telemetry • Cascading Deep Learning • PyTorch Captum XAI")

# ==============================================================================
# 2. MODEL & METADATA LOADER
# ==============================================================================
@st.cache_resource
def load_all_models():
    device = torch.device("cpu")
    
    # 1. ML Baseline Model (Logistic Regression Pipeline)
    ml_model = joblib.load("models/flood_model.pkl") if os.path.exists("models/flood_model.pkl") else None

    # 2. DL Stage 1: TabTransformer (Rainfall Regression)
    tab_trans = TabTransformer(input_dim=7, depth=3)
    if os.path.exists("DLmodels/tab_transformer_rainfall.pth"):
        tab_trans.load_state_dict(torch.load("DLmodels/tab_transformer_rainfall.pth", map_location=device))
    tab_trans.eval()
    tab_scaler = joblib.load("DLmodels/tab_transformer_scaler.pkl") if os.path.exists("DLmodels/tab_transformer_scaler.pkl") else None

    # 3. DL Stage 2 / Captum XAI: FloodDNN Model
    xai_explainer = ExplainableFloodModel(
        model_path="xai_dl/models/flood_dnn_xai.pth",
        scaler_path="xai_dl/models/scaler_xai.pkl",
        device=device
    )

    # 4. Topographic Reference
    cities_df = pd.read_csv("data/config/gujarat_cities.csv")
    cities_df.columns = cities_df.columns.str.lower()
    
    river_df = pd.read_csv("data/processed/gujarat_river_distance.csv")
    elev_df = pd.read_csv("data/processed/gujarat_elevation.csv")
    river_df.columns = river_df.columns.str.lower()
    elev_df.columns = elev_df.columns.str.lower()

    return ml_model, (tab_trans, tab_scaler), xai_explainer, cities_df, river_df, elev_df

ml_model, (tab_trans, tab_scaler), xai_explainer, cities_df, river_df, elev_df = load_all_models()

def find_nearest_gis(df, lat, lon, col):
    d = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return float(df.loc[d.idxmin()][col])

# ==============================================================================
# 3. SIDEBAR: JURISDICTION SELECTION & SENSOR TELEMETRY
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=70)
    st.header("Municipal Operations")
    
    city_list = sorted(cities_df["city"].unique().tolist())
    default_idx = city_list.index("Ahmedabad") if "Ahmedabad" in city_list else 0
    selected_city = st.selectbox("🎯 Select Jurisdiction", city_list, index=default_idx)
    
    city_row = cities_df[cities_df["city"] == selected_city].iloc[0]
    c_lat = float(city_row["lat"])
    c_lon = float(city_row["lon"])
    
    c_elev = find_nearest_gis(elev_df, c_lat, c_lon, "elevation")
    c_river = find_nearest_gis(river_df, c_lat, c_lon, "river_distance")
    
    # Population estimation
    pop_map = {"Ahmedabad": 8600000, "Surat": 6100000, "Vadodara": 2200000, "Rajkot": 1800000, "Saputara": 50000}
    c_pop = float(pop_map.get(selected_city, 1200000))
    
    st.divider()
    st.subheader("📍 Geospatial Profile")
    st.caption(f"**Coordinates:** {c_lat:.4f}°N, {c_lon:.4f}°E")
    st.caption(f"**SRTM Elevation:** {c_elev:.0f} meters")
    st.caption(f"**River Proximity:** {c_river:.0f} meters")
    st.caption(f"**Census Population:** {c_pop/1e6:.2f} Million")
    
    st.divider()
    # Telemetry Status
    pipe_status = get_pipeline_status()
    st.subheader("📡 Real-Time Data Feeds")
    st.caption(f"🛰️ Satellite (ERA5): {'🟢 Online' if pipe_status['satellite_log_exists'] else '🔴 Offline'}")
    st.caption(f"🌊 River Gauges: {'🟢 Online' if pipe_status['river_log_exists'] else '🔴 Offline'}")
    st.caption(f"⚡ Weather Stations: {'🟢 Online' if pipe_status['weather_log_exists'] else '⚪ Standby'}")

# Fetch real-time data for chosen city
rt_telemetry = get_all_realtime(selected_city)

# ==============================================================================
# 4. MAIN TABS INTERFACE
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ Real-Time Telemetry & ML Alert",
    "🌊 Cascading DL Pipeline (TabTransformer ➔ FloodDNN)",
    "🧠 Explainable AI (Captum Integrated Gradients)",
    "📍 Gujarat Municipal Benchmarks"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: REAL-TIME TELEMETRY & RAPID ML ALERT (LOGISTIC REGRESSION)
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🛰️ Live Hydrological Ingestion & Fast ML Baseline")
    st.caption("Engine: **Logistic Regression (Class-Balanced Pipeline)** with Automated Geospatial Ingestion")
    
    c1, c2, c3, c4 = st.columns(4)
    sat_rain = rt_telemetry["satellite"]["rainfall_mm"] if rt_telemetry.get("satellite") else 0.0
    river_lvl = rt_telemetry["river"]["level"] if rt_telemetry.get("river") else 0.0
    river_stat = rt_telemetry["river"]["status"] if rt_telemetry.get("river") else "Normal"
    river_name = rt_telemetry["river"]["river"] if rt_telemetry.get("river") else "Nearest Channel"
    
    c1.metric("🛰️ Satellite Rain (24h)", f"{sat_rain:.1f} mm")
    c2.metric("🏞️ River Level", f"{river_lvl:.1f} m³/s", delta=f"{river_stat}")
    c3.metric("🏔️ Topographic Elevation", f"{c_elev:.0f} m")
    c4.metric("📏 Distance to River", f"{c_river:.0f} m")
    
    if river_stat == "Above Danger":
        st.error(f"🚨 **CRITICAL RIVER WARNING:** {river_name} is currently ABOVE DANGER limit ({river_lvl:.1f} m³/s)!")
    elif river_stat == "Warning":
        st.warning(f"⚠️ **RIVER ADVISORY:** {river_name} is at WARNING stage ({river_lvl:.1f} m³/s).")

    st.divider()
    st.markdown("#### ⚡ Run Immediate Classification Alert")
    
    col_in1, col_in2 = st.columns([1, 1])
    with col_in1:
        use_live_rain = st.checkbox("Use Live Satellite Reading as Precipitation Input", value=True)
        if use_live_rain:
            eval_rain = sat_rain
            st.info(f"Input Rainfall bound to Live Satellite Telemetry: **{eval_rain:.1f} mm**")
        else:
            eval_rain = st.slider("Simulate Precipitation Intensity (mm)", 0.0, 250.0, 65.0, step=5.0)

        threshold = st.slider("Civic Alert Threshold", 0.1, 0.9, 0.5, step=0.05)

    with col_in2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("RUN ML FLOOD CLASSIFICATION", key="btn_ml"):
            feat = np.array([[eval_rain, c_elev, c_river, c_lat, c_lon]])
            ml_prob = float(ml_model.predict_proba(feat)[0][1])
            
            st.markdown(f"#### Assessed Flood Risk: **{ml_prob * 100:.1f}%**")
            if ml_prob >= threshold:
                badge = '<div class="badge-red">🚨 CRITICAL FLOOD ALERT</div>' if ml_prob >= 0.75 else '<div class="badge-yellow">⚠️ SIGNIFICANT RISK</div>'
                st.markdown(badge, unsafe_allow_html=True)
                st.error(f"Municipal Action Required: Order drainage deployment and sound sirens in low-lying sectors of {selected_city}.")
            else:
                st.markdown('<div class="badge-green">✅ CONDITIONS SAFE (GREEN)</div>', unsafe_allow_html=True)
                st.success(f"No immediate inundation threat detected for {selected_city}.")
            st.progress(ml_prob)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: CASCADING DEEP LEARNING PIPELINE (TABTRANSFORMER -> FLOODDNN)
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🌊 Sequential Two-Stage Deep Learning Pipeline")
    st.caption("Stage 1: **TabTransformer** (Rainfall Regression) ➔ Stage 2: **FloodDNN** (Calibrated Risk Probability)")
    
    st.markdown("""
    <div class="card-box">
        <b>Pipeline Flow:</b><br>
        1. <b>TabTransformer</b> projects continuous features (elevation, river distance, coordinates, population, 3-day antecedent saturation) into 32-dimensional embedding spaces, applying <b>Multi-Head Self-Attention</b> to forecast today's daily precipitation (mm).<br>
        2. That predicted rainfall immediately cascades into <b>FloodDNN</b> (4-layer Funnel Network), generating an end-to-end flood risk probability.
    </div>
    """, unsafe_allow_html=True)
    
    c_dl1, c_dl2 = st.columns([1, 1])
    with c_dl1:
        st.markdown("#### 🌧️ Step 1: Temporal Saturation Inputs")
        rain3 = st.slider("Antecedent 3-Day Accumulated Rain (mm)", 0.0, 300.0, 45.0, step=5.0,
                          help="Soil moisture saturation proxy representing prior storm runoff.")
        month_val = float(datetime.now().month)
        
        st.caption(f"Geography: Elevation = {c_elev:.0f}m | River Distance = {c_river:.0f}m | Pop = {c_pop/1e6:.1f}M | Month = {month_val:.0f}")

    with c_dl2:
        st.markdown("#### 🚀 Step 2: Trigger Cascading Inference")
        st.write("")
        if st.button("EXECUTE CASCADING DL INFERENCE", key="btn_dl"):
            # --- STAGE 1: TabTransformer ---
            r_in = [c_elev, c_river, c_lat, c_lon, c_pop, month_val, rain3]
            r_scaled = tab_scaler.transform([r_in])
            with torch.no_grad():
                pred_rain = max(0.0, tab_trans(torch.tensor(r_scaled, dtype=torch.float32)).item())

            # --- STAGE 2: FloodDNN ---
            f_in = [pred_rain, c_elev, c_river, c_lat, c_lon]
            f_scaled = xai_explainer.scaler.transform([f_in])
            with torch.no_grad():
                f_prob = xai_explainer.model(torch.tensor(f_scaled, dtype=torch.float32).to(xai_explainer.device)).item()

            st.divider()
            c_res1, c_res2 = st.columns(2)
            c_res1.metric("🌧️ Stage 1: TabTransformer Forecast", f"{pred_rain:.2f} mm")
            c_res2.metric("🌊 Stage 2: FloodDNN Risk Probability", f"{f_prob * 100:.1f}%")
            
            if f_prob >= 0.5:
                st.markdown('<div class="badge-red">🚨 HIGH FLOOD PROBABILITY</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-green">✅ SAFE / MANAGED RUNOFF</div>', unsafe_allow_html=True)
            st.progress(f_prob)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: EXPLAINABLE AI (CAPTUM INTEGRATED GRADIENTS)
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🧠 Explainable AI — Captum Integrated Gradients")
    st.caption("Axiomatic Attribution & Mathematical Audit Trail for Civic Authorities (Sundararajan et al., 2017)")
    
    col_x1, col_x2 = st.columns([1, 1])
    with col_x1:
        st.markdown("#### 🌦️ Evaluated Scenario Parameters")
        x_rain = st.slider("Precipitation Load (mm / day)", 0.0, 300.0, 140.0, step=5.0)
        x_elev = st.number_input("SRTM Elevation (meters)", value=float(c_elev), step=5.0)
        x_river = st.number_input("Distance to Active River Channel (meters)", value=float(c_river), step=50.0)

    with col_x2:
        st.markdown("#### ⚙️ Captum Quadrature Parameters")
        n_quad_steps = st.select_slider(
            "Riemannian Quadrature Integration Steps",
            options=[20, 50, 100, 200],
            value=50,
            help="Higher integration steps guarantee precision of the line integral from baseline."
        )
        st.write("")
        st.write("")
        run_xai = st.button("🚀 COMPUTE AXIOMATIC ATTRIBUTIONS", key="btn_xai", use_container_width=True)

    if run_xai:
        with st.spinner("Integrating gradients across deep hidden layers..."):
            raw_sample = [x_rain, x_elev, x_river, c_lat, c_lon]
            res = xai_explainer.explain_city(raw_sample, city_name=selected_city, n_steps=n_quad_steps)

        st.divider()
        m1, m2, m3 = st.columns(3)
        prob = res["probability"]
        badge = '<div class="badge-red">CRITICAL (RED ALERT)</div>' if prob >= 0.70 else ('<div class="badge-yellow">MODERATE WATCH</div>' if prob >= 0.40 else '<div class="badge-green">SAFE / LOW RISK</div>')
        
        with m1:
            st.markdown(f"#### Assessed Risk: **{prob * 100:.1f}%**")
            st.markdown(badge, unsafe_allow_html=True)
        with m2:
            st.metric("Baseline Neutral Risk", f"{res['baseline_probability'] * 100:.1f}%")
        with m3:
            st.metric("Completeness Delta (|Δ|)", f"{abs(res['convergence_delta']):.5f}",
                      help="Measures mathematical convergence: sum(attributions) vs [F(x) - F(baseline)]")

        st.markdown("#### 📋 Formal Municipal Decision Audit Trail")
        st.markdown(f"""
        <div class="card-box">
            <b>Target City:</b> {selected_city} | <b>Algorithm:</b> PyTorch Captum (Integrated Gradients)<br><br>
            • <b>Primary Risk Driver:</b> <span style="color:#ef4444; font-weight:bold;">{res['risk_drivers'][0]['feature']}</span> 
            ({res['risk_drivers'][0]['raw_value']:.1f}) contributing 
            <b>+{res['risk_drivers'][0]['importance_pct']:.1f}%</b> of risk divergence.<br>
            • <b>Primary Protective Buffer:</b> <span style="color:#10b981; font-weight:bold;">
            {res['protective_factors'][0]['feature'] if res['protective_factors'] else 'None'}</span> 
            mitigating risk by 
            <b>{res['protective_factors'][0]['attribution']:.4f}</b>.
        </div>
        """, unsafe_allow_html=True)

        # Visualizations
        subtab1, subtab2 = st.tabs(["📊 Signed Attribution Divergence", "🌊 Risk Decomposition Waterfall"])
        with subtab1:
            fig_path = plot_attribution_bars(res, output_path=f"/tmp/{selected_city}_bars.png")
            st.image(fig_path, use_container_width=True)
        with subtab2:
            wf_path = plot_waterfall(res, output_path=f"/tmp/{selected_city}_waterfall.png")
            st.image(wf_path, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: GUJARAT MUNICIPAL BENCHMARKS (1-CLICK CIVIC PRESETS)
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📍 Gujarat Municipal Case Studies & Presets")
    st.caption("Demonstrating how topography and river morphology alter disaster outcomes across Gujarat")
    
    presets = [
        {"name": "Surat (Tapi River Estuary)", "rain": 180.0, "elev": 13.0, "dist": 250.0, "lat": 21.17, "lon": 72.83, "desc": "Low elevation + heavy monsoon + extreme river proximity -> High Risk"},
        {"name": "Ahmedabad (Sabarmati Basin)", "rain": 140.0, "elev": 53.0, "dist": 600.0, "lat": 23.02, "lon": 72.57, "desc": "Substantial rainfall in dense urban alluvial plain -> High Risk"},
        {"name": "Vadodara (Vishwamitri Plain)", "rain": 125.0, "elev": 39.0, "dist": 450.0, "lat": 22.30, "lon": 73.18, "desc": "River basin prone to flash river swelling -> High Risk"},
        {"name": "Saputara (Dang Hill Station)", "rain": 190.0, "elev": 1000.0, "dist": 3500.0, "lat": 20.58, "lon": 73.75, "desc": "190mm rain BUT 1000m elevation acts as massive protective buffer -> SAFE"},
        {"name": "Rajkot (Saurashtra Plateau)", "rain": 35.0, "elev": 132.0, "dist": 2800.0, "lat": 22.30, "lon": 70.80, "desc": "Semi-arid plateau with moderate rain -> SAFE Baseline"}
    ]
    
    b_cols = st.columns(len(presets))
    for i, p in enumerate(presets):
        if b_cols[i].button(f"📌 {p['name'].split()[0]}", key=f"preset_{i}"):
            res_p = xai_explainer.explain_city([p['rain'], p['elev'], p['dist'], p['lat'], p['lon']], city_name=p['name'])
            st.divider()
            st.markdown(f"#### Municipal Evaluation for: **{p['name']}**")
            st.caption(p["desc"])
            
            c_p1, c_p2, c_p3 = st.columns(3)
            c_p1.metric("Assessed Flood Risk", f"{res_p['probability'] * 100:.1f}%")
            c_p2.metric("Primary Risk Driver", f"{res_p['risk_drivers'][0]['feature']} (+{res_p['risk_drivers'][0]['importance_pct']:.1f}%)")
            c_p3.metric("Completeness Delta", f"{res_p['convergence_delta']:.5f}")
            
            fig_p = plot_attribution_bars(res_p, output_path=f"/tmp/preset_{i}.png")
            st.image(fig_p, use_container_width=True)

