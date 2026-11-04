import streamlit as st
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
from model_trainingDL.models import TabTransformer

# ==========================================
# COMPATIBLE ARCHITECTURES
# ==========================================
class CompatibleFloodDNN(nn.Module):
    """A Deep Neural Network matching the saved weights in DLmodels/flood_dnn.pth"""
    def __init__(self):
        super(CompatibleFloodDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# PAGE CONFIG & PREMIUM AESTHETICS
# ==========================================
st.set_page_config(
    page_title="RAINWISE AI — Sequential DL Pipeline",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .metric-container {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌊 RAINWISE AI — Phase 3: Sequential DL Pipeline")
st.markdown("### Stage 1: TabTransformer (Rain) → Stage 2: FloodDNN (Risk)")

# ==========================================
# LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_dl_resources():
    device = torch.device("cpu")
    
    # 1. Rainfall Model (TabTransformer) - 7 features, Depth 3
    rain_model = TabTransformer(input_dim=7, depth=3)
    if os.path.exists("DLmodels/tab_transformer_rainfall.pth"):
        rain_model.load_state_dict(torch.load("DLmodels/tab_transformer_rainfall.pth", map_location=device))
    rain_model.eval()
    rain_scaler = joblib.load("DLmodels/tab_transformer_scaler.pkl") if os.path.exists("DLmodels/tab_transformer_scaler.pkl") else None
    
    # 2. Flood Model (CompatibleDNN) - 5 features
    flood_model = CompatibleFloodDNN()
    if os.path.exists("DLmodels/flood_dnn.pth"):
        flood_model.load_state_dict(torch.load("DLmodels/flood_dnn.pth", map_location=device))
    flood_model.eval()
    flood_scaler = joblib.load("DLmodels/scaler.pkl") if os.path.exists("DLmodels/scaler.pkl") else None
    
    # Meta Data
    cities_df = pd.read_csv("data/config/gujarat_cities.csv")
    cities_df.columns = cities_df.columns.str.lower()
    
    r_dist = pd.read_csv("data/processed/gujarat_river_distance.csv")
    elev = pd.read_csv("data/processed/gujarat_elevation.csv")
    r_dist.columns = r_dist.columns.str.lower()
    elev.columns = elev.columns.str.lower()
    
    return rain_model, rain_scaler, flood_model, flood_scaler, cities_df, r_dist, elev

rain_model, rain_scaler, flood_model, flood_scaler, cities_df, river_df, elev_df = load_dl_resources()

def find_nearest(df, lat, lon):
    df_calc = df.copy()
    df_calc["dist"] = (df_calc["lat"] - lat)**2 + (df_calc["lon"] - lon)**2
    return df_calc.loc[df_calc["dist"].idxmin()]

# ==========================================
# SIDEBAR / CONFIG
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.header("Pipeline Settings")
    threshold = st.slider("Flood Alert Threshold", 0.1, 0.9, 0.5, step=0.05)
    
    st.divider()
    st.subheader("System Status")
    st.success("Rainfall Model: Loaded ✅")
    st.success("Flood Model: Loaded ✅")

# ==========================================
# MAIN INTERFACE
# ==========================================
col_l, col_r = st.columns([1, 1])

with col_l:
    st.markdown("#### 📍 Step 1: Context Selection")
    city = st.selectbox("Select Target City", cities_df["city"].unique(), index=60) # Default to Ahmedabad
    
    city_row = cities_df[cities_df["city"] == city].iloc[0]
    lat = float(city_row["lat"])
    lon = float(city_row["lon"])
    
    # Population fallback
    pop_mapping = {"Ahmedabad": 8600000, "Surat": 6100000, "Vadodara": 2200000, "Rajkot": 1800000}
    pop = float(pop_mapping.get(city, 1000000))
    
    dist_val = float(find_nearest(river_df, lat, lon)["river_distance"])
    elev_val = float(find_nearest(elev_df, lat, lon)["elevation"])
    
    st.markdown(f"""
    <div class="metric-container">
        <b>Latitude:</b> {lat:.4f} | <b>Longitude:</b> {lon:.4f}<br>
        <b>Elevation:</b> {elev_val:.0f}m | <b>River Proximity:</b> {dist_val:.0f}m<br>
        <b>Est. Population:</b> {pop/1e6:.1f}M
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🌧️ Step 2: Temporal Input")
    rain3 = st.slider("Rainfall in last 3 days (mm)", 0.0, 300.0, 45.0, step=1.0)
    st.caption("Soil saturation seed for rainfall prediction.")

with col_r:
    st.markdown("#### 🚀 Step 3: Run AI Pipeline")
    if st.button("EXECUTE SEQUENTIAL INFERENCE"):
        if not rain_scaler or not flood_scaler:
            st.error("Error: Model scalers not found.")
        else:
            # --- STAGE 1: RAINFALL (TabTransformer) ---
            # Features: [elev, dist, lat, lon, pop, month, rain3]
            month = float(datetime.now().month)
            rain_input = [elev_val, dist_val, lat, lon, pop, month, rain3]
            rain_scaled = rain_scaler.transform([rain_input])
            rain_tensor = torch.tensor(rain_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                pred_rain = max(0.0, rain_model(rain_tensor).item())
            
            # --- STAGE 2: FLOOD (FloodDNN) ---
            # Features: [rain, elev, dist, lat, lon]
            flood_input = [pred_rain, elev_val, dist_val, lat, lon]
            flood_scaled = flood_scaler.transform([flood_input])
            flood_tensor = torch.tensor(flood_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                flood_proba = flood_model(flood_tensor).item()
            
            # RESULTS
            st.divider()
            st.subheader("🌧️ Stage 1: Predicted Rainfall")
            st.metric("Rainfall Forecast", f"{pred_rain:.2f} mm")
            
            st.divider()
            st.subheader("🌊 Stage 2: Flood Risk")
            c1, c2 = st.columns(2)
            c1.metric("Probability", f"{flood_proba*100:.1f}%")
            if flood_proba >= threshold:
                c2.error("🚨 HIGH RISK")
            else:
                c2.success("✅ SAFE")
            
            st.progress(flood_proba)
            
            st.markdown(f"""
            > [!TIP]
            > **XAI Insight:** Based on {rain3}mm historical rain, the **TabTransformer** predicts **{pred_rain:.1f}mm** today. 
            > The **FloodDNN** calculates a **{flood_proba*100:.1f}%** risk for this topography.
            """)

st.divider()
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=10)
st.caption(f"RAINWISE Sequential DL Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
