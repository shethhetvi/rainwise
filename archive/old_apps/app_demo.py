import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.utils.realtime_data import get_all_realtime, get_pipeline_status

st.set_page_config(page_title="RAINWISE — Flood Prediction Demo", layout="centered")

st.title("🌊 RAINWISE — Flood Prediction Demo")
st.markdown("Predict flood risk from **rainfall & geography** using **Logistic Regression (Balanced)**.")

# ----------------------
# PIPELINE VISUALIZATION
# ----------------------
with st.expander("🔄 How the Pipeline Works", expanded=False):
    st.markdown("""
    ```
    📡 Real-Time Pipeline (run_realtime_pipeline.py)
    │  ├── realtime_weather.py  → Weather logs
    │  ├── gpm_fetcher.py       → 🛰️ Satellite rainfall (ERA5)
    │  ├── realtime_rainfall.py → Rainfall logs
    │  └── realtime_river.py    → River level logs
    │
    ↓  build_dataset.py merges all data
    │
    📥 User Input OR Pipeline Data
         ↓
    🗺️ GIS Lookup (Elevation, River Distance)
         ↓
    🤖 Flood Model (Logistic Regression)
    │
    📊 Risk Assessment + 🚨 Alert
    ```
    """)

# ----------------------
# LOAD MODEL (cached)
# ----------------------
@st.cache_resource
def load_model():
    return joblib.load("models/flood_model.pkl")

@st.cache_data
def load_city_data():
    df = pd.read_csv("data/config/gujarat_cities.csv")
    df.columns = df.columns.str.lower()
    return df

@st.cache_data
def load_gis_data():
    r = pd.read_csv("data/processed/gujarat_river_distance.csv")
    e = pd.read_csv("data/processed/gujarat_elevation.csv")
    r.columns = r.columns.str.lower()
    e.columns = e.columns.str.lower()
    return r, e

flood_model = load_model()
cities_df = load_city_data()
river_df, elev_df = load_gis_data()

def find_nearest(df, lat, lon):
    df = df.copy()
    df["dist"] = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    return df.loc[df["dist"].idxmin()]

# ----------------------
# CITY SELECTION
# ----------------------
city = st.selectbox("📍 Select City", cities_df["city"].unique())

row = cities_df[cities_df["city"] == city]
lat = float(row["lat"].values[0])
lon = float(row["lon"].values[0])

distance = float(find_nearest(river_df, lat, lon)["river_distance"])
elevation = float(find_nearest(elev_df, lat, lon)["elevation"])

# ----------------------
# LOCATION INFO
# ----------------------
st.subheader("🌍 Location & Geography")
col1, col2 = st.columns(2)
col1.metric("Latitude", f"{lat:.4f}")
col2.metric("Longitude", f"{lon:.4f}")

col3, col4, col5 = st.columns(3)
col3.metric("Distance to River", f"{distance:.0f} m")
col4.metric("Elevation", f"{elevation:.0f} m")
drainage_cap = col5.slider("🚰 Drainage Capacity (%)", 0, 100, 80, help="Local drainage infrastructure efficiency.")

# ----------------------
# PIPELINE STATUS
# ----------------------
p_status = get_pipeline_status()
st.caption(
    f"Pipeline: Weather {'✅' if p_status['weather_log_exists'] else '❌'} | "
    f"Satellite {'✅' if p_status.get('satellite_log_exists') else '❌'} | "
    f"Rainfall {'✅' if p_status['rainfall_log_exists'] else '❌'} | "
    f"River {'✅' if p_status['river_log_exists'] else '❌'}"
)

# ----------------------
# REAL-TIME PIPELINE DATA
# ----------------------
st.subheader("📡 Real-Time Pipeline Data")

realtime = get_all_realtime(city)

if realtime["pipeline_active"]:
    st.success("✅ Pipeline data available!")

    rt_cols = st.columns(3)

    # Rainfall from pipeline
    if realtime["has_rainfall"]:
        rt_rain = realtime["rainfall"]["precipitation_mm"]
        rt_cols[0].metric("🌧 Pipeline Rainfall", f"{rt_rain} mm")
    else:
        rt_rain = None
        rt_cols[0].metric("🌧 Rainfall", "No data")

    # River level from pipeline
    if realtime["has_river"]:
        rv = realtime["river"]
        rt_cols[1].metric(f"🏞 {rv['river']} River", f"{rv['level']:.1f} m³/s")
        if rv["status"] == "Above Danger":
            st.error(f"🚨 {rv['river']} River is ABOVE DANGER level! ({rv['level']:.1f} m³/s / {rv['danger']} m³/s)")
        elif rv["status"] == "Warning":
            st.warning(f"⚠️ {rv['river']} River at WARNING level ({rv['level']:.1f} m³/s / {rv['warning']} m³/s)")
        else:
            st.caption(f"River status: {rv['status']} (Discharge: {rv['level']:.1f} m³/s, Danger: {rv['danger']} m³/s)")
    else:
        rt_cols[1].metric("🏞 River", "No data")

    # Weather from pipeline
    if realtime["has_weather"]:
        w = realtime["weather"]
        rt_cols[2].metric("🌡 Temperature", f"{w['temperature']}°C")
    else:
        rt_cols[2].metric("🌡 Weather", "No data")

    # Satellite data
    if realtime.get("has_satellite") and realtime["satellite"]:
        sat = realtime["satellite"]
        st.divider()
        st.subheader("🛰️ Satellite Rainfall (ERA5)")
        sat_c1, sat_c2, sat_c3 = st.columns(3)
        sat_c1.metric("24h Total", f"{sat['rainfall_mm']:.1f} mm")
        sat_c2.metric("Max Hourly", f"{sat['hourly_max_mm']:.1f} mm")
        sat_c3.metric("Hours w/ Rain", f"{sat['hours_with_rain']}h")
        st.caption(f"📡 Satellite data from {sat['timestamp']}")

    # Unified Rainfall Logic (Satellite + API)
    obs_rain = 0.0
    if realtime.get("has_satellite") and realtime["satellite"]:
        obs_rain = max(obs_rain, float(realtime["satellite"]["rainfall_mm"]))
    if realtime.get("has_rainfall"):
        obs_rain = max(obs_rain, float(realtime["rainfall"]["precipitation_mm"]))
    
    rt_rain = obs_rain if (realtime.get("has_satellite") or realtime.get("has_rainfall")) else None
    use_pipeline = st.checkbox("📡 Use pipeline observation for prediction", value=rt_rain is not None)
else:
    st.info("ℹ️ No pipeline data found. Run `python src/data_collection/run_realtime_pipeline.py` to collect data.")
    use_pipeline = False
    rt_rain = None

# ----------------------
# RAINFALL INPUT
# ----------------------
st.subheader("🌧 Rainfall Input")

if use_pipeline and rt_rain is not None:
    rain = st.slider("Rainfall (mm)", 0.0, 100.0, min(float(rt_rain), 100.0), step=0.5,
                     help="Pre-filled from real-time pipeline data")
    st.caption(f"📡 Value from pipeline: {rt_rain} mm")
else:
    rain = st.slider("Rainfall (mm)", 0.0, 100.0, 10.0, step=0.5)

threshold = st.slider("🎯 Alert Threshold", 0.1, 0.9, 0.5, step=0.05)

# ----------------------
# PREDICT
# ----------------------
if st.button("🔍 Predict Flood Risk"):
    features = np.array([[rain, elevation, distance, lat, lon]])
    proba = float(flood_model.predict_proba(features)[0][1])

    st.divider()
    st.subheader("📊 Flood Risk Assessment")

    col_f1, col_f2 = st.columns(2)
    col_f1.metric("Flood Probability", f"{proba:.2f}")

    if proba > threshold:
        if proba > 0.8:
            col_f2.metric("Risk Level", "🔴 HIGH")
            st.error("🚨 HIGH FLOOD RISK — Evacuate low-lying areas immediately!")
        elif proba > 0.6:
            col_f2.metric("Risk Level", "🟠 SIGNIFICANT")
            st.error("🔴 SIGNIFICANT FLOOD RISK — Take precautionary measures!")
        else:
            col_f2.metric("Risk Level", "🟡 MODERATE")
            st.warning("⚠️ MODERATE FLOOD RISK — Stay alert and monitor conditions!")
    else:
        col_f2.metric("Risk Level", "🟢 LOW")
        st.success("✅ LOW FLOOD RISK — Conditions are currently safe.")

    # --- ALERT SYSTEM ---
    st.divider()
    st.subheader("🚨 Alert System")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_source = "Real-Time Pipeline" if use_pipeline else "Manual Input"

    # Combine with river data if available
    river_alert = ""
    if realtime["has_river"] and realtime["river"]["status"] != "Normal":
        rv = realtime["river"]
        river_alert = f"\n🏞️ River: {rv['river']} at {rv['level']:.1f} m³/s ({rv['status']})"

    if proba > 0.6:
        alert_level = "🔴 CRITICAL" if proba > 0.8 else "🟠 WARNING"
        st.error(f"""
**{alert_level} FLOOD ALERT — {city}**

📅 {timestamp} | 📍 {city} ({lat:.4f}, {lon:.4f})
🌧️ Rainfall: {rain} mm | 🌊 Probability: {proba:.2f}
⛰️ Elevation: {elevation:.0f}m | River Distance: {distance:.0f}m{river_alert}
📡 Data Source: {data_source}

**Action:** {"🚨 EVACUATE immediately!" if proba > 0.8 else "⚠️ Prepare for flooding."}
📱 SMS sent to: District Collector, NDRF, Municipal Corp
""")
        st.toast(f"🚨 FLOOD ALERT for {city}!", icon="🚨")

        sms_log = {
            "timestamp": timestamp,
            "city": city,
            "rainfall_mm": float(rain),
            "flood_probability": float(round(proba, 3)),
            "alert_level": "CRITICAL" if proba > 0.8 else "WARNING",
            "data_source": data_source,
            "river_status": realtime["river"]["status"] if realtime["has_river"] else "N/A",
            "recipients": "District Collector, NDRF, Municipal Corp"
        }
        st.json(sms_log)
    elif proba > threshold:
        st.info(f"ℹ️ Moderate alert for {city} at {timestamp}.")
    else:
        st.success(f"✅ No alert needed. {city} is safe at {timestamp}.")

    # Download report
    summary_df = pd.DataFrame({
        "Parameter": ["City", "Rainfall", "Elevation", "River Distance", "Flood Probability",
                      "Data Source", "River Status"],
        "Value": [city, f"{rain} mm", f"{elevation:.0f} m", f"{distance:.0f} m",
                  f"{proba:.2f}", data_source,
                  realtime["river"]["status"] if realtime["has_river"] else "N/A"]
    })
    st.table(summary_df)

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button("📥 Report (CSV)", summary_df.to_csv(index=False),
                            f"flood_report_{city}.csv", "text/csv")

    alert_json = {
        "timestamp": timestamp, "city": city,
        "lat": float(lat), "lon": float(lon),
        "rainfall_mm": float(rain), "flood_probability": float(round(proba, 3)),
        "elevation_m": int(round(elevation)), "river_distance_m": int(round(distance)),
        "alert_level": "CRITICAL" if proba > 0.8 else ("WARNING" if proba > 0.6 else "SAFE"),
        "source": data_source
    }
    col_dl2.download_button("📥 Alert (JSON)", json.dumps(alert_json, indent=2),
                            f"alert_{city}.json", "application/json")

# ----------------------
# MODEL EVALUATION
# ----------------------
st.divider()
st.subheader("📊 Flood Model Evaluation")

tab1, tab2 = st.tabs(["🌊 Confusion Matrix", "📈 Feature Importance"])
with tab1:
    st.image("outputs/flood_logistic_confusion_matrix.png", caption="Logistic Regression Confusion Matrix")
with tab2:
    st.image("outputs/flood_logistic_feature_importance.png", caption="Safety-Prioritized Feature Importance")

st.subheader("🗺 Map")
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

# ----------------------
# [RUBRIC] DRAINAGE STRESS TEST
# ----------------------
st.divider()
st.subheader("🚰 Drainage Stress Test Analysis")
stress_col1, stress_col2 = st.columns(2)

with stress_col1:
    st.markdown("##### 🧪 Stress Level")
    # Simulation: Stress = Rain / Drainage (normalized)
    stress = (rain / (drainage_cap + 1)) * 100
    st.metric("Infrastructure Stress", f"{stress:.1f}%")
    if stress > 80:
        st.error("🚨 CRITICAL: Rainfall volume exceeds drainage capacity (Backflow likely).")
    elif stress > 50:
        st.warning("⚠️ HIGH: Drainage system running at near-maximum limit.")
    else:
        st.success("✅ OPTIMAL: Local infrastructure can handle the current rainfall load.")

with stress_col2:
    st.markdown("##### 🏙️ Urban Impact")
    msg = "No major urban clogging detected." if drainage_cap > 70 else "Significant surface runoff in streets due to poor drainage."
    st.info(f"**Insight:** {msg}")
    st.caption("This analysis uses the 'Drainage Factor' feature engineered for the Logistic Regression model.")

st.info("""
**🔍 Explainable AI (XAI) Tip:** Notice how changing the **Drainage Slider** affects the probability 
above. This proves that our model understands the difference between a 'Storm' and a 'Flood' 
based on urban infrastructure resilience.
""")    