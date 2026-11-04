import streamlit as st
import pandas as pd
import plotly.express as px
import time
import pymongo

# ==========================================
# PAGE CONFIG & DYNAMIC DESIGN
# ==========================================
st.set_page_config(
    page_title="RAINWISE | MongoDB Live Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .metric-card {
        background-color: #1E212B; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-left: 5px solid #00FF88;
        margin-bottom: 20px;
    }
    .metric-title { font-size: 1.1rem; color: #A0AAB2; margin-bottom: 5px; }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #00FF88; }
    .header-style {
        text-align: center; background: -webkit-linear-gradient(#00FF88, #00A6FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3.5rem; font-weight: 900; margin-bottom: 0px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header-style">🍃 MongoDB Live Analytics</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AAB2; font-size: 1.2rem;'>Real-Time Data Visualization (12 Metrics) directly from MongoDB NoSQL</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# MONGODB CONNECTION
# ==========================================
@st.cache_resource
def get_mongo_client():
    try:
        return pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
    except:
        return None

client = get_mongo_client()

if not client:
    st.error("MongoDB is currently offline or unreachable at localhost:27017.")
    st.stop()

db = client['rainwise_db']
col_cities = db['city_summaries']

# ==========================================
# DATA FETCHING (only re-fetch if new data)
# ==========================================
try:
    latest_doc = col_cities.find_one({}, sort=[("last_updated", pymongo.DESCENDING)])
    current_latest_time = latest_doc.get("last_updated") if latest_doc else None
except:
    current_latest_time = None

if 'last_updated_time' not in st.session_state:
    st.session_state.last_updated_time = None
    st.session_state.df_cache = pd.DataFrame()

if current_latest_time != st.session_state.last_updated_time:
    st.session_state.last_updated_time = current_latest_time
    cursor = col_cities.find({}, {"_id": 0})
    st.session_state.df_cache = pd.DataFrame(list(cursor))

df = st.session_state.df_cache

if df.empty:
    st.warning("MongoDB is currently empty. Waiting for live data from pipeline...")
else:
    # ── Top KPI Metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    latest_time  = df['last_updated'].max() if 'last_updated' in df.columns else "N/A"
    avg_risk     = df['Flood_Risk_Score'].mean() if 'Flood_Risk_Score' in df.columns else 0
    critical_cnt = len(df[df['Risk_Level'] == 'Critical']) if 'Risk_Level' in df.columns else 0

    c1.markdown(f'<div class="metric-card"><div class="metric-title">Active MongoDB Docs</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-title">Last Live Sync</div><div class="metric-value" style="font-size:1.4rem;margin-top:10px;">{latest_time}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-title">Avg Flood Risk Score</div><div class="metric-value">{avg_risk:.1f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-title">Critical Cities</div><div class="metric-value" style="color:#FF007F;">{critical_cnt}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── ROW 1: Geospatial & Risk ──────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. 🌧️ Live Rainfall Radar")
        if {'latitude', 'longitude', 'rain_mm'}.issubset(df.columns):
            df_geo = df.dropna(subset=['latitude', 'longitude', 'rain_mm'])
            if not df_geo.empty:
                fig1 = px.scatter_map(
                    df_geo, lat="latitude", lon="longitude",
                    color="rain_mm", size="rain_mm", hover_name="city",
                    color_continuous_scale=px.colors.sequential.Tealgrn,
                    size_max=15, zoom=5, map_style="carto-darkmatter"
                )
                fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
                st.plotly_chart(fig1, width='stretch', key="c1")

    with col2:
        st.markdown("#### 2. 🎯 Risk Level Breakdown")
        if 'Risk_Level' in df.columns:
            risk_counts = df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Level', 'Count']
            fig2 = px.pie(
                risk_counts, names='Risk_Level', values='Count', hole=0.4,
                color='Risk_Level',
                color_discrete_map={'Low':'#00FF88','Warning':'#FFC107','Critical':'#FF007F','Unknown':'#555555'}
            )
            fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
            st.plotly_chart(fig2, width='stretch', key="c2")

    with col3:
        st.markdown("#### 3. 🗺️ Flood Risk Density Map")
        if {'latitude', 'longitude', 'Flood_Risk_Score'}.issubset(df.columns):
            df_geo2 = df.dropna(subset=['latitude', 'longitude', 'Flood_Risk_Score'])
            if not df_geo2.empty:
                fig3 = px.density_map(
                    df_geo2, lat='latitude', lon='longitude',
                    z='Flood_Risk_Score', radius=20,
                    zoom=5, map_style="carto-darkmatter",
                    color_continuous_scale="Inferno"
                )
                fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
                st.plotly_chart(fig3, width='stretch', key="c3")

    st.markdown("---")

    # ── ROW 2: Rainfall & Humidity ────────────────────────────────────────
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("#### 4. ☔ Top 10 Raining Cities Right Now")
        if 'rain_mm' in df.columns:
            top_rain = df.sort_values('rain_mm', ascending=False).head(10)
            fig4 = px.bar(top_rain, x='city', y='rain_mm', color='rain_mm',
                          color_continuous_scale="Blues")
            fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
            st.plotly_chart(fig4, width='stretch', key="c4")

    with col5:
        st.markdown("#### 5. 🌊 Rainfall Volume Distribution")
        if 'rain_mm' in df.columns:
            fig5 = px.histogram(df, x='rain_mm', nbins=30,
                                color_discrete_sequence=['#00E5FF'])
            fig5.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
            st.plotly_chart(fig5, width='stretch', key="c5")

    with col6:
        st.markdown("#### 6. 🌡️ Humidity Distribution")
        if 'humidity' in df.columns:
            fig6 = px.histogram(df, x='humidity', nbins=30,
                                color_discrete_sequence=['#FF007F'])
            fig6.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
            st.plotly_chart(fig6, width='stretch', key="c6")

    st.markdown("---")

    # ── ROW 3: Geography & Correlation ───────────────────────────────────
    col7, col8, col9 = st.columns(3)

    with col7:
        st.markdown("#### 7. ⛰️ Elevation vs Flood Risk")
        if {'elevation_m', 'Flood_Risk_Score'}.issubset(df.columns):
            fig7 = px.scatter(df, x="elevation_m", y="Flood_Risk_Score",
                              color="Flood_Risk_Score", color_continuous_scale="Reds",
                              hover_name="city")
            fig7.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
            st.plotly_chart(fig7, width='stretch', key="c7")

    with col8:
        st.markdown("#### 8. 🏞️ River Proximity vs Risk")
        if {'distance_to_river_m', 'Flood_Risk_Score'}.issubset(df.columns):
            fig8 = px.scatter(df, x="distance_to_river_m", y="Flood_Risk_Score",
                              color="Flood_Risk_Score", color_continuous_scale="Purpor",
                              hover_name="city")
            fig8.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
            st.plotly_chart(fig8, width='stretch', key="c8")

    with col9:
        st.markdown("#### 9. 🧠 Rainfall vs Humidity Heatmap")
        if {'rain_mm', 'humidity'}.issubset(df.columns):
            fig9 = px.density_heatmap(df, x="rain_mm", y="humidity",
                                      color_continuous_scale="Viridis")
            fig9.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
            st.plotly_chart(fig9, width='stretch', key="c9")

    st.markdown("---")

    # ── ROW 4: Population / Demographics ─────────────────────────────────
    col10, col11, col12 = st.columns(3)

    with col10:
        st.markdown("#### 10. 💥 Projected Pop vs Live Risk")
        if {'projected_pop_2026', 'Flood_Risk_Score'}.issubset(df.columns):
            df_pop = df.dropna(subset=['projected_pop_2026', 'Flood_Risk_Score'])
            if not df_pop.empty:
                fig10 = px.scatter(
                    df_pop, x='projected_pop_2026', y='Flood_Risk_Score',
                    size='projected_pop_2026', color='Risk_Level', hover_name='city',
                    color_discrete_map={'Low':'#00FF88','Warning':'#FFC107','Critical':'#FF007F'}
                )
                fig10.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
                st.plotly_chart(fig10, width='stretch', key="c10")

    with col11:
        st.markdown("#### 11. 📈 Most Vulnerable Fast-Growing Cities")
        if {'projected_pop_2026', 'Flood_Risk_Score'}.issubset(df.columns):
            df_clean = df.dropna(subset=['projected_pop_2026', 'Flood_Risk_Score'])
            df_top = df_clean.sort_values(
                by=['Flood_Risk_Score', 'projected_pop_2026'], ascending=[False, False]
            ).head(10)
            if not df_top.empty:
                fig11 = px.bar(df_top, x='Flood_Risk_Score', y='city',
                               orientation='h', color='projected_pop_2026',
                               color_continuous_scale="Oranges")
                fig11.update_layout(yaxis={'categoryorder':'total ascending'},
                                    margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
                st.plotly_chart(fig11, width='stretch', key="c11")

    with col12:
        st.markdown("#### 12. 📊 Highest Absolute Growth Delta (2026)")
        if {'projected_pop_2026', 'population'}.issubset(df.columns):
            df['growth'] = (pd.to_numeric(df['projected_pop_2026'], errors='coerce') -
                            pd.to_numeric(df['population'], errors='coerce'))
            df_growth = df.dropna(subset=['growth']).sort_values('growth', ascending=False).head(10)
            if not df_growth.empty:
                fig12 = px.bar(df_growth, x='city', y='growth',
                               color='growth', color_continuous_scale="Plasma")
                fig12.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#0E1117")
                st.plotly_chart(fig12, width='stretch', key="c12")

# ==========================================
# SMART AUTO REFRESH (polls every 2s)
# ==========================================
time.sleep(2)
st.rerun()
