"""
Reads the latest real-time data collected by run_realtime_pipeline.py.
All three Streamlit apps (app_demo, app, final_app) use this module.
"""

import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

WEATHER_LOG = os.path.join(BASE_DIR, "data/raw/realtime/weather/realtime_weather_log.csv")
RAINFALL_LOG = os.path.join(BASE_DIR, "data/raw/realtime/rainfall/realtime_rainfall_log.csv")
RIVER_LOG = os.path.join(BASE_DIR, "data/raw/realtime/river/realtime_river_level_log.csv")
SATELLITE_LOG = os.path.join(BASE_DIR, "data/raw/realtime/satellite/satellite_rainfall_log.csv")
REALTIME_DATASET = os.path.join(BASE_DIR, "data/processed/realtime_dataset.csv")


def safe_read(path):
    """Read CSV safely, return empty DataFrame if missing or corrupt."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path, on_bad_lines="skip", low_memory=False)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def get_latest_weather(city=None):
    """Get latest weather data from the pipeline log.
    
    Returns dict with: temperature, humidity, pressure, wind_speed, cloud_cover, timestamp
    Returns None if no data available.
    """
    df = safe_read(WEATHER_LOG)
    if df.empty:
        return None

    # Clean: only keep rows with valid required columns
    required = ["timestamp", "city", "temperature_C", "humidity_percent", "wind_speed_kmh"]
    df.columns = df.columns.str.strip()

    # Filter rows where key columns exist and are not null
    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=["temperature_C", "city"])
    if df.empty:
        return None

    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])

    if city:
        df = df[df["city"].str.strip() == city.strip()]
        if df.empty:
            return None

    # Get latest row
    latest = df.sort_values("timestamp").iloc[-1]

    return {
        "temperature": float(latest.get("temperature_C", 0) or 30.0),
        "humidity": float(latest.get("humidity_percent", 0) or 50.0),
        "pressure": float(latest.get("surface_pressure", 0) or 1012.0),
        "wind_speed": float(latest.get("wind_speed_kmh", 0) or 10.0),
        "cloud_cover": float(latest.get("cloud_cover_percent", 0) or 50.0),
        "precipitation": float(latest.get("precipitation_mm", 0) or 0),
        "timestamp": str(latest["timestamp"]),
        "city": str(latest.get("city", "Unknown")),
        "source": "Real-Time Pipeline"
    }


def get_latest_rainfall(city=None):
    """Get latest rainfall from the pipeline log.
    
    Returns dict with: precipitation_mm, timestamp
    Returns None if no data.
    """
    df = safe_read(RAINFALL_LOG)
    if df.empty:
        return None

    if "precipitation_mm" not in df.columns or "city" not in df.columns:
        return None

    df = df.dropna(subset=["city"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if city:
        df = df[df["city"].str.strip() == city.strip()]
        if df.empty:
            return None

    latest = df.sort_values("date").iloc[-1]

    precip = latest.get("precipitation_mm")
    if pd.isna(precip):
        precip = 0.0

    return {
        "precipitation_mm": float(precip),
        "timestamp": str(latest["date"]),
        "city": str(latest.get("city", "Unknown")),
        "source": "Real-Time Pipeline"
    }


def get_latest_river(city=None):
    """Get latest river level from the pipeline log.
    
    Returns dict with: river, station, level, danger, warning, status, timestamp
    Returns None if no data.
    """
    df = safe_read(RIVER_LOG)
    if df.empty:
        return None

    required = ["city", "timestamp", "river", "level", "danger", "status"]
    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=["city", "river"])
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])

    if city:
        df = df[df["city"].str.strip() == city.strip()]
        if df.empty:
            return None

    latest = df.sort_values("timestamp").iloc[-1]

    return {
        "river": str(latest.get("river", "Unknown")),
        "station": str(latest.get("station", "Unknown")),
        "level": float(latest.get("level", 0)),
        "danger": float(latest.get("danger", 0)),
        "warning": float(latest.get("warning", 0)),
        "status": str(latest.get("status", "Unknown")),
        "timestamp": str(latest["timestamp"]),
        "city": str(latest.get("city", "Unknown")),
        "source": "Real-Time Pipeline"
    }


def get_latest_satellite(city=None):
    """Get latest satellite-derived rainfall.

    Returns dict with: rainfall_mm, hourly_max_mm, source, timestamp
    Returns None if no data.
    """
    df = safe_read(SATELLITE_LOG)
    if df.empty:
        return None

    if "city" not in df.columns:
        return None

    df = df.dropna(subset=["city"])
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])

    if city:
        df = df[df["city"].str.strip() == city.strip()]
        if df.empty:
            return None

    latest = df.sort_values("timestamp").iloc[-1]

    rain = latest.get("rainfall_mm", 0)
    if pd.isna(rain):
        rain = 0.0

    return {
        "rainfall_mm": float(rain),
        "hourly_max_mm": float(latest.get("hourly_max_mm", 0) or 0),
        "hours_with_rain": int(latest.get("hours_with_rain", 0) or 0),
        "source": str(latest.get("source", "Satellite")),
        "timestamp": str(latest["timestamp"]),
        "city": str(latest.get("city", "Unknown")),
    }


def get_all_realtime(city):
    """Get all real-time data for a city. Returns a combined dict."""
    weather = get_latest_weather(city)
    rainfall = get_latest_rainfall(city)
    satellite = get_latest_satellite(city)
    river = get_latest_river(city)

    return {
        "weather": weather,
        "rainfall": rainfall,
        "satellite": satellite,
        "river": river,
        "has_weather": weather is not None,
        "has_rainfall": rainfall is not None,
        "has_satellite": satellite is not None,
        "has_river": river is not None,
        "pipeline_active": any([weather, rainfall, satellite, river])
    }


def get_pipeline_status():
    """Check if the pipeline has run recently."""
    lock_file = os.path.join(BASE_DIR, "src", "pipeline.lock")
    last_run_file = os.path.join(BASE_DIR, "src", "pipeline_last_run.txt")

    status = {
        "is_running": os.path.exists(lock_file),
        "last_run": None,
        "weather_log_exists": os.path.exists(WEATHER_LOG),
        "rainfall_log_exists": os.path.exists(RAINFALL_LOG),
        "satellite_log_exists": os.path.exists(SATELLITE_LOG),
        "river_log_exists": os.path.exists(RIVER_LOG),
    }

    if os.path.exists(last_run_file):
        try:
            with open(last_run_file) as f:
                status["last_run"] = f.read().strip()
        except Exception:
            pass

    return status
