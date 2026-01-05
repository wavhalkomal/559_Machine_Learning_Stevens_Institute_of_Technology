# # ============================================================
# # NYC Citi Bike Congestion Intelligence
# # Live Map + Region Filter + Risk Badges
# # + Historical Forecasting (Pretrained XGBoost + ARIMA)
# # ============================================================
#
# import os
# import glob
# import time
# import numpy as np
# import pandas as pd
# import requests
# import streamlit as st
# import pydeck as pdk
# import plotly.express as px
# import joblib
#
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
# # ---------------- Optional imports ----------------
# HAS_STATSMODELS = False
# try:
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     HAS_STATSMODELS = True
# except Exception:
#     HAS_STATSMODELS = False
#
# # ============================================================
# # STREAMLIT CONFIG
# # ============================================================
# st.set_page_config(
#     page_title="NYC Citi Bike",
#     page_icon="🚲",
#     layout="wide"
# )
#
# st.title("🚲 NYC Citi Bike Congestion Intelligence")
# st.caption("Live monitoring · Region analytics · ML-based forecasting")
#
# # ============================================================
# # CONSTANTS
# # ============================================================
# DATA_DIR = "data"
# MODEL_PATH = "pkl/xgboost_multioutput.pkl"
# MIN_ROWS_PER_STATION = 100
# DEFAULT_HIST_STATION = "5 Ave & E 29 St"
#
# BASE_FEATURES = [
#     "hour", "dayofweek", "month",
#     "is_weekend", "is_holiday",
#     "lag_1", "lag_24", "roll_3", "roll_24"
# ]
#
# WEATHER_FEATURES = [
#     "temperature_2m",
#     "relative_humidity_2m",
#     "precipitation",
#     "wind_speed_10m"
# ]
#
# PRETRAINED_FEATURES = [
#     "_lat", "_long", "tot_docks",
#     "hour", "dayofweek", "month",
#     "is_weekend", "is_holiday",
#     "lag_occ_1", "lag_occ_24",
#     "roll_occ_3", "roll_occ_24",
#     "temperature_2m",
#     "relative_humidity_2m",
#     "precipitation",
#     "wind_speed_10m"
# ]
#
# # ============================================================
# # LOAD PRETRAINED XGBOOST
# # ============================================================
# @st.cache_resource
# def load_xgb():
#     if not os.path.exists(MODEL_PATH):
#         return None
#     return joblib.load(MODEL_PATH)
#
# PRETRAINED_XGB = load_xgb()
#
# # ============================================================
# # METRICS
# # ============================================================
# def safe_mape(y_true, y_pred):
#     nz = np.abs(y_true) > 1e-9
#     return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if np.any(nz) else 0.0
#
# def compute_metrics(y_true, y_pred):
#     return {
#         "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
#         "MAE": mean_absolute_error(y_true, y_pred),
#         "R2": r2_score(y_true, y_pred),
#         "MAPE (%)": safe_mape(y_true, y_pred)
#     }
#
# # ============================================================
# # LOAD HISTORICAL DATA
# # ============================================================
# @st.cache_data
# def load_hist_data():
#     files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
#     df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     df["dock_name"] = df["dock_name"].astype(str).str.replace('"', '').str.strip()
#     return df
#
# df_hist = load_hist_data()
# stations_hist = sorted(df_hist["dock_name"].unique())
#
# # ============================================================
# # BUILD STATION TIME SERIES
# # ============================================================
# def build_station_hourly_ts(df, station):
#     d = df[df["dock_name"] == station].copy()
#     if d.empty:
#         return pd.DataFrame()
#
#     d = d.set_index("timestamp").sort_index()
#
#     # ✅ FIX: only numeric columns
#     numeric_cols = d.select_dtypes(include=[np.number]).columns
#     hourly = d[numeric_cols].resample("1H").mean()
#
#     st.write("Numeric columns used for resampling:", numeric_cols.tolist())
#
#     hourly["hour"] = hourly.index.hour
#     hourly["dayofweek"] = hourly.index.dayofweek
#     hourly["month"] = hourly.index.month
#     hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)
#     hourly["is_holiday"] = hourly.get("is_holiday", 0)
#
#     hourly["lag_1"] = hourly["occupancy"].shift(1)
#     hourly["lag_24"] = hourly["occupancy"].shift(24)
#     hourly["roll_3"] = hourly["occupancy"].rolling(3).mean()
#     hourly["roll_24"] = hourly["occupancy"].rolling(24).mean()
#
#     hourly = hourly.dropna().reset_index()
#     return hourly
#
#
# # ============================================================
# # PRETRAINED XGBOOST FEATURES
# # ============================================================
# def build_pretrained_features(df, station):
#     d = df[df["dock_name"] == station].copy()
#     if d.empty:
#         return pd.DataFrame()
#
#     d = d.set_index("timestamp").sort_index()
#
#     # ✅ FIX: numeric only
#     numeric_cols = d.select_dtypes(include=[np.number]).columns
#     hourly = d[numeric_cols].resample("1H").mean().reset_index()
#
#     st.write("Numeric columns used for resampling:", numeric_cols.tolist())
#
#     hourly["hour"] = hourly["timestamp"].dt.hour
#     hourly["dayofweek"] = hourly["timestamp"].dt.dayofweek
#     hourly["month"] = hourly["timestamp"].dt.month
#     hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)
#     hourly["is_holiday"] = hourly.get("is_holiday", 0)
#
#     hourly["lag_occ_1"] = hourly["occupancy"].shift(1)
#     hourly["lag_occ_24"] = hourly["occupancy"].shift(24)
#     hourly["roll_occ_3"] = hourly["occupancy"].rolling(3).mean()
#     hourly["roll_occ_24"] = hourly["occupancy"].rolling(24).mean()
#
#     hourly = hourly.dropna()
#
#     for c in PRETRAINED_FEATURES:
#         if c not in hourly.columns:
#             hourly[c] = 0.0
#
#     return hourly
#
#
#
# # ============================================================
# # FORECASTING SECTION
# # ============================================================
# st.subheader("📈 Congestion Forecasting - ARIMA & XGBoost")
#
# station_hist = st.selectbox(
#     "Select station (historical data)",
#     stations_hist,
#     index=stations_hist.index(DEFAULT_HIST_STATION)
# )
#
# ts = build_station_hourly_ts(df_hist, station_hist)
#
# if ts.empty or len(ts) < MIN_ROWS_PER_STATION:
#     st.warning("Not enough data for this station.")
#     st.stop()
#
# split_date = pd.Timestamp("2018-03-01")
# train = ts[ts["timestamp"] < split_date]
# test  = ts[ts["timestamp"] >= split_date]
#
# y_test = test["occupancy"].values
# timestamps = test["timestamp"].values
#
# preds = {}
# metrics = []
#
# # ---------------- XGBOOST (PRETRAINED) ----------------
# if PRETRAINED_XGB is not None:
#     feat_df = build_pretrained_features(df_hist, station_hist)
#     X_pre = feat_df[PRETRAINED_FEATURES].fillna(0)
#     yhat = PRETRAINED_XGB.predict(X_pre)[:,2]
#     yhat = yhat[-len(y_test):]
#     preds["XGBoost"] = yhat
#     metrics.append({"Model":"XGBoost", **compute_metrics(y_test, yhat)})
#
# # ---------------- ARIMA ----------------
# if HAS_STATSMODELS:
#     exog_cols = BASE_FEATURES + [c for c in WEATHER_FEATURES if c in ts.columns]
#     X_train = train[exog_cols].fillna(0)
#     X_test  = test[exog_cols].fillna(0)
#
#     ar = SARIMAX(
#         train["occupancy"],
#         exog=X_train,
#         order=(1,0,1),
#         seasonal_order=(1,0,1,24),
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     ).fit(disp=False)
#
#     yhat = ar.get_forecast(steps=len(X_test), exog=X_test).predicted_mean.values
#     preds["ARIMA"] = yhat
#     metrics.append({"Model":"ARIMA", **compute_metrics(y_test, yhat)})
#
# # ============================================================
# # METRICS TABLE
# # ============================================================
# metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
# st.subheader("📊 Model Evaluation Metrics (Holdout Test Set)")
# st.dataframe(metrics_df, use_container_width=True)
#
# # ============================================================
# # PLOT
# # ============================================================
# plot_df = pd.DataFrame({
#     "timestamp": timestamps,
#     "Actual": y_test
# })
#
# for k,v in preds.items():
#     plot_df[k] = v
#
# st.plotly_chart(
#     px.line(
#         plot_df,
#         x="timestamp",
#         y=plot_df.columns.drop("timestamp"),
#         title=f"Actual vs Predicted Occupancy — {station_hist}"
#     ),
#     use_container_width=True
# )
# ============================================================
# NYC Citi Bike Congestion Intelligence
# Live Map + Region Filter + Risk Badges
# Historical Forecasting (Pretrained XGBoost + ARIMA)
# ============================================================

import base64
import streamlit as st
import os
import glob
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
import plotly.express as px
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
# ================================================================================================
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ================================================================================================

# ============================================================
# 🔧 XGBOOST COMPATIBILITY PATCH (CRITICAL)
# ============================================================
import xgboost as xgb

def _safe_get_params(self, deep=True):
    """
    Override XGBoost get_params to avoid missing legacy attributes (gpu_id).
    """
    params = {}
    for key in self.__dict__:
        try:
            params[key] = getattr(self, key)
        except Exception:
            continue
    return params

# 🔥 Monkey-patch XGBModel
xgb.XGBModel.get_params = _safe_get_params


# ================= OPTIONAL IMPORTS =================
HAS_STATSMODELS = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except Exception:
    pass

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="NYC Citi Bike",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= HERO =================
def render_hero(image_path):
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .hero {{
            position: relative;
            height: 600px;
            margin-bottom: 40px;
        }}
        .hero img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 16px;
            filter: brightness(80%);
        }}
        </style>
        <div class="hero">
            <img src="data:image/jpeg;base64,{img}">
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("🚲 NYC Citi Bike Congestion Intelligence")
st.caption("Real-time monitoring · Region analytics · ML-based forecasting")
render_hero("img-3.jpeg")

# ================= CONSTANTS =================
DATA_DIR = "data"
XGB_MODEL_PATH = "pkl/xgboost_multioutput.pkl"
MIN_ROWS_PER_STATION = 100
DEFAULT_HIST_STATION = "5 Ave & E 29 St"



def sanitize_xgb_model(multi_model):
    """
    Fix XGBoost version incompatibility issues (gpu_id, predictor, etc.)
    """
    try:
        for est in multi_model.estimators_:
            if hasattr(est, "set_params"):
                est.set_params(
                    predictor="cpu_predictor",
                    tree_method="hist"
                )
    except Exception:
        pass
    return multi_model




STATION_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

BASE_FEATURES = [
    "hour", "dayofweek", "month",
    "is_weekend", "is_holiday",
    "lag_1", "lag_24", "roll_3", "roll_24"
]

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]

PRETRAINED_FEATURES = [
    "_lat", "_long", "tot_docks",
    "hour", "dayofweek", "month",
    "is_weekend", "is_holiday",
    "lag_occ_1", "lag_occ_24",
    "roll_occ_3", "roll_occ_24",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]

######################################################################

# ============================================================
# CONSTANTS / SETTINGS
# ============================================================
DATA_DIR = "data"
MIN_ROWS_PER_STATION = 100  # after hourly resample + lag features

STATION_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

BASE_FEATURES = ["hour", "dayofweek", "month", "is_weekend", "is_holiday", "lag_1", "lag_24", "roll_3", "roll_24"]
WEATHER_FEATURES = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]

LOCATIONS = {
    "All NYC (Default)": {"lat": 40.730610, "lon": -73.935242, "zoom": 11},
    "Times Square": {"lat": 40.758896, "lon": -73.985130, "zoom": 14},
    "Grand Central": {"lat": 40.7527, "lon": -73.9772, "zoom": 14},
    "Central Park": {"lat": 40.785091, "lon": -73.968285, "zoom": 13},
    "Wall St / FiDi": {"lat": 40.7074, "lon": -74.0113, "zoom": 14},
    "Brooklyn Bridge": {"lat": 40.7061, "lon": -73.9969, "zoom": 14},
    "Downtown Brooklyn": {"lat": 40.6917, "lon": -73.9848, "zoom": 14},
    "Williamsburg": {"lat": 40.7128, "lon": -73.9610, "zoom": 13},
    "Long Island City": {"lat": 40.7447, "lon": -73.9485, "zoom": 14},
    "Jersey City / Hoboken": {"lat": 40.7282, "lon": -74.0776, "zoom": 13},
}

DEFAULT_REGION = "Jersey City / Hoboken"
DEFAULT_HIST_STATION = "5 Ave & E 29 St"


# ============================================================
# UTIL: METRICS (safe)
# ============================================================
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nz = np.abs(y_true) > 1e-9
    if not np.any(nz):
        return 0.0
    return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "MAPE (%)": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE (%)": mape}


# ============================================================
# GEO HELPERS
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance (km)."""
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def filter_by_region(df, region_name, radius_km=2.5):
    """Return stations within radius_km of the region center. If All NYC, return df."""
    if df.empty:
        return df
    if region_name == "All NYC (Default)":
        return df.copy()

    center = LOCATIONS[region_name]
    d = df.copy()
    d["dist_km"] = haversine_km(d["lat"], d["lon"], center["lat"], center["lon"])
    return d[d["dist_km"] <= radius_km].copy()


# ============================================================
# CONGESTION RISK (LIVE)
# ============================================================
def risk_badge(row):
    """
    Risk based on operational imbalance:
    - HIGH: station empty or full (no bikes or no docks) OR very extreme full %
    - MED: moderately high utilization
    - LOW: normal range
    """
    bikes = float(row.get("num_bikes_available", 0))
    docks = float(row.get("num_docks_available", 0))
    pct = float(row.get("percent_full", 0))

    if bikes <= 0 or docks <= 0 or pct >= 90 or pct <= 10:
        return "HIGH"
    if pct >= 75 or pct <= 25:
        return "MED"
    return "LOW"


# ============================================================
# LIVE DATA (GBFS) — with timeouts so it won’t hang forever
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def load_live_data():
    """
    Fetch live Citi Bike station info + status and merge.
    Adds: color, classic_bikes, ebikes, percent_full
    """
    try:
        info_resp = requests.get(STATION_INFO_URL, timeout=12)
        status_resp = requests.get(STATION_STATUS_URL, timeout=12)

        info = info_resp.json()["data"]["stations"]
        status = status_resp.json()["data"]["stations"]

        info_df = pd.DataFrame(info)
        status_df = pd.DataFrame(status)

        info_df["station_id"] = info_df["station_id"].astype(str)
        status_df["station_id"] = status_df["station_id"].astype(str)

        df = pd.merge(info_df, status_df, on="station_id", suffixes=("_info", "_status"))

        # ✅ REMOVE DOUBLE QUOTES FROM LIVE STATION NAMES
        df["name"] = df["name"].astype(str).str.replace('"', '', regex=False).str.strip()

        df["name_clean"] = (
            df["name"]
            .str.lower()
            .str.replace("’", "'", regex=False)
        )

        df["capacity"] = pd.to_numeric(df.get("capacity", 0), errors="coerce").fillna(0).astype(float)
        df["capacity"] = df["capacity"].replace(0, np.nan)

        df["num_bikes_available"] = pd.to_numeric(df.get("num_bikes_available", 0), errors="coerce").fillna(0).astype(
            float)
        df["num_docks_available"] = pd.to_numeric(df.get("num_docks_available", 0), errors="coerce").fillna(0).astype(
            float)
        df["num_ebikes_available"] = pd.to_numeric(df.get("num_ebikes_available", 0), errors="coerce").fillna(0).astype(
            float)

        df["percent_full"] = (df["num_bikes_available"] / df["capacity"]) * 100.0
        df["percent_full"] = df["percent_full"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df["ebikes"] = df["num_ebikes_available"].clip(lower=0)
        df["classic_bikes"] = (df["num_bikes_available"] - df["ebikes"]).clip(lower=0)

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])

        def get_color(row):
            if int(row.get("is_installed", 1)) == 0 or int(row.get("is_renting", 1)) == 0:
                return [128, 128, 128, 200]  # offline
            if row["num_bikes_available"] <= 0:
                return [255, 0, 0, 200]  # empty
            if row["num_docks_available"] <= 0:
                return [0, 0, 255, 200]  # full
            return [0, 200, 0, 200]  # normal

        df["color"] = df.apply(get_color, axis=1)
        df["risk"] = df.apply(risk_badge, axis=1)

        return df

    except Exception as e:
        st.error(f"Live feed error (GBFS): {e}")
        return pd.DataFrame()


# ============================================================
# HISTORICAL DATA LOADING — your processed occupancy CSVs
# ============================================================
@st.cache_data(show_spinner=False)
def load_historical_data(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True))
    if not files:
        return pd.DataFrame(), []
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(), files
    return pd.concat(frames, ignore_index=True), files





DEFAULT_REGION = "Jersey City / Hoboken"

# ================= LOAD MODEL =================
@st.cache_resource
def load_xgb():
    if not os.path.exists(XGB_MODEL_PATH):
        return None
    return joblib.load(XGB_MODEL_PATH)

# PRETRAINED_XGB = load_xgb()

PRETRAINED_XGB = load_xgb()
if PRETRAINED_XGB is not None:
    PRETRAINED_XGB = sanitize_xgb_model(PRETRAINED_XGB)





# ================= METRICS =================
def safe_mape(y_true, y_pred):
    nz = np.abs(y_true) > 1e-9
    return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if np.any(nz) else 0.0

def compute_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE (%)": safe_mape(y_true, y_pred)
    }

# ================= DATA LOADING =================
@st.cache_data
def load_hist_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["dock_name"] = df["dock_name"].astype(str).str.replace('"', '').str.strip()
    return df

df_hist = load_hist_data()
stations_hist = sorted(df_hist["dock_name"].unique())

def coerce_hist_types(df):
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp", "dock_name", "occupancy"])

    # ✅ REMOVE DOUBLE QUOTES + NORMALIZE
    df["dock_name"] = (
        df["dock_name"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
    )

    # optional but STRONGLY recommended
    df["dock_name_clean"] = (
        df["dock_name"]
        .str.lower()
        .str.replace("’", "'", regex=False)
    )

    for c in [
        "_lat", "_long", "avail_bikes", "avail_docks", "tot_docks",
        "occupancy", "is_holiday", "is_weekend",
        "temperature_2m", "relative_humidity_2m",
        "precipitation", "wind_speed_10m"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ================= FEATURE ENGINEERING =================
def build_station_hourly_ts(df, station):
    d = df[df["dock_name"] == station].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.set_index("timestamp").sort_index()
    numeric_cols = d.select_dtypes(include=[np.number]).columns
    hourly = d[numeric_cols].resample("1H").mean()

    hourly["hour"] = hourly.index.hour
    hourly["dayofweek"] = hourly.index.dayofweek
    hourly["month"] = hourly.index.month
    hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)
    hourly["is_holiday"] = hourly.get("is_holiday", 0)

    hourly["lag_1"] = hourly["occupancy"].shift(1)
    hourly["lag_24"] = hourly["occupancy"].shift(24)
    hourly["roll_3"] = hourly["occupancy"].rolling(3).mean()
    hourly["roll_24"] = hourly["occupancy"].rolling(24).mean()

    return hourly.dropna().reset_index()

def build_pretrained_features(df, station):
    d = df[df["dock_name"] == station].copy()
    d = d.set_index("timestamp").sort_index()
    numeric_cols = d.select_dtypes(include=[np.number]).columns
    hourly = d[numeric_cols].resample("1H").mean().reset_index()

    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["dayofweek"] = hourly["timestamp"].dt.dayofweek
    hourly["month"] = hourly["timestamp"].dt.month
    hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)
    hourly["is_holiday"] = 0

    hourly["lag_occ_1"] = hourly["occupancy"].shift(1)
    hourly["lag_occ_24"] = hourly["occupancy"].shift(24)
    hourly["roll_occ_3"] = hourly["occupancy"].rolling(3).mean()
    hourly["roll_occ_24"] = hourly["occupancy"].rolling(24).mean()

    hourly = hourly.dropna()
    for c in PRETRAINED_FEATURES:
        if c not in hourly.columns:
            hourly[c] = 0.0
    return hourly






############===================Script for live data ==========================

#
# # ============================================================
# # SIDEBAR
# # ============================================================
st.sidebar.header("Controls")
show_empty = st.sidebar.checkbox("Show Empty (No Bikes) — 🔴", True)
show_full = st.sidebar.checkbox("Show Full (No Docks) — 🔵", True)
show_normal = st.sidebar.checkbox("Show Normal — 🟢", True)
show_offline = st.sidebar.checkbox("Show Offline — ⚫", False)

st.sidebar.markdown("---")

region_radius_km = st.sidebar.slider("Region radius (km)", 1.0, 6.0, 6.0, 0.5)

st.sidebar.markdown("---")

st.sidebar.markdown("### Historical Data Checks")
show_missing_hist_stations = st.sidebar.checkbox(
    "Show historical stations not in live data",
    value=False
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================
with st.spinner("Loading historical dataset from ./data ..."):
    df_hist_raw, hist_files = load_historical_data(DATA_DIR)

if df_hist_raw.empty:
    st.error("No historical CSVs found/loaded from ./data. Put your processed occupancy CSV(s) into a `data/` folder.")
    st.stop()

df_hist = coerce_hist_types(df_hist_raw)
stations_hist = sorted(df_hist["dock_name"].astype(str).dropna().unique().tolist())
if not stations_hist:
    st.error("No dock_name values found in historical data.")
    st.stop()

with st.spinner("Fetching live Citi Bike feed..."):
    df_live = load_live_data()

# ============================================================
# HISTORICAL STATIONS NOT PRESENT IN LIVE DATA (SIDEBAR CONTROLLED)
# ============================================================

if show_missing_hist_stations:

    hist_station_set = set(
        df_hist["dock_name"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
    )

    live_station_set = set(
        df_live["name"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
    )

    missing_in_live = sorted(hist_station_set - live_station_set)

    st.markdown("## 📛 Historical Stations NOT Present in Live Data")

    if not missing_in_live:
        st.success("✅ All historical stations are present in the live GBFS feed.")
    else:
        st.warning(
            f"⚠️ {len(missing_in_live)} station(s) found in historical data but missing from live feed."
        )

        missing_df = pd.DataFrame(
            {"Station Name (Historical Only)": missing_in_live}
        )

        st.dataframe(
            missing_df,
            use_container_width=True,
            height=400
        )

        st.download_button(
            label="📥 Download Missing Stations (CSV)",
            data=missing_df.to_csv(index=False),
            file_name="stations_missing_in_live_feed.csv",
            mime="text/csv"
        )

# # ============================================================





if df_live.empty:
    st.warning("Live feed not available right now (timeout/network). Historical modeling still works below.")
else:
    live = df_live.copy()

    # offline (gray)
    is_offline = (live.get("is_installed", 1).astype(int) == 0) | (live.get("is_renting", 1).astype(int) == 0)
    if not show_offline:
        live = live[~is_offline]

    if not show_empty:
        live = live[live["num_bikes_available"] > 0]
    if not show_full:
        live = live[live["num_docks_available"] > 0]

    is_normal = (live["num_bikes_available"] > 0) & (live["num_docks_available"] > 0)
    if not show_normal:
        live = live[~is_normal]

    # KPIs
    ############# UI  ###################
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations shown (all filters)", len(live))
    c2.metric("Total bikes", int(live["num_bikes_available"].sum()))
    c3.metric("E-bikes", int(live["ebikes"].sum()))
    c4.metric("Last update", time.strftime("%H:%M"))

    ############# UI  ###################
    st.caption("")

    ##################################################################################################################
    st.sidebar.markdown("---")
    ##################################################################################################################

    # region default = Jersey City / Hoboken
    region_keys = list(LOCATIONS.keys())
    default_region_idx = region_keys.index(DEFAULT_REGION) if DEFAULT_REGION in region_keys else 0

    colA, colB = st.columns([0.55, 0.45])
    with colA:
        loc_select = st.selectbox("Jump to region", region_keys, index=default_region_idx)
    with colB:
        # region-filtered stations for dropdown
        region_live = filter_by_region(live, loc_select, radius_km=region_radius_km)
        if region_live.empty:
            st.warning("No stations found in this region with current filters.")
            station_live_name = None
        else:
            station_opts = sorted(region_live["name"].astype(str).unique().tolist())
            station_live_name = st.selectbox(
                f"Station details (live feed) — {loc_select}",
                options=station_opts,
                index=0,
            )

    st.caption("Add some description")

    # ============================================================
    # TOP 5 HIGH RISK STATIONS (REGION LEVEL)
    # ============================================================
    st.markdown(f"### 🚨 Top 5 HIGH Congestion Risk Stations — {loc_select}")
    st.caption("Add some description")

    high_risk_df = (
        region_live[region_live["risk"] == "HIGH"]
        .copy()
    )

    if high_risk_df.empty:
        st.success("✅ No HIGH congestion risk stations in this region right now.")
    else:
        # Rank by most problematic first
        high_risk_df["severity_score"] = (
            high_risk_df["percent_full"].abs()
        )

        top5_high_risk = (
            high_risk_df
            .sort_values(
                by=["num_bikes_available", "num_docks_available", "severity_score"],
                ascending=[True, True, False]
            )
            .head(5)
        )

        display_cols = [
            "name",
            "num_bikes_available",
            "num_docks_available",
            "percent_full",
            "classic_bikes",
            "ebikes",
            "risk",
        ]

        st.dataframe(
            top5_high_risk[display_cols]
            .rename(columns={
                "name": "Station",
                "num_bikes_available": "Bikes",
                "num_docks_available": "Docks",
                "percent_full": "% Full",
                "classic_bikes": "Classic Bikes",
                "ebikes": "E-Bikes",
                "risk": "Risk Level",
            }),
            use_container_width=True,
            hide_index=True
        )

    ##############=============================================================
    # selected station details ONLY from region
    if station_live_name:
        sel = region_live[region_live["name"].astype(str) == str(station_live_name)].copy()
    else:
        sel = pd.DataFrame()

    # map center
    if not sel.empty:
        center_lat = float(sel["lat"].iloc[0])
        center_lon = float(sel["lon"].iloc[0])
        zoom = 15
    else:
        center_lat = LOCATIONS[loc_select]["lat"]
        center_lon = LOCATIONS[loc_select]["lon"]
        zoom = LOCATIONS[loc_select]["zoom"]

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)

    # ============================================================
    # LIVE MAP SECTION
    # ============================================================


    st.subheader("📍 Interactive City Map")
    # st.caption("Live Citi Bike station status (GBFS). Region filter drives station dropdown + Top Stations chart.")
    st.caption("Add some description")


    # ============================================================

    st.info("🔴 Red = No bikes | 🔵 Blue = No docks | 🟢 Green = Normal | ⚫ Gray = Offline | 🚦 Risk = LOW/MED/HIGH")

    # add risk badge to tooltip
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    region_live if not region_live.empty else live,
                    get_position=["lon", "lat"],
                    get_color="color",
                    get_radius=85,
                    pickable=True,
                    opacity=0.85,
                    filled=True,
                )
            ],
            tooltip={
                "html": """
                <b>{name}</b><br/>
                🚦 Risk: <b>{risk}</b><br/>
                🚲 Bikes: {num_bikes_available}<br/>
                ⚡ E-bikes: {ebikes}<br/>
                🅿️ Docks: {num_docks_available}<br/>
                📦 Capacity: {capacity}<br/>
                % Full: {percent_full}%
                """
            },
        )
    )

    # ============================================================
    # Station detail table (region-aware) + badge
    st.markdown(f"#### 🧾 Selected Station (Live) — {loc_select}")
    if sel.empty:
        st.write("No station selected (or no stations in region).")
    else:
        risk = str(sel["risk"].iloc[0])
        if risk == "HIGH":
            st.error("🚦 Congestion Risk: HIGH")
        elif risk == "MED":
            st.warning("🚦 Congestion Risk: MED")
        else:
            st.success("🚦 Congestion Risk: LOW")

        detail = sel[[
            "name",
            "station_id",
            "lat", "lon",
            "capacity",
            "num_bikes_available",
            "classic_bikes",
            "ebikes",
            "num_docks_available",
            "percent_full",
            "risk",
        ]].copy()

        detail = detail.rename(columns={
            "name": "station_name",
            "num_bikes_available": "bikes_total",
            "num_docks_available": "docks_available",
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

    # Region-level CSV download (snapshot)
    st.download_button(
        label=f"📥 Download Region Snapshot CSV — {loc_select}",
        data=(region_live if not region_live.empty else live).to_csv(index=False),
        file_name=f"live_region_snapshot_{loc_select.replace(' ', '_').replace('/', '-')}.csv",
        mime="text/csv",
    )

    ###### ============================================================

    # Top stations chart — REGION DRIVEN
    st.markdown(f"#### 📊 Top Stations (Live) — Bikes by Type ({loc_select})")

    ###### added a new - image here

    import base64


    def render_section_image(image_path, height=420):
        with open(image_path, "rb") as f:
            img_bytes = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <div style="margin-top:40px; margin-bottom:40px;">
                <img src="data:image/jpeg;base64,{img_bytes}"
                     style="
                         width:100%;
                         max-height:{height}px;
                         object-fit:cover;
                         border-radius:16px;
                         box-shadow:0px 6px 18px rgba(0,0,0,0.15);
                     ">
            </div>
            """,
            unsafe_allow_html=True
        )


    render_section_image("img01.jpg")

    ###### ============================================================

    if region_live.empty:
        st.warning("No stations available in region for Top Stations chart.")
    else:
        top_n = 12
        top = region_live.sort_values("num_bikes_available", ascending=False).head(top_n).copy()
        top_long = pd.melt(
            top,
            id_vars=["name"],
            value_vars=["classic_bikes", "ebikes"],
            var_name="BIKE_TYPE",
            value_name="count",
        )
        top_long["BIKE_TYPE"] = top_long["BIKE_TYPE"].replace({"classic_bikes": "classic", "ebikes": "electric"})
        fig_top = px.bar(
            top_long,
            y="name",
            x="count",
            color="BIKE_TYPE",
            orientation="h",
            title=f"Top Stations in {loc_select} by Total Bikes Available (split by type)",
        )
        fig_top.update_layout(yaxis_title="", xaxis_title="Bikes available")
        st.plotly_chart(fig_top, use_container_width=True)

st.markdown("---")

# ================= FORECASTING =================
# st.subheader("📈 Congestion Forecasting")
st.subheader("📈 Congestion Forecasting - ARIMA & XGBoost Model")
st.caption(
    "Historical occupancy forecasting using hourly-resampled station data. "
    "Models: XGBoost (tree regression) + ARIMA (time-series baseline)."
)

station_hist = st.selectbox(
    "Select station",
    stations_hist,
    index=stations_hist.index(DEFAULT_HIST_STATION)
)

ts = build_station_hourly_ts(df_hist, station_hist)
if len(ts) < MIN_ROWS_PER_STATION:
    st.warning("Not enough data")
    st.stop()

split = pd.Timestamp("2018-03-01")
train = ts[ts["timestamp"] < split]
test = ts[ts["timestamp"] >= split]

y_test = test["occupancy"].values
timestamps = test["timestamp"].values

preds = {}
metrics = []

# ---- XGBOOST ----
if PRETRAINED_XGB is not None:
    feat = build_pretrained_features(df_hist, station_hist)
    Xp = feat[PRETRAINED_FEATURES].fillna(0)
    yhat = PRETRAINED_XGB.predict(Xp)[:, 2][-len(y_test):]
    preds["XGBoost"] = yhat
    metrics.append({"Model": "XGBoost", **compute_metrics(y_test, yhat)})

# ---- ARIMA ----
if HAS_STATSMODELS:
    exog = BASE_FEATURES
    ar = SARIMAX(
        train["occupancy"],
        exog=train[exog],
        order=(1,0,1),
        seasonal_order=(1,0,1,24),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    yhat = ar.get_forecast(
        steps=len(test),
        exog=test[exog]
    ).predicted_mean.values

    preds["ARIMA"] = yhat
    metrics.append({"Model": "ARIMA", **compute_metrics(y_test, yhat)})

# ================= RESULTS =================
metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
st.dataframe(metrics_df, use_container_width=True)

plot_df = pd.DataFrame({"timestamp": timestamps, "Actual": y_test})
for k,v in preds.items():
    plot_df[k] = v

# st.plotly_chart(
#     px.line(plot_df, x="timestamp", y=plot_df.columns.drop("timestamp")),
#     use_container_width=True
# )

# =======================================================================================================================
# ========================================Best performing model: ========================================================

best_model = metrics_df.iloc[0]["Model"] if not metrics_df.empty else None
if best_model:
    st.success(f"✅ Best performing model: **{best_model}**")

# ============================================================
# Build Forecast Plot DataFrame
# ============================================================
# ✅ FIX: define ts_test explicitly (same as old train_models output)
ts_test = test["timestamp"]
split = pd.Timestamp("2018-03-01")

ts = ts.copy()
ts["timestamp"] = pd.to_datetime(ts["timestamp"])

train = ts[ts["timestamp"] < split]
test  = ts[ts["timestamp"] >= split]

if train.empty or test.empty:
    st.warning("⚠️ Not enough samples after train/test split.")
    st.stop()

# ✅ REQUIRED VARIABLES (MATCH OLD CONTRACT)
ts_test = test["timestamp"]     # ← ADD THIS LINE
y_test  = test["occupancy"]

plot_df = pd.DataFrame({
    "timestamp": ts_test.values,
    "Actual": y_test.values
})

for model_name, yhat in preds.items():
    if yhat is not None:
        plot_df[model_name] = yhat

# ============================================================
# 🔧 Forecast Visualization Controls
# ============================================================

st.markdown("**Select Forecasting Model(s)**")

available_models = [c for c in plot_df.columns if c not in ["timestamp", "Actual"]]

model_checks = {}
for model in available_models:
    model_checks[model] = st.checkbox(model, value=True)

selected_models = [m for m, v in model_checks.items() if v]

if not selected_models:
    st.warning("Please select at least one forecasting model to display.")
    st.stop()


# ============================================================
# DATE RANGE SLIDER (MONTH–YEAR)
# ============================================================

plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])

min_date = plot_df["timestamp"].min().to_pydatetime()
max_date = plot_df["timestamp"].max().to_pydatetime()

default_start = pd.Timestamp("2018-03-01").to_pydatetime()
default_end   = pd.Timestamp("2018-09-30").to_pydatetime()

start_date, end_date = st.slider(
    "Select Date Range (Month-Year)",
    min_value=min_date,
    max_value=max_date,
    value=(default_start, default_end),
    format="MMM YYYY"
)

filtered_df = plot_df[
    (plot_df["timestamp"] >= start_date) &
    (plot_df["timestamp"] <= end_date)
].copy()

# ============================================================
# st.markdown("📈 Actual vs Predicted Occupancy")
# st.subheader("📈 Actual vs Predicted Occupancy")
# ============================================================

st.plotly_chart(
    px.line(
        filtered_df,
        x="timestamp",
        y=["Actual"] + selected_models,
        title=f"Actual vs Predicted Occupancy — {station_hist}",
        labels={
            "value": "Occupancy",
            "timestamp": "Time"
        }
    ),
    use_container_width=True
)



# ============================================================

st.subheader("📈 Actual vs Predicted Occupancy (Station Level)")

# ============================================================

st.plotly_chart(
    px.line(
        filtered_df,
        x="timestamp",
        y=["Actual", "XGBoost"],
        labels={"value": "Occupancy", "timestamp": "Time"},
        title=f"XGBoost — Actual vs Predicted Occupancy ({station_hist})"
    ),
    use_container_width=True
)

# ============================================================

st.subheader("🌆 Aggregate Forecast (City-Level Mean)")

# ============================================================

agg_df = plot_df.groupby("timestamp").mean(numeric_only=True).reset_index()

st.plotly_chart(
    px.line(
        agg_df,
        x="timestamp",
        y=["Actual", "XGBoost"],
        title="City-Level Mean Occupancy — XGBoost",
        labels={"value": "Mean Occupancy"}
    ),
    use_container_width=True
)

#========================================================

st.subheader("🎯 Predicted vs Actual (Scatter)")

#========================================================
scatter_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": preds["XGBoost"]
})

fig = px.scatter(
    scatter_df,
    x="Actual",
    y="Predicted",
    opacity=0.4,
    title="XGBoost — Predicted vs Actual Occupancy"
)

fig.add_shape(
    type="line",
    x0=scatter_df["Actual"].min(),
    y0=scatter_df["Actual"].min(),
    x1=scatter_df["Actual"].max(),
    y1=scatter_df["Actual"].max(),
    line=dict(color="red", dash="dash")
)

st.plotly_chart(fig, use_container_width=True)





# ============================================================
#
st.subheader("📉 Residual Distribution")

# ============================================================
#
residuals = y_test - preds["XGBoost"]

fig = px.histogram(
    residuals,
    nbins=60,
    marginal="box",
    title="XGBoost Residual Distribution (Actual − Predicted)",
    labels={"value": "Residual"}
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================

st.subheader("🔥 Error Density Heatmap")

# ============================================================

heat_df = pd.DataFrame({
    "Actual": y_test,
    "Error": residuals
})

fig = px.density_heatmap(
    heat_df,
    x="Actual",
    y="Error",
    nbinsx=40,
    nbinsy=40,
    color_continuous_scale="Viridis",
    title="XGBoost — Error Density Heatmap"
)

st.plotly_chart(fig, use_container_width=True)
# ============================================================
#
st.subheader("📊 Model Performance Comparison")
# ============================================================
#
metric_choice = st.selectbox(
    "Select Metric",
    ["RMSE", "MAE", "R2", "MAPE (%)"]
)

fig = px.bar(
    metrics_df,
    x="Model",
    y=metric_choice,
    color="Model",
    title=f"{metric_choice} Comparison"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
#
st.subheader("🧠 XGBoost Feature Importance (Occupancy)")
#
# ============================================================
#
xgb_occ = PRETRAINED_XGB.estimators_[2]

importance_df = pd.DataFrame({
    "Feature": PRETRAINED_FEATURES,
    "Importance": xgb_occ.feature_importances_
}).sort_values("Importance", ascending=False)

st.plotly_chart(
    px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="XGBoost Feature Importance — Occupancy"
    ),
    use_container_width=True
)


# ============================================================
#
st.subheader("⏰ Error vs Hour of Day")
#
# ============================================================
#
error_hour_df = test.copy()
error_hour_df["abs_error"] = np.abs(y_test - preds["XGBoost"])
error_hour_df["hour"] = error_hour_df["timestamp"].dt.hour

fig = px.box(
    error_hour_df,
    x="hour",
    y="abs_error",
    title="XGBoost Absolute Error by Hour of Day",
    labels={"abs_error": "Absolute Error"}
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
#
if "precipitation" in test.columns:
    st.subheader("🌧️ Error vs Precipitation")

    weather_df = test.copy()
    weather_df["abs_error"] = np.abs(y_test - preds["XGBoost"])

    fig = px.scatter(
        weather_df,
        x="precipitation",
        y="abs_error",
        opacity=0.4,
        title="XGBoost Error vs Precipitation"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
#st.subheader("🏆 Best & Worst Predictions (Occupancy)")

err_df = test.copy()
err_df["Actual"] = y_test
err_df["Predicted"] = preds["XGBoost"]
err_df["Error"] = np.abs(err_df["Actual"] - err_df["Predicted"])

best20 = err_df.sort_values("Error").head(20)
worst20 = err_df.sort_values("Error", ascending=False).head(20)

tab1, tab2 = st.tabs(["✅ Best 20", "❌ Worst 20"])

with tab1:
    st.dataframe(best20[["timestamp", "Actual", "Predicted", "Error"]])

with tab2:
    st.dataframe(worst20[["timestamp", "Actual", "Predicted", "Error"]])

# ============================================================
#
# ============================================================
#
# ============================================================
#
# ============================================================
# 🔹 RISK LEGEND COUNTS (REGION LEVEL)
# ============================================================

st.subheader("🔹RISK LEGEND COUNTS (REGION LEVEL)")

if not region_live.empty:
    risk_counts = (
        region_live["risk"]
        .value_counts()
        .reindex(["LOW", "MED", "HIGH"], fill_value=0)
    )

    rc1, rc2, rc3 = st.columns(3)

    rc1.metric("🟢 LOW Risk Stations", int(risk_counts["LOW"]))
    rc2.metric("🟡 MED Risk Stations", int(risk_counts["MED"]))
    rc3.metric("🔴 HIGH Risk Stations", int(risk_counts["HIGH"]))

else:
    st.info("No stations available to compute congestion risk for this region.")


# =======================================================================================================================
# ============================================  LSTM   ===========================================================================


# ============================================================
# LSTM Forecasting — NYC Citi Bike Congestion Intelligence
# Pretrained Global Multi-Output LSTM (Inference Only)
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
st.caption("Pretrained Global Multi-Output LSTM | Station-Level Forecasting")

FEATURES = [
    "_lat", "_long", "tot_docks",
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "lag_occ_1", "lag_occ_24", "roll_occ_3", "roll_occ_24",
    "temperature_2m", "relative_humidity_2m",
    "precipitation", "wind_speed_10m"
]

TARGETS = ["avail_bikes", "avail_docks", "occupancy"]

WINDOW = 48  # MUST match training

def build_global_hourly(df):
    hourly = (
        df.set_index("timestamp")
        .groupby("dock_name")
        .resample("1H")
        .mean(numeric_only=True)
        .reset_index()
    )

    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["dayofweek"] = hourly["timestamp"].dt.dayofweek
    hourly["month"] = hourly["timestamp"].dt.month
    hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)
    hourly["is_holiday"] = hourly.get("is_holiday", 0)

    hourly["lag_occ_1"] = hourly.groupby("dock_name")["occupancy"].shift(1)
    hourly["lag_occ_24"] = hourly.groupby("dock_name")["occupancy"].shift(24)

    hourly["roll_occ_3"] = (
        hourly.groupby("dock_name")["occupancy"]
        .rolling(3).mean().reset_index(0, drop=True)
    )
    hourly["roll_occ_24"] = (
        hourly.groupby("dock_name")["occupancy"]
        .rolling(24).mean().reset_index(0, drop=True)
    )

    return hourly.dropna().reset_index(drop=True)

from sklearn.preprocessing import StandardScaler

@st.cache_resource
def rebuild_lstm_scalers(df_hist):
    df = df_hist.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "dock_name"])

    hourly_global = build_global_hourly(df)

    X = hourly_global[FEATURES].values
    y = hourly_global[TARGETS].values

    return StandardScaler().fit(X), StandardScaler().fit(y)

@st.cache_resource
def load_lstm_assets(df_hist):
    from tensorflow.keras import layers, models
    import tensorflow as tf
    import os

    tf.keras.mixed_precision.set_global_policy("float32")

    model = models.Sequential([
        layers.Input(shape=(WINDOW, len(FEATURES))),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(TARGETS))
    ])

    model.compile(optimizer="adam", loss="mse")

    model.load_weights("model_export/lstm_multioutput.h5")

    x_scaler, y_scaler = rebuild_lstm_scalers(df_hist)
    return model, x_scaler, y_scaler

st.markdown("---")
st.subheader("🧠 LSTM Forecasting (Global Model)")

lstm_station = st.selectbox("Select station", stations_hist)
split_date = pd.Timestamp("2018-03-01")

lstm_model, x_scaler, y_scaler = load_lstm_assets(df_hist)

hourly_global = build_global_hourly(df_hist)
hourly_station = hourly_global[
    hourly_global["dock_name"] == lstm_station
].copy()

hourly_station = hourly_station.sort_values("timestamp").reset_index(drop=True)

X_raw = hourly_station[FEATURES].values
y_true = hourly_station[TARGETS].values

X_scaled = x_scaler.transform(X_raw)

def make_windows(X, window):
    return np.array([X[i:i+window] for i in range(len(X)-window)])

X_seq = make_windows(X_scaled, WINDOW)
y_pred = y_scaler.inverse_transform(
    lstm_model.predict(X_seq, verbose=0)
)

aligned = hourly_station.iloc[WINDOW:].copy()
aligned["pred_occupancy"] = y_pred[:, 2]

test = aligned[aligned["timestamp"] >= split_date]

st.plotly_chart(
    px.line(
        test,
        x="timestamp",
        y=["occupancy", "pred_occupancy"],
        title=f"LSTM — Occupancy Forecast ({lstm_station})"
    ),
    use_container_width=True
)

# ================= LSTM PLOT DATA =================
lstm_plot_df = aligned.copy()

lstm_plot_df = lstm_plot_df.rename(columns={
    "occupancy": "Actual",
    "pred_occupancy": "LSTM"
})

lstm_plot_df["timestamp"] = pd.to_datetime(lstm_plot_df["timestamp"])

# Cap to Sep 2018 (to match XGBoost/ARIMA UI)
lstm_plot_df = lstm_plot_df[
    lstm_plot_df["timestamp"] <= pd.Timestamp("2018-09-30")
].copy()

# ✅ STEP 2 — 📈 LSTM Actual vs Predicted (Station Level) + Date Slider
st.subheader("📈 LSTM Actual vs Predicted Occupancy (Station Level)")

min_date = lstm_plot_df["timestamp"].min().to_pydatetime()
max_date = lstm_plot_df["timestamp"].max().to_pydatetime()

start_date, end_date = st.slider(
    "Select Date Range (Month-Year) — LSTM",
    min_value=min_date,
    max_value=max_date,
    value=(pd.Timestamp("2018-03-01").to_pydatetime(),
           pd.Timestamp("2018-09-30").to_pydatetime()),
    format="MMM YYYY",
    key="lstm_date_slider_station"
)

lstm_filtered = lstm_plot_df[
    (lstm_plot_df["timestamp"] >= start_date) &
    (lstm_plot_df["timestamp"] <= end_date)
]

st.plotly_chart(
    px.line(
        lstm_filtered,
        x="timestamp",
        y=["Actual", "LSTM"],
        title=f"LSTM — Actual vs Predicted Occupancy ({lstm_station})",
        labels={"value": "Occupancy", "timestamp": "Time"}
    ),
    use_container_width=True
)

# ✅ STEP 3 — 🌆 LSTM Aggregate Forecast (City-Level Mean) + Slider
st.subheader("🌆 LSTM Aggregate Forecast (City-Level Mean)")

agg_lstm = (
    lstm_plot_df
    .groupby("timestamp")[["Actual", "LSTM"]]
    .mean()
    .reset_index()
)

start_date_agg, end_date_agg = st.slider(
    "Select Date Range (Month-Year) — LSTM Aggregate",
    min_value=min_date,
    max_value=max_date,
    value=(pd.Timestamp("2018-03-01").to_pydatetime(),
           pd.Timestamp("2018-09-30").to_pydatetime()),
    format="MMM YYYY",
    key="lstm_date_slider_agg"
)

agg_filtered = agg_lstm[
    (agg_lstm["timestamp"] >= start_date_agg) &
    (agg_lstm["timestamp"] <= end_date_agg)
]

st.plotly_chart(
    px.line(
        agg_filtered,
        x="timestamp",
        y=["Actual", "LSTM"],
        title="LSTM — City-Level Mean Occupancy",
        labels={"value": "Mean Occupancy"}
    ),
    use_container_width=True
)

# ✅ STEP 4 — 🎯 LSTM Predicted vs Actual (Scatter)
st.subheader("🎯 LSTM Predicted vs Actual (Scatter)")

fig = px.scatter(
    lstm_plot_df,
    x="Actual",
    y="LSTM",
    opacity=0.4,
    title="LSTM — Predicted vs Actual Occupancy"
)

fig.add_shape(
    type="line",
    x0=lstm_plot_df["Actual"].min(),
    y0=lstm_plot_df["Actual"].min(),
    x1=lstm_plot_df["Actual"].max(),
    y1=lstm_plot_df["Actual"].max(),
    line=dict(color="red", dash="dash")
)

st.plotly_chart(fig, use_container_width=True)

# ✅ STEP 5 — 📉 LSTM Residual Distribution
st.subheader("📉 LSTM Residual Distribution")

lstm_residuals = lstm_plot_df["Actual"] - lstm_plot_df["LSTM"]

st.plotly_chart(
    px.histogram(
        lstm_residuals,
        nbins=60,
        marginal="box",
        title="LSTM Residual Distribution (Actual − Predicted)"
    ),
    use_container_width=True
)

# ✅ STEP 6 — 🔥 LSTM Error Density Heatmap
st.subheader("🔥 LSTM Error Density Heatmap")

heat_df = pd.DataFrame({
    "Actual": lstm_plot_df["Actual"],
    "Error": lstm_residuals
})

st.plotly_chart(
    px.density_heatmap(
        heat_df,
        x="Actual",
        y="Error",
        nbinsx=40,
        nbinsy=40,
        color_continuous_scale="Viridis",
        title="LSTM — Error Density Heatmap"
    ),
    use_container_width=True
)

# ✅ STEP 7 — ⏰ LSTM Error vs Hour of Day
st.subheader("⏰ LSTM Error vs Hour of Day")

hour_df = lstm_plot_df.copy()
hour_df["hour"] = hour_df["timestamp"].dt.hour
hour_df["abs_error"] = np.abs(hour_df["Actual"] - hour_df["LSTM"])

st.plotly_chart(
    px.box(
        hour_df,
        x="hour",
        y="abs_error",
        title="LSTM Absolute Error by Hour of Day",
        labels={"abs_error": "Absolute Error"}
    ),
    use_container_width=True
)

# ✅ STEP 8 — ✅ Best / ❌ Worst 20 (LSTM)
st.subheader("🏆 LSTM Best & Worst Predictions")

err_df = lstm_plot_df.copy()
err_df["Error"] = np.abs(err_df["Actual"] - err_df["LSTM"])

best20 = err_df.sort_values("Error").head(20)
worst20 = err_df.sort_values("Error", ascending=False).head(20)

tab1, tab2 = st.tabs(["✅ Best 20 (LSTM)", "❌ Worst 20 (LSTM)"])

with tab1:
    st.dataframe(best20[["timestamp", "Actual", "LSTM", "Error"]])

with tab2:
    st.dataframe(worst20[["timestamp", "Actual", "LSTM", "Error"]])


# ================= Citi Bike Image =================
st.markdown(
    """
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <img src="https://images.ctfassets.net/p6ae3zqfb1e3/3qiREUnpfzi8XvYlba1Xx1/579054ff61fa8e05919c928a4b45b4d8/01-CITI_BIKE-MAIN.svg"
             width="220">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# ================= Citi Bike Image =================


