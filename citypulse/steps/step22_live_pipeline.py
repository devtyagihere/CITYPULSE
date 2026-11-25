# =============================================================================
# STEP NAME  : step22 — Live Prediction Pipeline
# WHAT       : Fetches live weather from api.weather.gov (KNYC Central Park,
#              using a proper User-Agent header), replays test traffic in
#              "Simulated Input" mode, uses the SAME feature code (common.py),
#              and saves predictions with source name and timestamp.
# WHY        : Training is offline, but CITYPULSE must run continuously on
#              incoming data without data leakage or silent fake-data fallbacks.
#              Missing live fields are marked "unavailable", never faked!
# INPUT      : Live API (api.weather.gov/stations/KNYC/observations/latest)
#              data/processed/traffic_clean.parquet
#              data/processed/weather_clean.parquet
#              data/processed/usual_speed_table.parquet
#              data/processed/adj_matrix.npy, sensor_order.csv, sensor_to_edge.csv
#              models/lgb_horizon_{15,30,45,60}m.joblib, crisis_lgb.joblib, aqi_lgb_1h.joblib
# OUTPUT     : data/processed/live_snapshot.json
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os, json
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import joblib
import requests
from config import (DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
                    CRISIS_CONGESTED_RATIO, TIMEZONE)
from common import build_traffic_features, pm25_to_aqi_category

print("=" * 65)
print("  STEP 22 — Live Prediction Pipeline")
print("=" * 65)

FEATURE_COLS = [
    "lag_0", "lag_1", "lag_2", "lag_4",
    "roll_mean_1h", "roll_std_1h",
    "usual_speed_ratio",
    "neighbour_speed_ratio",
    "hour", "weekday", "is_weekend", "is_holiday",
    "temp", "precip", "wind",
    "speed_was_missing",
]


def fetch_live_weather_knyc():
    """
    Fetch latest weather from NOAA/NWS Central Park station (KNYC).
    Requires a User-Agent header. Never invents fake numbers if unavailable!
    """
    url = "https://api.weather.gov/stations/KNYC/observations/latest"
    headers = {"User-Agent": "CITYPULSE-StudentProject/1.0 (university-ml-project)"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        props = resp.json().get("properties", {})

        temp_c = props.get("temperature", {}).get("value")
        wind_kmh = props.get("windSpeed", {}).get("value")
        precip_mm = props.get("precipitationLastHour", {}).get("value")

        return {
            "status": "live",
            "source": "api.weather.gov (KNYC)",
            "timestamp": props.get("timestamp", "unavailable"),
            "temp_f": round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else "unavailable",
            "wind_mph": round(wind_kmh * 0.621371, 1) if wind_kmh is not None else "unavailable",
            "precip_in": round(precip_mm / 25.4, 3) if precip_mm is not None else "unavailable",
        }
    except Exception as ex:
        return {
            "status": "unavailable",
            "source": f"api.weather.gov (error: {type(ex).__name__})",
            "timestamp": "unavailable",
            "temp_f": "unavailable",
            "wind_mph": "unavailable",
            "precip_in": "unavailable",
        }


def run_live_cycle():
    """Run one prediction cycle and save live_snapshot.json."""
    now_iso = datetime.now(timezone.utc).astimezone().isoformat()

    # 1. Fetch live weather (honestly reports "unavailable" if offline/missing)
    live_weather = fetch_live_weather_knyc()
    print(f"\n[A] Weather status: {live_weather['status']} ({live_weather['source']})")
    print(f"    temp_f={live_weather['temp_f']}, wind_mph={live_weather['wind_mph']}, precip_in={live_weather['precip_in']}")

    # 2. Load recent test window in "Simulated Input" mode
    traffic_df   = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet"))
    weather_df   = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "weather_clean.parquet"))
    usual_df     = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "usual_speed_table.parquet"))
    adj_matrix   = np.load(os.path.join(DATA_PROCESSED_DIR, "adj_matrix.npy"))
    order_df     = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "sensor_order.csv"))
    sensor_edge  = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "sensor_to_edge.csv"))
    sensor_order = list(order_df["link_id"])

    # Pick test window matching current NYC hour (so it doesn't stay stuck at midnight 00:00!)
    from zoneinfo import ZoneInfo
    nyc_hour = datetime.now(ZoneInfo(TIMEZONE)).hour
    all_times = np.sort(traffic_df["time"].unique())
    matching = [t for t in all_times if pd.Timestamp(t).tz_convert(TIMEZONE).hour == nyc_hour]
    end_t = matching[-1] if matching else all_times[-1]
    end_idx = int(np.searchsorted(all_times, end_t))
    last_times = all_times[max(0, end_idx - 5) : end_idx + 1]
    recent_tr  = traffic_df[traffic_df["time"].isin(last_times)].copy()
    recent_w   = weather_df[weather_df["time"].isin(last_times)].copy()

    # Use the EXACT SAME feature builder from src/common.py
    feat_df = build_traffic_features(
        recent_tr, recent_w,
        adj_matrix=adj_matrix,
        sensor_order=sensor_order,
        usual_speed_df=usual_df,
    )
    latest_t = last_times[-1]
    cur_df   = feat_df[feat_df["time"] == latest_t].copy()

    # 3. Predict +15, +30, +45, +60 min speed_ratio for all 50 sensors
    horizon_preds = {}
    for mins in [15, 30, 45, 60]:
        model = joblib.load(os.path.join(MODELS_DIR, f"lgb_horizon_{mins}m.joblib"))
        horizon_preds[f"+{mins}m"] = model.predict(cur_df[FEATURE_COLS]).round(4).tolist()

    # Quantiles P10 & P90 for +15m
    q10_model = joblib.load(os.path.join(MODELS_DIR, "lgb_p10_15m.joblib"))
    q90_model = joblib.load(os.path.join(MODELS_DIR, "lgb_p90_15m.joblib"))
    p10_vals  = q10_model.predict(cur_df[FEATURE_COLS]).round(4).tolist()
    p90_vals  = q90_model.predict(cur_df[FEATURE_COLS]).round(4).tolist()

    # 4. City Congestion Index (now + per horizon) & Crisis Warning
    mean_sr    = float(cur_df["speed_ratio"].mean())
    share_cong = float((cur_df["speed_ratio"] < CRISIS_CONGESTED_RATIO).mean())
    cong_index = round(float(np.clip(100.0 * (0.6 * (1.0 - mean_sr) + 0.4 * share_cong), 0, 100)), 1)

    horizon_cong = {}
    for h_key, vals in horizon_preds.items():
        arr = np.array(vals)
        m_sr = float(arr.mean())
        s_cg = float((arr < CRISIS_CONGESTED_RATIO).mean())
        horizon_cong[h_key] = round(float(np.clip(100.0 * (0.6 * (1.0 - m_sr) + 0.4 * s_cg), 0, 100)), 1)

    crisis_clf = joblib.load(os.path.join(MODELS_DIR, "crisis_lgb.joblib"))
    city_row = pd.DataFrame([{
        "mean_speed_ratio": mean_sr,
        "share_congested": share_cong,
        "mean_sr_lag1": float(cur_df["lag_1"].mean()),
        "share_cong_lag1": float((cur_df["lag_1"] < CRISIS_CONGESTED_RATIO).mean()),
        "precip": float(cur_df["precip"].iloc[0]),
        "temp": float(cur_df["temp"].iloc[0]),
        "wind": float(cur_df["wind"].iloc[0]),
        "hour": int(cur_df["hour"].iloc[0]),
        "weekday": int(cur_df["weekday"].iloc[0]),
        "is_weekend": int(cur_df["is_weekend"].iloc[0]),
        "is_holiday": int(cur_df["is_holiday"].iloc[0]),
    }])
    crisis_prob = round(float(crisis_clf.predict_proba(city_row)[0, 1]), 4)

    # 5. PM2.5 +1h prediction (using latest hourly PM2.5)
    pm25_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "pm25_clean.parquet"))
    pm_h = pm25_df.set_index("time")["pm25"].resample("1h").mean().dropna().tail(7)
    aqi_model = joblib.load(os.path.join(MODELS_DIR, "aqi_lgb_1h.joblib"))
    aqi_row = pd.DataFrame([{
        "pm25_lag0": float(pm_h.iloc[-1]),
        "pm25_lag1h": float(pm_h.iloc[-2]),
        "pm25_lag3h": float(pm_h.iloc[-4]),
        "pm25_lag6h": float(pm_h.iloc[-7]),
        "pm25_roll_3h": float(pm_h.tail(3).mean()),
        "pm25_roll_6h": float(pm_h.tail(6).mean()),
        "speed_ratio": mean_sr,
        "wind": float(cur_df["wind"].iloc[0]),
        "precip": float(cur_df["precip"].iloc[0]),
        "temp": float(cur_df["temp"].iloc[0]),
        "hour": int(cur_df["hour"].iloc[0]),
        "weekday": int(cur_df["weekday"].iloc[0]),
    }])
    pm25_pred_1h = round(float(aqi_model.predict(aqi_row)[0]), 2)

    # Merge sensor GPS coords for dashboard map
    cur_with_geo = cur_df[["link_id", "speed_ratio"]].merge(
        sensor_edge[["link_id", "road_name", "sensor_lon", "sensor_lat"]],
        on="link_id", how="left"
    )

    sensors_out = []
    for i, r in cur_with_geo.iterrows():
        sensors_out.append({
            "link_id": r["link_id"],
            "road_name": r["road_name"],
            "lat": float(r["sensor_lat"]),
            "lon": float(r["sensor_lon"]),
            "current_sr": round(float(r["speed_ratio"]), 4),
            "pred_15m": horizon_preds["+15m"][i],
            "pred_30m": horizon_preds["+30m"][i],
            "pred_45m": horizon_preds["+45m"][i],
            "pred_60m": horizon_preds["+60m"][i],
            "p10_15m": p10_vals[i],
            "p90_15m": p90_vals[i],
        })

    snapshot = {
        "source": "Simulated Input (Test Replay) + Live Weather Check",
        "traffic_mode": "Simulated Input",
        "weather": live_weather,
        "generated_at": now_iso,
        "simulated_data_timestamp": str(latest_t),
        "city_metrics": {
            "congestion_index": cong_index,
            "horizon_congestion": horizon_cong,
            "crisis_probability_30m": crisis_prob,
            "crisis_warning": crisis_prob >= 0.5,
            "mean_speed_ratio": round(mean_sr, 4),
            "pm25_pred_1h": pm25_pred_1h,
            "aqi_category": pm25_to_aqi_category(pm25_pred_1h),
        },
        "sensors": sensors_out,
    }

    out_path = os.path.join(DATA_PROCESSED_DIR, "live_snapshot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n[B] Snapshot saved: {out_path}")
    print(f"    Mode             : {snapshot['traffic_mode']}")
    print(f"    Congestion Index : {cong_index} / 100")
    print(f"    Crisis Prob 30m  : {crisis_prob:.2%}")
    print(f"    PM2.5 +1h        : {pm25_pred_1h} µg/m³ ({snapshot['city_metrics']['aqi_category']})")
    return snapshot


if __name__ == "__main__":
    snap = run_live_cycle()
    summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("## Step 22: Live Pipeline\n\n")
        f.write("**What we did:** Built a live collector checking `api.weather.gov` (KNYC), "
                "replaying test traffic in `Simulated Input` mode through `common.py`.\n\n")
        f.write(f"**What we got:** Saved `live_snapshot.json` with source, timestamp, "
                f"and 'unavailable' handling for offline weather.\n\n")
        f.write(f"**Key number:** Congestion Index = **{snap['city_metrics']['congestion_index']}/100**, "
                f"PM2.5 +1h = **{snap['city_metrics']['pm25_pred_1h']} µg/m³**.\n\n---\n\n")
    print("\n" + "=" * 65)
    print("  STEP 22 DONE.")
    print("=" * 65)
