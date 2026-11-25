"""
predictor.py — Stage 3: Live speed & calibrated congestion probability predictor.

Outputs per segment at +15, +30, +45, +60 min:
  - pred_speed_15/30/45/60, speed_ratio_15/30/45/60
  - p90_speed_15
  - calibrated congestion_prob_15/30/45/60
  - staleness & Historical Average fallback flags
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "speed_model_v1.pkl"
HIST_AVG_PKL_PATH = PROCESSED_DIR / "hist_avg_table.pkl.gz"
HIST_AVG_PATH = PROCESSED_DIR / "hist_avg_table.parquet"

HORIZONS = ["15", "30", "45", "60"]


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_or_build_hist_avg() -> pd.DataFrame:
    """Load or build train-only Historical Average lookup (segment_id, hour, dayofweek)."""
    if HIST_AVG_PKL_PATH.exists():
        return pd.read_pickle(HIST_AVG_PKL_PATH, compression="gzip")
    if HIST_AVG_PATH.exists():
        return pd.read_parquet(HIST_AVG_PATH)

    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet", columns=["segment_id", "timestamp", "hour", "dayofweek"])
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet", columns=["segment_id", "timestamp", "y_15"])
    df = feat.merge(labels, on=["segment_id", "timestamp"])
    unique_times = np.sort(df["timestamp"].unique())
    t_train_end = unique_times[int(len(unique_times) * 0.70) - 1]
    tr = df[df["timestamp"] <= t_train_end]
    hist = (
        tr.groupby(["segment_id", "hour", "dayofweek"])["y_15"]
        .mean()
        .reset_index()
        .rename(columns={"y_15": "hist_avg_mph"})
    )
    hist.to_parquet(HIST_AVG_PATH, index=False)
    return hist


def predict_live(
    live_feat: pd.DataFrame,
    feeds: Dict[str, dict],
) -> pd.DataFrame:
    """Run LightGBM + P90 + Isotonic calibrator with Historical Average fallback."""
    bundle = _load_model_bundle()

    models = bundle["models"]
    p90_model = bundle["p90_model"]
    calibrator = bundle["calibrator"]
    free_flow_map = bundle["free_flow_map"]
    feature_cols = bundle["feature_cols"]

    hist_table = _load_or_build_hist_avg()
    out = live_feat[["segment_id", "timestamp", "hour", "dayofweek", "speed_lag_1", "was_missing_lag_1"]].copy()
    out = out.merge(hist_table, on=["segment_id", "hour", "dayofweek"], how="left")

    ff = out["segment_id"].map(free_flow_map).fillna(25.0).to_numpy(dtype=float)
    out["free_flow_mph"] = ff
    out["hist_avg_mph"] = out["hist_avg_mph"].fillna(pd.Series(ff, index=out.index) * 0.75)

    feed_stale = bool(feeds.get("traffic", {}).get("is_stale", False))
    row_missing = (out["was_missing_lag_1"].fillna(1) == 1) | out["speed_lag_1"].isna()
    fallback_mask = (feed_stale | row_missing).to_numpy()

    X = live_feat[feature_cols]
    for h in HORIZONS:
        pred = models[f"y_{h}"].predict(X)
        # Fallback to Historical Average where feed/segment is stale or missing
        pred_final = np.where(fallback_mask, out["hist_avg_mph"].to_numpy(), pred)
        pred_final = np.clip(pred_final, 1.0, 120.0)
        ratio = np.clip(pred_final / ff, 0.05, 1.50)
        raw_score = np.clip(1.0 - ratio, 0.0, 1.0)
        cal_prob = calibrator.predict(raw_score)

        out[f"pred_speed_{h}"] = np.round(pred_final, 2)
        out[f"speed_ratio_{h}"] = np.round(ratio, 4)
        out[f"congestion_prob_{h}"] = np.round(cal_prob, 4)

    p90 = p90_model.predict(X)
    out["p90_speed_15"] = np.round(np.maximum(p90, out["pred_speed_15"]), 2)
    out["is_stale"] = fallback_mask
    out["fallback_used"] = fallback_mask
    out["source"] = feeds.get("traffic", {}).get("source", "unknown")
    out["fetched_at"] = feeds.get("traffic", {}).get("fetched_at", "")
    return out
