"""
monitor.py — Stage 6a: Prediction logging and delayed error tracking.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = Path("/tmp/outputs") if os.environ.get("VERCEL") else ROOT_DIR / "outputs"
PRED_LOG_PATH = OUTPUTS_DIR / "prediction_log.jsonl"
MONITOR_METRICS_PATH = OUTPUTS_DIR / "monitor_metrics.json"


def log_and_evaluate(
    preds_df: pd.DataFrame,
    history_df: pd.DataFrame,
    target_time: pd.Timestamp,
) -> Dict[str, object]:
    """Append current +15m predictions and evaluate any matured prior predictions."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_target_ts = (target_time + pd.Timedelta(minutes=15)).isoformat()
    issued_ts = target_time.isoformat()

    # Append compact summary records for this bin
    with open(PRED_LOG_PATH, "a", encoding="utf-8") as f:
        for r in preds_df.itertuples(index=False):
            rec = {
                "issued_at": issued_ts,
                "target_time_15": eval_target_ts,
                "segment_id": int(r.segment_id),
                "pred_speed_15": float(r.pred_speed_15),
                "congestion_prob_15": float(r.congestion_prob_15),
                "is_stale": bool(r.is_stale),
            }
            f.write(json.dumps(rec) + "\n")

    # Delayed error tracking against observed bins in history_df
    obs = history_df[["segment_id", "timestamp", "speed_mph"]].dropna().copy()
    obs["ts_iso"] = obs["timestamp"].dt.floor("15min").astype(str)

    # Read recent log entries (up to last 5000 lines)
    lines = PRED_LOG_PATH.read_text(encoding="utf-8").splitlines()[-5000:]
    log_df = pd.DataFrame([json.loads(ln) for ln in lines])

    obs_key = obs.assign(target_time_15=obs["timestamp"].apply(lambda x: x.isoformat()))
    matched = log_df.merge(
        obs_key[["segment_id", "target_time_15", "speed_mph"]],
        on=["segment_id", "target_time_15"],
        how="inner",
    )

    if len(matched) > 0:
        err = np.abs(matched["speed_mph"] - matched["pred_speed_15"])
        delayed_mae = float(err.mean())
        matched_count = int(len(matched))
    else:
        delayed_mae = None
        matched_count = 0

    metrics = {
        "total_logged_predictions": int(len(log_df)),
        "matched_delayed_observations": matched_count,
        "delayed_mae_15m_mph": round(delayed_mae, 4) if delayed_mae is not None else None,
        "stale_predictions_in_batch": int(preds_df["is_stale"].sum()),
        "last_updated": issued_ts,
    }
    MONITOR_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
