"""
baselines.py — Stage 3a: Strict chronological split & 3 baseline models.

Baselines evaluated on val and test across +15, +30, +45, +60 min horizons:
  1. Persistence      : speed_lag_1 (last observed speed strictly < t)
  2. Seasonal Naive   : speed_lag_672 (same time 7 days ago, fallback to 24h / train mean)
  3. Historical Avg   : mean speed per (segment_id, hour, dayofweek) on TRAIN only
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

HORIZONS = ["y_15", "y_30", "y_45", "y_60"]


def get_time_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Strict chronological split (earliest -> train, next -> val, last -> test)."""
    unique_times = np.sort(df["timestamp"].unique())
    n_t = len(unique_times)
    t_train_end = unique_times[int(n_t * train_frac) - 1]
    t_val_end = unique_times[int(n_t * (train_frac + val_frac)) - 1]

    train_df = df[df["timestamp"] <= t_train_end].copy()
    val_df = df[(df["timestamp"] > t_train_end) & (df["timestamp"] <= t_val_end)].copy()
    test_df = df[df["timestamp"] > t_val_end].copy()

    split_info = {
        "train_start": str(unique_times[0]),
        "train_end": str(t_train_end),
        "val_start": str(val_df["timestamp"].min()),
        "val_end": str(t_val_end),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(unique_times[-1]),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    return train_df, val_df, test_df, split_info


def compute_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Persistence, Seasonal Naive, and Historical Average on val and test."""
    global_mean = float(train_df["y_15"].mean())
    seg_mean = train_df.groupby("segment_id")["y_15"].mean().to_dict()

    # Historical Average table: (segment_id, hour, dayofweek) -> mean speed on TRAIN ONLY
    hist_table = (
        train_df.groupby(["segment_id", "hour", "dayofweek"])["y_15"]
        .mean()
        .reset_index()
        .rename(columns={"y_15": "pred_hist_avg"})
    )

    rows = []
    for split_name, sdf in [("val", val_df), ("test", test_df)]:
        s = sdf.merge(hist_table, on=["segment_id", "hour", "dayofweek"], how="left")
        s["pred_hist_avg"] = (
            s["pred_hist_avg"]
            .fillna(s["segment_id"].map(seg_mean))
            .fillna(global_mean)
        )
        s["pred_persistence"] = (
            s["speed_lag_1"]
            .fillna(s["segment_id"].map(seg_mean))
            .fillna(global_mean)
        )
        s["pred_seasonal_naive"] = (
            s["speed_lag_672"]
            .fillna(s["speed_lag_96"])
            .fillna(s["segment_id"].map(seg_mean))
            .fillna(global_mean)
        )

        for horizon in HORIZONS:
            valid = s[s[horizon].notna()]
            y_true = valid[horizon].to_numpy()
            for b_name, col in [
                ("Persistence", "pred_persistence"),
                ("Seasonal Naive", "pred_seasonal_naive"),
                ("Historical Average", "pred_hist_avg"),
            ]:
                y_pred = valid[col].to_numpy()
                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                rows.append(
                    {
                        "split": split_name,
                        "model": b_name,
                        "horizon": horizon,
                        "mae": mae,
                        "rmse": rmse,
                        "n_samples": len(y_true),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    df = feat.merge(labels, on=["segment_id", "timestamp"])

    train_df, val_df, test_df, split_info = get_time_splits(df)
    print(f"[baselines] Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    print(f"[baselines] Train: {split_info['train_start']} -> {split_info['train_end']}")
    print(f"[baselines] Val  : {split_info['val_start']} -> {split_info['val_end']}")
    print(f"[baselines] Test : {split_info['test_start']} -> {split_info['test_end']}")

    res = compute_baselines(train_df, val_df, test_df)
    out_path = OUTPUTS_DIR / "baseline_metrics.csv"
    res.to_csv(out_path, index=False)
    print(f"\n[baselines] Saved {out_path}")
    print(res[res["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
