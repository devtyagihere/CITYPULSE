"""
train_model.py — Stage 3b: Train LightGBM (+15/+30/+45/+60), P90 quantile model,
and Isotonic Regression congestion probability calibrator.

Exports:
  models/speed_model_v1.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from baselines import HORIZONS, get_time_splits

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

CONGESTION_RATIO = 0.60  # speed < 60% of segment's train 85th-pct free-flow speed


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    df = feat.merge(labels, on=["segment_id", "timestamp"])

    feature_cols = [c for c in feat.columns if c not in ("segment_id", "timestamp")]
    train_df, val_df, test_df, split_info = get_time_splits(df)

    # Free-flow speed per segment computed STRICTLY on TRAIN split
    free_flow_map = (
        train_df.groupby("segment_id")["speed_lag_1"]
        .quantile(0.85)
        .clip(lower=10.0)
        .to_dict()
    )

    lgb_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample_for_bin=50000,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    models = {}
    for horizon in HORIZONS:
        tr = train_df[train_df[horizon].notna()]
        va = val_df[val_df[horizon].notna()]
        print(f"[train] Fitting LightGBM for {horizon} (train={len(tr):,}, val={len(va):,})...")
        model = lgb.LGBMRegressor(objective="regression_l1", **lgb_params)
        model.fit(
            tr[feature_cols],
            tr[horizon],
            eval_set=[(va[feature_cols], va[horizon])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        models[horizon] = model
        print(f"[train]   best_iteration={model.best_iteration_}")

    # Optional P90 quantile model (+15m upper bound)
    tr15 = train_df[train_df["y_15"].notna()]
    va15 = val_df[val_df["y_15"].notna()]
    print("[train] Fitting LightGBM P90 quantile model (y_15, alpha=0.90)...")
    p90_model = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **lgb_params)
    p90_model.fit(
        tr15[feature_cols],
        tr15["y_15"],
        eval_set=[(va15[feature_cols], va15["y_15"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Isotonic Regression calibrator on VAL split for congestion probability at t+15m
    print("[train] Fitting Isotonic Regression calibrator on validation split...")
    val_ff = va15["segment_id"].map(free_flow_map).fillna(25.0).to_numpy()
    val_pred_15 = models["y_15"].predict(va15[feature_cols])
    val_raw_score = np.clip(1.0 - (val_pred_15 / val_ff), 0.0, 1.0)
    val_congested = (va15["y_15"].to_numpy() < CONGESTION_RATIO * val_ff).astype(int)

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(val_raw_score, val_congested)
    print(f"[train]   val congestion base rate = {val_congested.mean():.3%}")

    bundle = {
        "models": models,
        "p90_model": p90_model,
        "calibrator": calibrator,
        "free_flow_map": free_flow_map,
        "congestion_ratio": CONGESTION_RATIO,
        "feature_cols": feature_cols,
        "split_info": split_info,
    }

    out_path = MODELS_DIR / "speed_model_v1.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[train] Exported model bundle -> {out_path}")


if __name__ == "__main__":
    main()
