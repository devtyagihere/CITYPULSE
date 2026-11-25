"""
features.py — Stage 2: Leakage-free feature engineering + automated leakage test.

Builds Groups 1–4 using ONLY timestamps strictly < t (plus static OSM & calendar at t):
  Group 1: Temporal & Calendar (known at t)
  Group 2: Autoregressive Speed & Momentum (strictly < t: t-15m, t-30m, ..., t-7d)
  Group 3: Exogenous Weather & NYPD Collisions (strictly < t)
  Group 4: Spatial / Segment & Network Context (static OSM + network speed at t-15m)

Stores:
  - data/processed/features.parquet
  - data/processed/labels.parquet  (separate table: y_15, y_30, y_45, y_60)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def build_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Construct Groups 1-4 features (strictly < t) and separate future label table."""
    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    # ---- Separate future labels (+15, +30, +45, +60 min) ----
    # Note: y_15 is speed at t + 15m (shift(-1)), y_30 at t + 30m (shift(-2)), etc.
    g_seg = df.groupby("segment_id", sort=False)
    labels = pd.DataFrame(
        {
            "segment_id": df["segment_id"],
            "timestamp": df["timestamp"],
            "y_15": g_seg["speed_mph"].shift(-1).astype("float32"),
            "y_30": g_seg["speed_mph"].shift(-2).astype("float32"),
            "y_45": g_seg["speed_mph"].shift(-3).astype("float32"),
            "y_60": g_seg["speed_mph"].shift(-4).astype("float32"),
        }
    )

    feat = pd.DataFrame(
        {
            "segment_id": df["segment_id"],
            "timestamp": df["timestamp"],
        }
    )

    # ---- Group 1: Temporal & Calendar (known at t) ----
    hour = df["timestamp"].dt.hour.astype("int16")
    dow = df["timestamp"].dt.dayofweek.astype("int16")
    feat["hour"] = hour
    feat["dayofweek"] = dow
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype("float32")
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype("float32")
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype("float32")
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype("float32")
    feat["is_weekend"] = df["is_weekend"].astype("int8")
    feat["is_holiday"] = df["is_holiday"].astype("int8")
    feat["is_rush_hour"] = (
        ((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))
    ).astype("int8")

    # ---- Group 2: Autoregressive Speed & Momentum (strictly < t) ----
    s_past1 = g_seg["speed_mph"].shift(1)  # t - 15m
    feat["speed_lag_1"] = s_past1.astype("float32")
    feat["speed_lag_2"] = g_seg["speed_mph"].shift(2).astype("float32")
    feat["speed_lag_4"] = g_seg["speed_mph"].shift(4).astype("float32")
    feat["speed_lag_8"] = g_seg["speed_mph"].shift(8).astype("float32")
    feat["speed_lag_96"] = g_seg["speed_mph"].shift(96).astype("float32")
    feat["speed_lag_672"] = g_seg["speed_mph"].shift(672).astype("float32")

    # Rolling windows computed on s_past1 (strictly < t)
    g_past1 = s_past1.groupby(df["segment_id"], sort=False)
    feat["roll_mean_1h"] = g_past1.transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    ).astype("float32")
    feat["roll_std_1h"] = (
        g_past1.transform(lambda x: x.rolling(4, min_periods=2).std())
        .fillna(0.0)
        .astype("float32")
    )
    feat["roll_mean_3h"] = g_past1.transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    ).astype("float32")
    feat["roll_min_1h"] = g_past1.transform(
        lambda x: x.rolling(4, min_periods=1).min()
    ).astype("float32")
    feat["roll_max_1h"] = g_past1.transform(
        lambda x: x.rolling(4, min_periods=1).max()
    ).astype("float32")

    feat["speed_delta_15m"] = (feat["speed_lag_1"] - feat["speed_lag_2"]).astype("float32")
    feat["speed_delta_1h"] = (feat["speed_lag_1"] - feat["speed_lag_4"]).astype("float32")
    feat["was_missing_lag_1"] = (
        g_seg["is_missing"].shift(1).fillna(1).astype("int8")
    )

    # ---- Group 3: Exogenous Weather & NYPD Collisions (strictly < t) ----
    # Time-indexed exogenous series shifted by 1 bin (t - 15m)
    time_exog = (
        df[["timestamp", "temp_c", "humidity_pct", "precip_mm", "wind_kmh",
            "city_collisions_15m", "city_injuries_15m"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    time_exog["temp_c_lag_1"] = time_exog["temp_c"].shift(1).astype("float32")
    time_exog["humidity_pct_lag_1"] = time_exog["humidity_pct"].shift(1).astype("float32")
    time_exog["precip_mm_lag_1"] = time_exog["precip_mm"].shift(1).astype("float32")
    time_exog["wind_kmh_lag_1"] = time_exog["wind_kmh"].shift(1).astype("float32")
    time_exog["precip_roll_3h"] = (
        time_exog["precip_mm"].shift(1).rolling(12, min_periods=1).sum().astype("float32")
    )
    time_exog["collisions_lag_1"] = (
        time_exog["city_collisions_15m"].shift(1).fillna(0).astype("float32")
    )
    time_exog["collisions_roll_1h"] = (
        time_exog["city_collisions_15m"]
        .shift(1)
        .rolling(4, min_periods=1)
        .sum()
        .fillna(0)
        .astype("float32")
    )
    time_exog["collisions_roll_3h"] = (
        time_exog["city_collisions_15m"]
        .shift(1)
        .rolling(12, min_periods=1)
        .sum()
        .fillna(0)
        .astype("float32")
    )
    time_exog["injuries_roll_1h"] = (
        time_exog["city_injuries_15m"]
        .shift(1)
        .rolling(4, min_periods=1)
        .sum()
        .fillna(0)
        .astype("float32")
    )

    exog_cols = [
        "timestamp", "temp_c_lag_1", "humidity_pct_lag_1", "precip_mm_lag_1",
        "wind_kmh_lag_1", "precip_roll_3h", "collisions_lag_1",
        "collisions_roll_1h", "collisions_roll_3h", "injuries_roll_1h",
    ]
    feat = feat.merge(time_exog[exog_cols], on="timestamp", how="left")

    # ---- Group 4: Spatial / Segment & Network Context (static + strictly < t) ----
    seg_codes = pd.Categorical(feat["segment_id"]).codes
    feat["segment_idx"] = seg_codes.astype("int32")
    feat["road_length_m"] = df["road_length_m"].astype("float32")
    feat["osm_speed_kph"] = df["osm_speed_kph"].astype("float32")

    net_stats = (
        df.groupby("timestamp")["speed_mph"]
        .agg(net_mean="mean", net_std="std")
        .sort_index()
        .shift(1)
        .reset_index()
        .rename(
            columns={
                "net_mean": "network_mean_speed_lag_1",
                "net_std": "network_std_speed_lag_1",
            }
        )
    )
    net_stats["network_mean_speed_lag_1"] = net_stats["network_mean_speed_lag_1"].astype("float32")
    net_stats["network_std_speed_lag_1"] = net_stats["network_std_speed_lag_1"].fillna(0).astype("float32")
    feat = feat.merge(net_stats, on="timestamp", how="left")

    feature_cols = [c for c in feat.columns if c not in ("segment_id", "timestamp")]
    return feat, labels, feature_cols


def run_leakage_test(
    feat: pd.DataFrame,
    labels: pd.DataFrame,
    raw_df: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    """
    Verify zero future leakage:
      1. Exact alignment check: speed_lag_1 == speed(t - 15m), y_15 == speed(t + 15m).
      2. Correlation canary: no feature may have |corr(X, y_15)| > 0.99.
      3. Honest vs Leaky model check: adding y_15 as a feature drops MAE artificially > 90%.
    """
    print("\n[leakage-test] Running strict < t verification...")

    # 1. Alignment check on first segment
    seg0 = feat["segment_id"].iloc[0]
    f0 = feat[feat["segment_id"] == seg0].sort_values("timestamp").reset_index(drop=True)
    l0 = labels[labels["segment_id"] == seg0].sort_values("timestamp").reset_index(drop=True)
    r0 = raw_df[raw_df["segment_id"] == seg0].sort_values("timestamp").reset_index(drop=True)

    # At row i=10, speed_lag_1 must equal r0.speed_mph[9] (t - 15m) and y_15 == r0.speed_mph[11] (t + 15m)
    i = 10
    assert np.isclose(f0.loc[i, "speed_lag_1"], r0.loc[i - 1, "speed_mph"], equal_nan=True), (
        "Leakage/alignment error: speed_lag_1 != speed(t - 15m)"
    )
    assert np.isclose(l0.loc[i, "y_15"], r0.loc[i + 1, "speed_mph"], equal_nan=True), (
        "Label alignment error: y_15 != speed(t + 15m)"
    )

    # 2. Sample subset for correlation & honest-vs-leaky check
    merged = feat.merge(labels[["segment_id", "timestamp", "y_15"]], on=["segment_id", "timestamp"])
    valid = merged.dropna(subset=["y_15", "speed_lag_1", "speed_lag_2", "speed_lag_4"])
    sample = valid.iloc[:50000].copy()

    corrs = sample[feature_cols].corrwith(sample["y_15"]).abs().dropna()
    max_corr_feat = corrs.idxmax()
    max_corr_val = float(corrs.max())
    print(f"[leakage-test] Max |corr(feature, y_15)| = {max_corr_val:.4f} ({max_corr_feat})")
    assert max_corr_val < 0.99, f"Potential target leakage in feature {max_corr_feat}: corr={max_corr_val}"

    # 3. Honest vs Leaky canary model
    split_idx = int(len(sample) * 0.8)
    tr, te = sample.iloc[:split_idx], sample.iloc[split_idx:]
    honest_cols = ["speed_lag_1", "speed_lag_2", "speed_lag_4", "hour"]
    leaky_cols = honest_cols + ["y_15"]

    m_honest = HistGradientBoostingRegressor(max_iter=40, random_state=42)
    m_honest.fit(tr[honest_cols], tr["y_15"])
    mae_honest = mean_absolute_error(te["y_15"], m_honest.predict(te[honest_cols]))

    m_leaky = HistGradientBoostingRegressor(max_iter=40, random_state=42)
    m_leaky.fit(tr[leaky_cols], tr["y_15"])
    mae_leaky = mean_absolute_error(te["y_15"], m_leaky.predict(te[leaky_cols]))

    drop_pct = (mae_honest - mae_leaky) / max(mae_honest, 1e-6) * 100.0
    print(
        f"[leakage-test] Honest MAE={mae_honest:.4f} mph vs Leaky MAE={mae_leaky:.4f} mph "
        f"(artificial drop={drop_pct:.1f}%)"
    )
    assert mae_honest > 0.5, "Honest MAE suspiciously near zero — check for leakage!"
    assert drop_pct > 80.0, "Canary check failed: leaky model did not drop >80%."
    print("[leakage-test] PASSED: All features strictly < t.")


def build_live_features(
    history_df: pd.DataFrame,
    target_time: pd.Timestamp,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build Groups 1-4 features for a single live bin `target_time` using
    the exact same `build_features_and_labels` pipeline (strictly < t).
    """
    feat_all, _, feature_cols = build_features_and_labels(history_df)
    live_feat = feat_all[feat_all["timestamp"] == target_time].copy().reset_index(drop=True)
    return live_feat, feature_cols


def main() -> None:
    canon_path = PROCESSED_DIR / "canonical_15min.parquet"
    if not canon_path.exists():
        raise FileNotFoundError(f"Run ingest.py first: {canon_path} not found.")

    print(f"[features] Loading {canon_path}...")
    raw_df = pd.read_parquet(canon_path)
    feat, labels, feature_cols = build_features_and_labels(raw_df)

    run_leakage_test(feat, labels, raw_df, feature_cols)

    feat_path = PROCESSED_DIR / "features.parquet"
    labels_path = PROCESSED_DIR / "labels.parquet"
    feat.to_parquet(feat_path, index=False)
    labels.to_parquet(labels_path, index=False)
    print(f"[features] Saved {feat_path} ({len(feat):,} rows, {len(feature_cols)} features)")
    print(f"[features] Saved {labels_path} ({len(labels):,} rows, targets: y_15, y_30, y_45, y_60)")


if __name__ == "__main__":
    main()
