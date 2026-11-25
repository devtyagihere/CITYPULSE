# =============================================================================
# STEP NAME  : step17 — City Crisis Prediction & Anomaly Detection
# WHAT       : Labels historical traffic crises (>30% sensors congested for
#              >=30 min), trains a LightGBM classifier to warn 30 min ahead,
#              fits an Isolation Forest anomaly detector, and computes a
#              0-100 City Congestion Index.
# WHY        : Hospitals and city dispatchers need an early city-wide warning
#              before Manhattan locks up, not just individual road speeds.
# INPUT      : data/processed/traffic_clean.parquet
#              data/processed/weather_clean.parquet
#              data/processed/lgb_train.parquet, lgb_val.parquet, lgb_test.parquet
#              models/lgb_horizon_15m.joblib
# OUTPUT     : models/crisis_lgb.joblib, models/crisis_iforest.joblib
#              reports/step17_crisis_results.csv
#              reports/step17_crisis_and_congestion.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, f1_score)
from config import (DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
                    CRISIS_CONGESTED_RATIO, CRISIS_MIN_SHARE,
                    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED, USE_SYNTHETIC)
from common import add_time_features, add_holiday_feature

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 17 — City Crisis Model & Congestion Index")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Build city-level 15-minute table and crisis labels
# ===========================================================================
print("\n[A] Building city-level 15-minute timeline ...")
traffic_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet"))
weather_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "weather_clean.parquet"))

# Compute per-timestamp city stats
traffic_df["is_congested"] = (traffic_df["speed_ratio"] < CRISIS_CONGESTED_RATIO).astype(float)
city_df = (
    traffic_df.groupby("time")
              .agg(
                  mean_speed_ratio=("speed_ratio", "mean"),
                  share_congested=("is_congested", "mean"),
              )
              .reset_index()
              .sort_values("time")
)

# Merge weather
w_cols = weather_df[["time", "precip", "temp", "wind"]].drop_duplicates("time")
city_df = city_df.merge(w_cols, on="time", how="left")
city_df["precip"] = city_df["precip"].fillna(0.0)
city_df["temp"]   = city_df["temp"].ffill().bfill()
city_df["wind"]   = city_df["wind"].ffill().bfill()

city_df = add_time_features(city_df, "time")
city_df = add_holiday_feature(city_df, "time")

# Crisis definition: >30% of sensors with speed_ratio < 0.5 for >= 30 min (2 steps)
raw_high = (city_df["share_congested"] > CRISIS_MIN_SHARE).astype(int)
city_df["is_crisis_now"] = ((raw_high == 1) & (raw_high.shift(1).fillna(0) == 1)).astype(int)

# Target: will there be a crisis in the next 30 minutes (t+15m or t+30m)?
city_df["crisis_next_30m"] = (
    (city_df["is_crisis_now"].shift(-1) == 1) |
    (city_df["is_crisis_now"].shift(-2) == 1)
).astype(int)

# City Congestion Index (0 = completely free, 100 = gridlock)
city_df["congestion_index"] = (
    100.0 * (0.6 * (1.0 - city_df["mean_speed_ratio"]).clip(0, 1) + 0.4 * city_df["share_congested"])
).clip(0, 100)

# Add predicted city average speed_ratio from our +15m LightGBM model
# (or lag_0 / lag_1 city features — strictly past/current!)
city_df["mean_sr_lag1"] = city_df["mean_speed_ratio"].shift(1)
city_df["share_cong_lag1"] = city_df["share_congested"].shift(1)
city_df = city_df.dropna().reset_index(drop=True)

print(f"  City timeline rows : {len(city_df):,}")
print(f"  Crisis-next-30m    : {city_df['crisis_next_30m'].sum():,} "
      f"({city_df['crisis_next_30m'].mean() * 100:.1f}%)")
print(f"  Congestion index   : min={city_df['congestion_index'].min():.1f}, "
      f"mean={city_df['congestion_index'].mean():.1f}, "
      f"max={city_df['congestion_index'].max():.1f}")

# ===========================================================================
# SECTION B — Time split (70 / 10 / 20)
# ===========================================================================
n = len(city_df)
n_tr = int(n * TRAIN_FRAC)
n_va = int(n * (TRAIN_FRAC + VAL_FRAC))

tr_df = city_df.iloc[:n_tr]
va_df = city_df.iloc[n_tr:n_va]
te_df = city_df.iloc[n_va:].copy()

CITY_FEATURES = [
    "mean_speed_ratio", "share_congested",
    "mean_sr_lag1", "share_cong_lag1",
    "precip", "temp", "wind",
    "hour", "weekday", "is_weekend", "is_holiday",
]

# ===========================================================================
# SECTION C — Train LightGBM Crisis Classifier + Isolation Forest Anomaly Detector
# ===========================================================================
print("\n[C] Training LightGBM crisis classifier ...")
clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    verbose=-1,
)
clf.fit(
    tr_df[CITY_FEATURES], tr_df["crisis_next_30m"],
    eval_set=[(va_df[CITY_FEATURES], va_df["crisis_next_30m"])],
    callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
)
joblib.dump(clf, os.path.join(MODELS_DIR, "crisis_lgb.joblib"))

# Evaluate on TEST
te_probs = clf.predict_proba(te_df[CITY_FEATURES])[:, 1]
te_preds = (te_probs >= 0.5).astype(int)
y_true   = te_df["crisis_next_30m"].values

pr_auc = average_precision_score(y_true, te_probs)
prec   = precision_score(y_true, te_preds, zero_division=0)
rec    = recall_score(y_true, te_preds, zero_division=0)
f1     = f1_score(y_true, te_preds, zero_division=0)

print(f"  Test PR-AUC   : {pr_auc:.4f}")
print(f"  Test Precision: {prec:.4f}")
print(f"  Test Recall   : {rec:.4f}")
print(f"  Test F1       : {f1:.4f}")

# Isolation Forest anomaly detector (trained on TRAIN city features only)
print("\n[D] Training Isolation Forest anomaly detector ...")
iforest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
iforest.fit(tr_df[CITY_FEATURES])
joblib.dump(iforest, os.path.join(MODELS_DIR, "crisis_iforest.joblib"))

te_df["is_anomaly"] = (iforest.predict(te_df[CITY_FEATURES]) == -1).astype(int)
anom_pct = te_df["is_anomaly"].mean() * 100
print(f"  Test anomalies flagged: {te_df['is_anomaly'].sum()} ({anom_pct:.2f}%)")

# Save results CSV
res_df = pd.DataFrame([{
    "pr_auc": round(float(pr_auc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4),
    "anomaly_pct": round(float(anom_pct), 2),
    "mean_congestion_index": round(float(te_df["congestion_index"].mean()), 2),
}])
res_df.to_csv(os.path.join(REPORTS_DIR, "step17_crisis_results.csv"), index=False)

# ===========================================================================
# SECTION E — Plot Congestion Index + Crisis Warnings (first 7 days of test)
# ===========================================================================
week_te = te_df.head(7 * 96)   # 7 days × 96 steps/day
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.suptitle("CITYPULSE — Step 17: Congestion Index & Crisis Warning (Test Week 1)",
             fontsize=13, fontweight="bold")

axes[0].plot(week_te["time"], week_te["congestion_index"], color="darkorange", label="Congestion Index (0-100)")
axes[0].axhline(60, color="red", linestyle="--", alpha=0.6, label="High Congestion (60)")
axes[0].set_ylabel("Index (0-100)")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

te_probs_week = clf.predict_proba(week_te[CITY_FEATURES])[:, 1]
axes[1].plot(week_te["time"], te_probs_week, color="crimson", label="Predicted Crisis Prob (next 30m)")
axes[1].fill_between(week_te["time"], 0, week_te["crisis_next_30m"],
                     color="gray", alpha=0.25, label="Actual Crisis Window")
axes[1].set_ylabel("Probability")
axes[1].set_xlabel("Time")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "step17_crisis_and_congestion.png"), dpi=120, bbox_inches="tight")
plt.close()

# Append summary
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 17: Crisis Model & Congestion Index\n\n")
    f.write("**What we did:** Labeled historical city-wide crises (>30% sensors congested "
            "for >=30 min), trained a LightGBM crisis classifier + Isolation Forest anomaly detector, "
            "and computed a 0–100 Congestion Index.\n\n")
    f.write(f"**What we got:** Test PR-AUC = **{pr_auc:.4f}**, F1 = **{f1:.4f}** "
            f"(Precision={prec:.3f}, Recall={rec:.3f}). Anomalies flagged = {anom_pct:.1f}%.\n\n")
    f.write(f"**Key number:** Crisis classifier PR-AUC = **{pr_auc:.4f}**.\n\n")
    f.write("**Plot:** `reports/step17_crisis_and_congestion.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 17 DONE.  PR-AUC = {pr_auc:.4f}  |  F1 = {f1:.4f}")
print("=" * 65)
