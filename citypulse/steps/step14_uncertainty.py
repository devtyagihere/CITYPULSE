# =============================================================================
# STEP NAME  : step14 — Uncertainty Estimation (Quantile LightGBM)
# WHAT       : Trains Quantile LightGBM models for P10 (10th percentile),
#              P50 (median), and P90 (90th percentile) at +15 min ahead.
# WHY        : Ambulances need to know not just the average predicted speed,
#              but also the worst-case risk (P10 slow speed / P90 long time).
#              We verify that ~80% of true values fall inside [P10, P90].
# INPUT      : data/processed/lgb_train.parquet, lgb_val.parquet, lgb_test.parquet
# OUTPUT     : models/lgb_q10_15m.joblib, lgb_q50_15m.joblib, lgb_q90_15m.joblib
#              data/processed/test_quantiles_15m.parquet
#              reports/step14_uncertainty_band.png
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
from config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED, USE_SYNTHETIC

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 14 — Quantile LightGBM (P10, P50, P90 Uncertainty)")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

train_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_val.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_test.parquet"))

FEATURE_COLS = [
    "lag_0", "lag_1", "lag_2", "lag_4",
    "roll_mean_1h", "roll_std_1h",
    "usual_speed_ratio",
    "neighbour_speed_ratio",
    "hour", "weekday", "is_weekend", "is_holiday",
    "temp", "precip", "wind",
    "speed_was_missing",
]

preds = {}
for alpha, label in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
    print(f"\n[A] Training Quantile LightGBM ({label.upper()}, alpha={alpha}) ...")
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        train_df[FEATURE_COLS], train_df["target_15m"],
        eval_set=[(val_df[FEATURE_COLS], val_df["target_15m"])],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )
    preds[label] = model.predict(test_df[FEATURE_COLS])
    joblib.dump(model, os.path.join(MODELS_DIR, f"lgb_{label}_15m.joblib"))

# Ensure monotonic order P10 <= P50 <= P90
p10 = np.minimum(preds["p10"], preds["p50"])
p50 = preds["p50"]
p90 = np.maximum(preds["p90"], preds["p50"])
y_true = test_df["target_15m"].values

# Check coverage: how often is y_true inside [p10, p90]? Should be ~80%!
inside = (y_true >= p10) & (y_true <= p90)
coverage_pct = inside.mean() * 100
avg_width    = np.mean(p90 - p10)

print(f"\n[B] Calibration check on TEST set:")
print(f"  80% interval [P10, P90] actual coverage : {coverage_pct:.2f}%  (target ~80%)")
print(f"  Average interval width (P90 - P10)      : {avg_width:.4f}")

# Save test quantile predictions for Step 15
out_df = test_df[["link_id", "time", "target_15m"]].copy()
out_df["p10"] = p10
out_df["p50"] = p50
out_df["p90"] = p90
out_df.to_parquet(os.path.join(DATA_PROCESSED_DIR, "test_quantiles_15m.parquet"), index=False)

# Plot first 80 steps for one sensor
one = out_df[out_df["link_id"] == sorted(out_df["link_id"].unique())[0]].head(80)
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(one))
ax.fill_between(x, one["p10"], one["p90"], color="skyblue", alpha=0.35, label="P10–P90 band (~80%)")
ax.plot(x, one["p50"], color="steelblue", linewidth=1.8, label="P50 prediction")
ax.plot(x, one["target_15m"], color="black", linewidth=1.2, linestyle="--", label="True speed_ratio")
ax.set_title(f"Step 14 — Uncertainty Band on Test Set (Coverage = {coverage_pct:.1f}%)")
ax.set_xlabel("Time steps (15 min)")
ax.set_ylabel("speed_ratio")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "step14_uncertainty_band.png"), dpi=120, bbox_inches="tight")
plt.close()

# Append summary
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 14: Uncertainty Estimation (Quantile LightGBM)\n\n")
    f.write("**What we did:** Trained P10, P50, and P90 Quantile LightGBM models "
            "for +15 min predictions.\n\n")
    f.write(f"**What we got:** **{coverage_pct:.1f}%** of true test values fell "
            f"inside [P10, P90] (expected ~80%). Average band width = {avg_width:.4f}.\n\n")
    f.write(f"**Key number:** Empirical 80% interval coverage = **{coverage_pct:.1f}%**.\n\n")
    f.write("**Plot:** `reports/step14_uncertainty_band.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 14 DONE.  [P10, P90] coverage = {coverage_pct:.2f}% (target ~80%)")
print("=" * 65)
