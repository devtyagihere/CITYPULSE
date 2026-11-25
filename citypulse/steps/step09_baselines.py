# =============================================================================
# STEP NAME  : step09 — Baseline Models (Simple Benchmarks)
# WHAT       : Evaluates 3 simple baseline methods on the test set:
#              1) Persistence (future speed = speed right now)
#              2) Historical Average (usual speed for sensor + hour + weekday)
#              3) Seasonal Naive (same speed as 7 days ago at the exact same time)
# WHY        : Before trusting any ML model, we must beat simple guesses.
#              If a complex model cannot beat "just guess current speed", it is useless!
# INPUT      : data/processed/lgb_test.parquet
#              data/processed/traffic_clean.parquet
# OUTPUT     : reports/step09_baselines_results.csv
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from config import DATA_PROCESSED_DIR, REPORTS_DIR, USE_SYNTHETIC

print("=" * 65)
print("  STEP 09 — Baseline Models")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Load test table and clean traffic (for 1-week-ago lookup)
# ===========================================================================
print("\n[A] Loading test data ...")
test_df    = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_test.parquet"))
traffic_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet"))

# Build Seasonal Naive lookup: speed_ratio exactly 7 days earlier
# 7 days * 24 hours * 4 steps/hour = 672 steps of 15 minutes
traffic_df = traffic_df.sort_values(["link_id", "time"]).copy()
traffic_df["seasonal_naive_pred"] = traffic_df.groupby("link_id")["speed_ratio"].shift(672)

test_df = test_df.merge(
    traffic_df[["link_id", "time", "seasonal_naive_pred"]],
    on=["link_id", "time"],
    how="left"
)
# Fallback to usual_speed_ratio if 1-week-ago value was missing
test_df["seasonal_naive_pred"] = test_df["seasonal_naive_pred"].fillna(test_df["usual_speed_ratio"])

print(f"  Test rows: {len(test_df):,}")

# ===========================================================================
# SECTION B — Evaluate all 3 baselines for each horizon (+15, +30, +45, +60 min)
# ===========================================================================
print("\n[B] Evaluating baselines on test set ...")

horizons = [15, 30, 45, 60]
results  = []

for mins in horizons:
    y_true = test_df[f"target_{mins}m"].values

    # 1. Persistence: predict future = current speed_ratio (lag_0)
    pred_pers = test_df["lag_0"].values

    # 2. Historical Average: predict future = usual speed for (sensor, hour, weekday)
    pred_hist = test_df["usual_speed_ratio"].values

    # 3. Seasonal Naive: predict future = same time 1 week ago
    pred_seas = test_df["seasonal_naive_pred"].values

    for name, pred in [
        ("Persistence",        pred_pers),
        ("Historical_Average", pred_hist),
        ("Seasonal_Naive",     pred_seas),
    ]:
        mae  = mean_absolute_error(y_true, pred)
        rmse = root_mean_squared_error(y_true, pred)
        results.append({
            "model"   : name,
            "horizon" : f"+{mins}m",
            "mae"     : round(float(mae), 5),
            "rmse"    : round(float(rmse), 5),
        })

results_df = pd.DataFrame(results)
print("\n  Baseline Results (speed_ratio error — lower is better):")
print(results_df.to_string(index=False))

# Save results table
out_csv = os.path.join(REPORTS_DIR, "step09_baselines_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}")

# ===========================================================================
# SECTION C — Append to step summaries
# ===========================================================================
best_15 = results_df[results_df["horizon"] == "+15m"].sort_values("mae").iloc[0]
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 09: Baseline Models\n\n")
    f.write("**What we did:** Tested 3 simple baselines (Persistence, Historical Average, "
            "Seasonal Naive) on the test set across +15, +30, +45, and +60 min.\n\n")
    f.write("**What we got:** Saved MAE and RMSE benchmark table to "
            "`reports/step09_baselines_results.csv`.\n\n")
    f.write(f"**Key number:** Best baseline at +15m is **{best_15['model']}** "
            f"with MAE = **{best_15['mae']:.4f}** (target for ML models to beat!).\n\n---\n\n")

print("\n" + "=" * 65)
print("  STEP 09 DONE.")
print(f"  → Best +15m baseline: {best_15['model']} (MAE = {best_15['mae']:.4f})")
print("=" * 65)
