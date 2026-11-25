# =============================================================================
# STEP NAME  : step15 — Probability Calibration (Isotonic Regression)
# WHAT       : Uses Isotonic Regression to convert predicted speed_ratio (P50)
#              into a calibrated probability of congestion (speed_ratio < 0.5).
# WHY        : A raw speed prediction like 0.55 doesn't directly say "what is
#              the chance of a traffic jam?". Calibration turns it into an
#              honest probability (e.g., 0.70 means 70% of the time it jams).
# INPUT      : data/processed/lgb_val.parquet
#              data/processed/test_quantiles_15m.parquet
#              models/lgb_p50_15m.joblib
# OUTPUT     : models/isotonic_congestion.joblib
#              reports/step15_calibration_curve.png
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
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from config import (DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
                    CRISIS_CONGESTED_RATIO, USE_SYNTHETIC)

print("=" * 65)
print("  STEP 15 — Congestion Probability Calibration (Isotonic)")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
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

# Load validation set + P50 model to fit Isotonic Regression (NO test leakage!)
val_df    = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_val.parquet"))
p50_model = joblib.load(os.path.join(MODELS_DIR, "lgb_p50_15m.joblib"))

val_pred_sr = p50_model.predict(val_df[FEATURE_COLS])
val_is_cong = (val_df["target_15m"].values < CRISIS_CONGESTED_RATIO).astype(int)

# Fit Isotonic Regression: lower predicted speed_ratio -> higher congestion probability
iso = IsotonicRegression(out_of_bounds="clip", increasing=False)
iso.fit(val_pred_sr, val_is_cong)
joblib.dump(iso, os.path.join(MODELS_DIR, "isotonic_congestion.joblib"))

# Evaluate on TEST set
test_q = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "test_quantiles_15m.parquet"))
test_is_cong = (test_q["target_15m"].values < CRISIS_CONGESTED_RATIO).astype(int)
test_prob    = iso.predict(test_q["p50"].values)

brier = brier_score_loss(test_is_cong, test_prob)
print(f"\n  Test congestion rate : {test_is_cong.mean() * 100:.2f}%")
print(f"  Mean predicted prob  : {test_prob.mean() * 100:.2f}%")
print(f"  Test Brier score     : {brier:.5f}  (0 = perfect, 0.25 = random 50/50)")

# Reliability plot
prob_true, prob_pred = calibration_curve(test_is_cong, test_prob, n_bins=10, strategy="uniform")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.plot(prob_pred, prob_true, "o-", color="steelblue", linewidth=2,
        label=f"Isotonic (Brier = {brier:.4f})")
ax.set_xlabel("Predicted Probability of Congestion")
ax.set_ylabel("Observed Fraction Congested")
ax.set_title("Step 15 — Reliability (Calibration) Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "step15_calibration_curve.png"), dpi=120, bbox_inches="tight")
plt.close()

# Append summary
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 15: Congestion Probability Calibration\n\n")
    f.write("**What we did:** Fitted Isotonic Regression on validation predictions "
            "to map predicted speed_ratio into probability of congestion (< 0.5).\n\n")
    f.write(f"**What we got:** Test Brier score = **{brier:.4f}**; predicted mean "
            f"probability ({test_prob.mean()*100:.1f}%) matches observed rate ({test_is_cong.mean()*100:.1f}%).\n\n")
    f.write(f"**Key number:** Brier score = **{brier:.4f}**.\n\n")
    f.write("**Plot:** `reports/step15_calibration_curve.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 15 DONE.  Brier score = {brier:.5f}")
print("=" * 65)
