# =============================================================================
# STEP NAME  : step16 — Explain Predictions with SHAP
# WHAT       : Uses SHAP (TreeExplainer) on the +15 min LightGBM model to
#              generate a summary plot and print the top 5 reasons in easy words.
# WHY        : Teachers and city operators want to know WHY the model predicts
#              a jam — not just a black-box number.
# INPUT      : models/lgb_horizon_15m.joblib
#              data/processed/lgb_test.parquet
# OUTPUT     : reports/step16_shap_summary.png
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
import shap
from config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED, USE_SYNTHETIC

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 16 — SHAP Model Explanation")
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

FRIENDLY_NAMES = {
    "lag_0"                 : "Current speed right now",
    "lag_1"                 : "Speed 15 minutes ago",
    "lag_2"                 : "Speed 30 minutes ago",
    "lag_4"                 : "Speed 60 minutes ago",
    "roll_mean_1h"          : "Average speed over last 1 hour",
    "roll_std_1h"           : "How bumpy speed was over last 1 hour",
    "usual_speed_ratio"     : "Usual speed for this road, hour, and weekday",
    "neighbour_speed_ratio" : "Speed of nearby connected roads right now",
    "hour"                  : "Hour of the day (rush hour vs night)",
    "weekday"               : "Day of the week",
    "is_weekend"            : "Whether it is a weekend",
    "is_holiday"            : "Whether it is a public holiday",
    "temp"                  : "Air temperature",
    "precip"                : "Rain / precipitation",
    "wind"                  : "Wind speed",
    "speed_was_missing"     : "Whether sensor reading was missing",
}

model   = joblib.load(os.path.join(MODELS_DIR, "lgb_horizon_15m.joblib"))
test_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_test.parquet"))

# Sample 2,000 test rows so SHAP runs in a few seconds
sample_X = test_df[FEATURE_COLS].sample(n=min(2000, len(test_df)), random_state=RANDOM_SEED)

print("\n[A] Computing SHAP values (2,000 test samples) ...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_X)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
rank_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "mean_shap" : mean_abs_shap,
}).sort_values("mean_shap", ascending=False)

print("\n[B] Top 5 reasons behind LightGBM's +15 min traffic prediction:")
for i, row in enumerate(rank_df.head(5).itertuples(), start=1):
    desc = FRIENDLY_NAMES.get(row.feature, row.feature)
    print(f"  {i}. {row.feature:<22} (impact = {row.mean_shap:.4f}) -> {desc}")

# Plot SHAP summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample_X, show=False)
plt.title("Step 16 — SHAP Summary (+15 min Traffic Speed Prediction)")
plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "step16_shap_summary.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: {plot_path}")

top1 = rank_df.iloc[0]
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 16: SHAP Explanation\n\n")
    f.write("**What we did:** Computed TreeSHAP values on 2,000 test samples for "
            "the +15 min LightGBM model.\n\n")
    f.write("**What we got:** Top drivers are "
            + ", ".join(f"`{r}`" for r in rank_df["feature"].head(5)) + ".\n\n")
    f.write(f"**Key number:** #1 driver is **`{top1['feature']}`** "
            f"(mean |SHAP| = **{top1['mean_shap']:.4f}**).\n\n")
    f.write("**Plot:** `reports/step16_shap_summary.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 16 DONE.  #1 SHAP driver: {top1['feature']} ({top1['mean_shap']:.4f})")
print("=" * 65)
