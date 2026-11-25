# =============================================================================
# STEP NAME  : step10 — LightGBM Traffic Speed Model
# WHAT       : Trains one LightGBM regression model per horizon (+15, +30, +45,
#              +60 min) using early stopping on the validation set.
# WHY        : LightGBM is fast, accurate on tabular features (lags, hour,
#              weather, neighbour speed), and works well on a normal laptop.
# INPUT      : data/processed/lgb_train.parquet, lgb_val.parquet, lgb_test.parquet
# OUTPUT     : models/lgb_horizon_{15,30,45,60}m.joblib
#              reports/step10_lightgbm_results.csv
#              reports/step10_feature_importance.png
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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED, USE_SYNTHETIC

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 10 — LightGBM Traffic Speed Models")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Load train, val, test tables
# ===========================================================================
print("\n[A] Loading LightGBM train / val / test tables ...")
train_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_val.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_test.parquet"))

# Feature columns (strictly past/current info — NO targets or future data!)
FEATURE_COLS = [
    "lag_0", "lag_1", "lag_2", "lag_4",
    "roll_mean_1h", "roll_std_1h",
    "usual_speed_ratio",
    "neighbour_speed_ratio",
    "hour", "weekday", "is_weekend", "is_holiday",
    "temp", "precip", "wind",
    "speed_was_missing",
]

print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  Rows — train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")

# ===========================================================================
# SECTION B — Train one LightGBM model per horizon (+15, +30, +45, +60m)
# ===========================================================================
horizons = [15, 30, 45, 60]
results  = []
models   = {}

for mins in horizons:
    target_col = f"target_{mins}m"
    print(f"\n[B] Training LightGBM for +{mins} min ahead ...")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample_for_bin=50000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    # Early stopping on validation MAE so we don't overfit
    model.fit(
        train_df[FEATURE_COLS], train_df[target_col],
        eval_set=[(val_df[FEATURE_COLS], val_df[target_col])],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Predict on test set
    preds = model.predict(test_df[FEATURE_COLS])
    mae   = mean_absolute_error(test_df[target_col], preds)
    rmse  = root_mean_squared_error(test_df[target_col], preds)

    print(f"  Best iteration: {model.best_iteration_}  |  Test MAE: {mae:.5f}  |  RMSE: {rmse:.5f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"lgb_horizon_{mins}m.joblib")
    joblib.dump(model, model_path)
    models[mins] = model

    results.append({
        "model"   : "LightGBM",
        "horizon" : f"+{mins}m",
        "mae"     : round(float(mae), 5),
        "rmse"    : round(float(rmse), 5),
    })

results_df = pd.DataFrame(results)
out_csv = os.path.join(REPORTS_DIR, "step10_lightgbm_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"\n  Results saved: {out_csv}")

# ===========================================================================
# SECTION C — Plot feature importance (from +15m model)
# ===========================================================================
print("\n[C] Plotting feature importance (+15m model) ...")
imp = pd.Series(models[15].feature_importances_, index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
imp.plot(kind="barh", color="steelblue", ax=ax)
ax.set_title("LightGBM Feature Importance (+15 min prediction)")
ax.set_xlabel("Split Importance Count")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(REPORTS_DIR, "step10_feature_importance.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {plot_path}")

# ===========================================================================
# SECTION D — Append to step summaries
# ===========================================================================
mae_15 = results_df.loc[results_df["horizon"] == "+15m", "mae"].values[0]
mae_60 = results_df.loc[results_df["horizon"] == "+60m", "mae"].values[0]
top_feat = imp.index[-1]

summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 10: LightGBM Traffic Models\n\n")
    f.write("**What we did:** Trained 4 LightGBM models (+15, +30, +45, +60 min) "
            "using early stopping on validation MAE.\n\n")
    f.write(f"**What we got:** Test MAE = **{mae_15:.4f}** (+15m) up to "
            f"**{mae_60:.4f}** (+60m). Most important feature: `{top_feat}`.\n\n")
    f.write(f"**Key number:** LightGBM +15m MAE = **{mae_15:.4f}**.\n\n")
    f.write("**Plot:** `reports/step10_feature_importance.png`\n\n---\n\n")

print("\n" + "=" * 65)
print("  STEP 10 DONE.")
print(f"  → +15m MAE: {mae_15:.4f}  |  +60m MAE: {mae_60:.4f}")
print("=" * 65)
