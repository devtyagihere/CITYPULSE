# =============================================================================
# STEP NAME  : step18 — Air Quality (PM2.5) Prediction Model
# WHAT       : Predicts PM2.5 at +1, +3, and +6 hours ahead using PM2.5 lags,
#              rolling means, city traffic speed_ratio, and weather features.
# WHY        : Ambulance routes and city alerts should consider air quality
#              spikes (PM2.5). Rain and wind clear PM2.5; heavy traffic raises it.
# INPUT      : data/processed/pm25_clean.parquet
#              data/processed/traffic_clean.parquet
#              data/processed/weather_clean.parquet
# OUTPUT     : models/aqi_lgb_{1,3,6}h.joblib
#              reports/step18_aqi_results.csv
#              reports/step18_aqi_shap.png
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
import shap
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from config import (DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
                    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED, USE_SYNTHETIC)
from common import add_time_features, pm25_to_aqi_category

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 18 — Air Quality (PM2.5) Prediction Model")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Build hourly merged dataset (PM2.5 + city traffic + weather)
# ===========================================================================
print("\n[A] Building hourly PM2.5 + traffic + weather table ...")
pm25_df    = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "pm25_clean.parquet"))
traffic_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet"))
weather_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "weather_clean.parquet"))

# Aggregate to hourly on the hour
pm_h = pm25_df.set_index("time")["pm25"].resample("1h").mean().reset_index()
tr_h = traffic_df.set_index("time")["speed_ratio"].resample("1h").mean().reset_index()
w_h  = weather_df.set_index("time")[["wind", "precip", "temp"]].resample("1h").mean().reset_index()

df = pm_h.merge(tr_h, on="time", how="inner").merge(w_h, on="time", how="inner")
df = df.sort_values("time").reset_index(drop=True)
df = add_time_features(df, "time")

# Past-only PM2.5 lags and rolling means
df["pm25_lag0"]       = df["pm25"]
df["pm25_lag1h"]      = df["pm25"].shift(1)
df["pm25_lag3h"]      = df["pm25"].shift(3)
df["pm25_lag6h"]      = df["pm25"].shift(6)
df["pm25_roll_3h"]    = df["pm25"].rolling(3).mean()
df["pm25_roll_6h"]    = df["pm25"].rolling(6).mean()

# Future targets: +1h, +3h, +6h
for h in [1, 3, 6]:
    df[f"target_{h}h"] = df["pm25"].shift(-h)

df = df.dropna().reset_index(drop=True)
print(f"  Hourly rows ready: {len(df):,}")

# Time split (70 / 10 / 20)
n = len(df)
n_tr = int(n * TRAIN_FRAC)
n_va = int(n * (TRAIN_FRAC + VAL_FRAC))
tr_df = df.iloc[:n_tr]
va_df = df.iloc[n_tr:n_va]
te_df = df.iloc[n_va:]

AQI_FEATURES = [
    "pm25_lag0", "pm25_lag1h", "pm25_lag3h", "pm25_lag6h",
    "pm25_roll_3h", "pm25_roll_6h",
    "speed_ratio", "wind", "precip", "temp",
    "hour", "weekday",
]

# ===========================================================================
# SECTION B — Train LightGBM vs Persistence Baseline (+1h, +3h, +6h)
# ===========================================================================
print("\n[B] Training AQI LightGBM models ...")
results = []
models  = {}

for h in [1, 3, 6]:
    target_col = f"target_{h}h"
    y_true = te_df[target_col].values

    # Persistence baseline: future PM2.5 = current PM2.5
    pers_pred = te_df["pm25_lag0"].values
    pers_mae  = mean_absolute_error(y_true, pers_pred)
    pers_rmse = root_mean_squared_error(y_true, pers_pred)
    results.append({"model": "Persistence", "horizon": f"+{h}h",
                    "mae": round(float(pers_mae), 3), "rmse": round(float(pers_rmse), 3)})

    # LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        random_state=RANDOM_SEED, verbose=-1
    )
    model.fit(
        tr_df[AQI_FEATURES], tr_df[target_col],
        eval_set=[(va_df[AQI_FEATURES], va_df[target_col])],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )
    preds = model.predict(te_df[AQI_FEATURES])
    mae   = mean_absolute_error(y_true, preds)
    rmse  = root_mean_squared_error(y_true, preds)

    joblib.dump(model, os.path.join(MODELS_DIR, f"aqi_lgb_{h}h.joblib"))
    models[h] = model
    results.append({"model": "LightGBM", "horizon": f"+{h}h",
                    "mae": round(float(mae), 3), "rmse": round(float(rmse), 3)})

res_df = pd.DataFrame(results)
print("\n  PM2.5 Prediction Results (µg/m³):")
print(res_df.to_string(index=False))
res_df.to_csv(os.path.join(REPORTS_DIR, "step18_aqi_results.csv"), index=False)

# Demo AQI category conversion from common.py
sample_pred = float(models[1].predict(te_df[AQI_FEATURES].iloc[:1])[0])
print(f"\n  Sample +1h prediction: {sample_pred:.2f} µg/m³ -> AQI Category: '{pm25_to_aqi_category(sample_pred)}'")

# ===========================================================================
# SECTION C — SHAP plot for +1h AQI model
# ===========================================================================
explainer   = shap.TreeExplainer(models[1])
shap_values = explainer.shap_values(te_df[AQI_FEATURES])
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_values, te_df[AQI_FEATURES], show=False)
plt.title("Step 18 — SHAP Summary (+1h PM2.5 Prediction)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "step18_aqi_shap.png"), dpi=120, bbox_inches="tight")
plt.close()

lgb_1h = res_df[(res_df["model"] == "LightGBM") & (res_df["horizon"] == "+1h")]["mae"].values[0]
per_1h = res_df[(res_df["model"] == "Persistence") & (res_df["horizon"] == "+1h")]["mae"].values[0]

summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 18: AQI (PM2.5) Prediction Model\n\n")
    f.write("**What we did:** Trained LightGBM models to forecast PM2.5 at +1h, +3h, and +6h "
            "using PM2.5 lags, traffic speed_ratio, wind, rain, and temperature.\n\n")
    f.write(f"**What we got:** +1h LightGBM MAE = **{lgb_1h:.2f} µg/m³** vs "
            f"Persistence MAE = {per_1h:.2f} µg/m³.\n\n")
    f.write(f"**Key number:** +1h PM2.5 MAE = **{lgb_1h:.2f} µg/m³**.\n\n")
    f.write("**Plot:** `reports/step18_aqi_shap.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 18 DONE.  +1h PM2.5 MAE = {lgb_1h:.2f} µg/m³ (Persistence = {per_1h:.2f})")
print("=" * 65)
