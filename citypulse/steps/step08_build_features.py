# =============================================================================
# STEP NAME  : step08 — Build Features (Flat Table + Deep Learning Arrays)
# WHAT       : Builds two datasets from cleaned traffic, weather, and graph:
#              (a) Flat table for LightGBM (lags, rolling stats, usual speed,
#                  hour, weekday, holiday, rain, temp, neighbour speed).
#              (b) 4D/3D numpy arrays for GRU and Graph+GRU models.
# WHY        : Tree models (LightGBM) need flat rows, while sequence/graph
#              models (PyTorch GRU/GCN) need (samples, time_steps, sensors, features).
#              Splitting strictly by TIME (70/10/20) prevents data leakage.
# INPUT      : data/processed/traffic_clean.parquet
#              data/processed/weather_clean.parquet
#              data/processed/adj_matrix.npy
#              data/processed/sensor_order.csv
# OUTPUT     : data/processed/lgb_train.parquet, lgb_val.parquet, lgb_test.parquet
#              data/processed/usual_speed_table.parquet
#              data/processed/deep_train.npz, deep_val.npz, deep_test.npz
#              data/processed/deep_scaler.joblib
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from config import (DATA_PROCESSED_DIR, REPORTS_DIR, TRAIN_FRAC, VAL_FRAC,
                    RANDOM_SEED, USE_SYNTHETIC)
from common import add_time_features, build_traffic_features

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 08 — Build Features (LightGBM + Deep Learning)")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

# ===========================================================================
# SECTION A — Load cleaned data & adjacency matrix
# ===========================================================================
print("\n[A] Loading cleaned data and adjacency matrix ...")

traffic_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet"))
weather_df = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "weather_clean.parquet"))
adj_matrix = np.load(os.path.join(DATA_PROCESSED_DIR, "adj_matrix.npy"))
order_df   = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "sensor_order.csv"))
sensor_order = list(order_df["link_id"])

print(f"  Traffic rows : {len(traffic_df):,}  |  Sensors: {len(sensor_order)}")
print(f"  Adjacency    : {adj_matrix.shape}")

# ===========================================================================
# SECTION B — Compute train/val/test time cutoffs (70% / 10% / 20%)
# ===========================================================================
print("\n[B] Computing time-based split cutoffs (70% train / 10% val / 20% test) ...")

unique_times = np.sort(traffic_df["time"].unique())
n_t = len(unique_times)
t_train_end = unique_times[int(n_t * TRAIN_FRAC) - 1]
t_val_end   = unique_times[int(n_t * (TRAIN_FRAC + VAL_FRAC)) - 1]

print(f"  Total time steps : {n_t:,}")
print(f"  Train ends       : {t_train_end}")
print(f"  Val ends         : {t_val_end}")
print(f"  Test ends        : {unique_times[-1]}")

# ===========================================================================
# SECTION C — Compute usual speed per (sensor, hour, weekday) on TRAIN ONLY
# ===========================================================================
print("\n[C] Computing usual speed table on TRAIN period only ...")

train_only = add_time_features(traffic_df[traffic_df["time"] <= t_train_end], "time")
usual_speed_df = (
    train_only.groupby(["link_id", "hour", "weekday"])["speed_ratio"]
              .mean()
              .reset_index()
              .rename(columns={"speed_ratio": "usual_speed_ratio"})
)
usual_path = os.path.join(DATA_PROCESSED_DIR, "usual_speed_table.parquet")
usual_speed_df.to_parquet(usual_path, index=False)
print(f"  Saved usual speed table: {len(usual_speed_df):,} rows")

# ===========================================================================
# SECTION D — (a) Build flat table for LightGBM
# ===========================================================================
print("\n[D] Building flat feature table for LightGBM ...")

flat_df = build_traffic_features(
    traffic_df,
    weather_df,
    adj_matrix=adj_matrix,
    sensor_order=sensor_order,
    usual_speed_df=usual_speed_df,
)

# Add 4 future targets: +15m (shift -1), +30m (-2), +45m (-3), +60m (-4)
flat_df = flat_df.sort_values(["link_id", "time"]).copy()
for step_ahead, mins in [(1, 15), (2, 30), (3, 45), (4, 60)]:
    flat_df[f"target_{mins}m"] = flat_df.groupby("link_id")["speed_ratio"].shift(-step_ahead)

# Drop rows where lags or targets are NaN (only first 4 & last 4 steps per sensor)
req_cols = ["lag_0", "lag_1", "lag_2", "lag_4",
            "target_15m", "target_30m", "target_45m", "target_60m"]
flat_df = flat_df.dropna(subset=req_cols).copy()

# Split by time
lgb_train = flat_df[flat_df["time"] <= t_train_end]
lgb_val   = flat_df[(flat_df["time"] > t_train_end) & (flat_df["time"] <= t_val_end)]
lgb_test  = flat_df[flat_df["time"] > t_val_end]

lgb_train.to_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_train.parquet"), index=False)
lgb_val.to_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_val.parquet"), index=False)
lgb_test.to_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_test.parquet"), index=False)

print(f"  LightGBM train : {len(lgb_train):,} rows")
print(f"  LightGBM val   : {len(lgb_val):,} rows")
print(f"  LightGBM test  : {len(lgb_test):,} rows")

# ===========================================================================
# SECTION E — (b) Build sequence arrays for Deep Models (GRU & Graph+GRU)
#             X shape: (samples, 12 past steps, 50 sensors, 5 features)
#             Y shape: (samples, 4 future horizons, 50 sensors)
# ===========================================================================
print("\n[E] Building 4D sequence arrays for GRU / Graph+GRU ...")

PAST_STEPS = 12     # 12 × 15 min = 3 hours of past history
HORIZONS   = 4      # +15, +30, +45, +60 min ahead

# Pivot speed_ratio into (time_steps, sensors) matrix
wide_sr = traffic_df.pivot(index="time", columns="link_id", values="speed_ratio")
wide_sr = wide_sr.reindex(columns=sensor_order).sort_index()
wide_sr = wide_sr.ffill().bfill()   # ensure no NaNs inside sequence tensor

times_arr = wide_sr.index
sr_mat    = wide_sr.values.astype(np.float32)   # (T, N)

# Align weather & time features to the exact same timestamps
w_aligned = weather_df.set_index("time").reindex(times_arr).ffill().bfill()
hours     = times_arr.hour.values
hour_sin  = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
hour_cos  = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
precip    = w_aligned["precip"].values.astype(np.float32)
temp      = w_aligned["temp"].values.astype(np.float32)

T_total, N_sens = sr_mat.shape
# Build tensor of shape (T_total, N_sens, 5)
# Features: [speed_ratio, hour_sin, hour_cos, precip, temp]
feat_tensor = np.zeros((T_total, N_sens, 5), dtype=np.float32)
feat_tensor[:, :, 0] = sr_mat
feat_tensor[:, :, 1] = hour_sin[:, None]
feat_tensor[:, :, 2] = hour_cos[:, None]
feat_tensor[:, :, 3] = precip[:, None]
feat_tensor[:, :, 4] = temp[:, None]

# Fit StandardScaler on TRAIN timestamps ONLY (Rule 6: no leakage!)
train_time_mask = times_arr <= t_train_end
scaler = StandardScaler()
scaler.fit(feat_tensor[train_time_mask].reshape(-1, 5))
feat_scaled = scaler.transform(feat_tensor.reshape(-1, 5)).reshape(T_total, N_sens, 5).astype(np.float32)

joblib.dump(scaler, os.path.join(DATA_PROCESSED_DIR, "deep_scaler.joblib"))

# Create sliding windows: use [t - 11 .. t] to predict [t+1, t+2, t+3, t+4]
X_list, Y_list, t_list = [], [], []
for t in range(PAST_STEPS - 1, T_total - HORIZONS):
    X_list.append(feat_scaled[t - PAST_STEPS + 1 : t + 1])   # (12, N, 5)
    Y_list.append(sr_mat[t + 1 : t + 1 + HORIZONS])           # (4, N)
    t_list.append(times_arr[t])

X_all = np.array(X_list, dtype=np.float32)
Y_all = np.array(Y_list, dtype=np.float32)
t_all = pd.DatetimeIndex(t_list)

# Split by time
m_tr = t_all <= t_train_end
m_va = (t_all > t_train_end) & (t_all <= t_val_end)
m_te = t_all > t_val_end

np.savez_compressed(os.path.join(DATA_PROCESSED_DIR, "deep_train.npz"), X=X_all[m_tr], Y=Y_all[m_tr])
np.savez_compressed(os.path.join(DATA_PROCESSED_DIR, "deep_val.npz"),   X=X_all[m_va], Y=Y_all[m_va])
np.savez_compressed(os.path.join(DATA_PROCESSED_DIR, "deep_test.npz"),  X=X_all[m_te], Y=Y_all[m_te])

print(f"  Deep X train shape : {X_all[m_tr].shape}  |  Y: {Y_all[m_tr].shape}")
print(f"  Deep X val shape   : {X_all[m_va].shape}  |  Y: {Y_all[m_va].shape}")
print(f"  Deep X test shape  : {X_all[m_te].shape}  |  Y: {Y_all[m_te].shape}")

# ===========================================================================
# SECTION F — Append to step summaries
# ===========================================================================
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 08: Build Features\n\n")
    f.write("**What we did:** Built past-only features for LightGBM (lags, rolling stats, "
            "train-only usual speed, weather, neighbour speed) and 4D sequence arrays "
            "(12 past steps → 4 future steps) for deep models, split 70/10/20 by time.\n\n")
    f.write(f"**What we got:** LightGBM train/val/test = "
            f"{len(lgb_train):,} / {len(lgb_val):,} / {len(lgb_test):,} rows; "
            f"Deep arrays train = {X_all[m_tr].shape}.\n\n")
    f.write(f"**Key number:** **{len(lgb_test):,} test rows** and **{X_all[m_te].shape[0]:,} "
            f"test sequence windows** — zero future leakage.\n\n---\n\n")

print("\n" + "=" * 65)
print("  STEP 08 DONE.")
print(f"  → Flat LightGBM splits saved (train={len(lgb_train):,}, val={len(lgb_val):,}, test={len(lgb_test):,})")
print(f"  → Deep sequence splits saved (X_train={X_all[m_tr].shape})")
print("=" * 65)
