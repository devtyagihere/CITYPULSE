# =============================================================================
# STEP NAME  : step07 — Data Leakage Test (Write This FIRST!)
# WHAT       : Trains a quick model TWO ways on one week of data:
#              1) HONEST: uses only past speeds (t, t-15, t-30 min)
#              2) LEAKY : deliberately sneaks in a FUTURE speed (t+15 min)
# WHY        : To prove how dangerous data leakage is. If a model peeks at the
#              future, its error drops unrealistically close to zero — which
#              fools us into thinking the model is amazing, when in real life
#              it will fail because future data does not exist yet!
# INPUT      : data/processed/traffic_clean.parquet
# OUTPUT     : reports/step07_leakage_test.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from config import DATA_PROCESSED_DIR, REPORTS_DIR, RANDOM_SEED, USE_SYNTHETIC

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 07 — Data Leakage Test")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Load clean traffic data (use first 10 sensors to run fast)
# ===========================================================================
print("\n[A] Loading clean traffic data ...")
traffic_path = os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet")
df = pd.read_parquet(traffic_path)

# Take 10 sensors so this test finishes in ~2 seconds
sample_sensors = sorted(df["link_id"].unique())[:10]
df = df[df["link_id"].isin(sample_sensors)].sort_values(["link_id", "time"]).copy()
print(f"  Using {len(sample_sensors)} sensors, {len(df):,} rows")

# ===========================================================================
# SECTION B — Build honest features (PAST ONLY) + target (+15 min ahead)
# ===========================================================================
print("\n[B] Building features ...")

# Target: speed_ratio 1 step (15 minutes) in the future
df["target_15m"] = df.groupby("link_id")["speed_ratio"].shift(-1)

# Honest features: only values we ALREADY know at time t
df["lag_0"]  = df["speed_ratio"]                                   # speed right now
df["lag_1"]  = df.groupby("link_id")["speed_ratio"].shift(1)       # 15 min ago
df["lag_2"]  = df.groupby("link_id")["speed_ratio"].shift(2)       # 30 min ago
df["lag_4"]  = df.groupby("link_id")["speed_ratio"].shift(4)       # 60 min ago
df["hour"]   = df["time"].dt.hour

# DELIBERATE LEAKY FEATURE: peeking at the future (t + 15 min)!
# In real life, at time t we do NOT have this value!
df["LEAKY_future_speed"] = df.groupby("link_id")["speed_ratio"].shift(-1)

# Drop rows with NaN from shifting
df = df.dropna(subset=["target_15m", "lag_1", "lag_2", "lag_4", "LEAKY_future_speed"]).copy()

# Split by TIME (first 80% = train, last 20% = test) — NEVER random!
unique_times = np.sort(df["time"].unique())
split_time   = unique_times[int(len(unique_times) * 0.80)]

train_df = df[df["time"] <= split_time]
test_df  = df[df["time"] >  split_time]
print(f"  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

# ===========================================================================
# SECTION C — Train HONEST model vs LEAKY model
# ===========================================================================
print("\n[C] Training HONEST model (past only) vs LEAKY model (peeks at future) ...")

honest_cols = ["lag_0", "lag_1", "lag_2", "lag_4", "hour"]
leaky_cols  = ["lag_0", "lag_1", "lag_2", "lag_4", "hour", "LEAKY_future_speed"]

# 1. Honest model
m_honest = HistGradientBoostingRegressor(max_iter=50, random_state=RANDOM_SEED)
m_honest.fit(train_df[honest_cols], train_df["target_15m"])
pred_honest = m_honest.predict(test_df[honest_cols])
mae_honest  = mean_absolute_error(test_df["target_15m"], pred_honest)

# 2. Leaky model
m_leaky = HistGradientBoostingRegressor(max_iter=50, random_state=RANDOM_SEED)
m_leaky.fit(train_df[leaky_cols], train_df["target_15m"])
pred_leaky = m_leaky.predict(test_df[leaky_cols])
mae_leaky  = mean_absolute_error(test_df["target_15m"], pred_leaky)

print(f"\n  HONEST model MAE (past only)      : {mae_honest:.5f}")
print(f"  LEAKY  model MAE (uses future!)   : {mae_leaky:.5f}")
print(f"  Error drop from cheating          : {(1 - mae_leaky / mae_honest) * 100:.1f}%")

print("\n  WHY THIS MATTERS:")
print("  - The leaky model looks 'almost perfect' because it cheated and looked")
print("    at the answer (future speed at t+15 min).")
print("  - In real life, an ambulance dispatch system does NOT know t+15 min yet!")
print("  - That is why we ALWAYS split by TIME and NEVER use future columns.")

# ===========================================================================
# SECTION D — Plot comparison
# ===========================================================================
print("\n[D] Plotting leakage comparison ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("CITYPULSE — Step 07: Data Leakage Test", fontsize=13, fontweight="bold")

# Plot 1: MAE bar chart
bars = axes[0].bar(["Honest Model\n(Past Only)", "Leaky Model\n(Peeks at Future)"],
                   [mae_honest, mae_leaky],
                   color=["steelblue", "crimson"], width=0.5)
axes[0].set_ylabel("Test MAE (speed_ratio)")
axes[0].set_title("Unrealistically Low Error = Data Leakage!")
for bar, val in zip(bars, [mae_honest, mae_leaky]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.4f}", ha="center", fontweight="bold")
axes[0].grid(True, axis="y", alpha=0.3)

# Plot 2: First 50 test points for one sensor
one_sensor = test_df[test_df["link_id"] == sample_sensors[0]].head(50)
p_h_one = m_honest.predict(one_sensor[honest_cols])
p_l_one = m_leaky.predict(one_sensor[leaky_cols])

axes[1].plot(one_sensor["target_15m"].values, label="True future", color="black", linewidth=2)
axes[1].plot(p_h_one, label="Honest prediction", color="steelblue", linestyle="--")
axes[1].plot(p_l_one, label="Leaky (cheating)",  color="crimson",   linestyle=":")
axes[1].set_title(f"Predictions on {sample_sensors[0]} (first 50 test steps)")
axes[1].set_xlabel("Time steps (15 min)")
axes[1].set_ylabel("speed_ratio")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "step07_leakage_test.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {plot_path}")

# ===========================================================================
# SECTION E — Append to step summaries
# ===========================================================================
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 07: Data Leakage Test\n\n")
    f.write("**What we did:** Trained an honest model (past data only) and a "
            "leaky model (deliberately given future speed at t+15 min).\n\n")
    f.write(f"**What we got:** Honest MAE = **{mae_honest:.4f}**, whereas "
            f"Leaky MAE dropped unrealistically to **{mae_leaky:.4f}**.\n\n")
    f.write(f"**Key number:** Leakage caused a **{(1 - mae_leaky / mae_honest) * 100:.1f}%** "
            f"fake error drop — proving why we must strictly split by time and never touch future rows.\n\n")
    f.write("**Plot:** `reports/step07_leakage_test.png`\n\n---\n\n")

print("\n" + "=" * 65)
print("  STEP 07 DONE.")
print(f"  → Honest MAE: {mae_honest:.4f}  |  Leaky MAE: {mae_leaky:.4f}")
print("=" * 65)
