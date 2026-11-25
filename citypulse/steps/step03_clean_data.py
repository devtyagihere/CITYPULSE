# =============================================================================
# STEP NAME  : step03 — Clean Data
# WHAT       : Fixes timezones, resamples all data to 15-minute steps,
#              fills short gaps (up to 1 hour), removes impossible speeds,
#              and computes free-flow speed + speed_ratio.
# WHY        : Raw data is messy — wrong values, different frequencies, missing
#              gaps. The model needs clean, consistent data to learn properly.
# INPUT      : data/raw/SYNTHETIC_traffic.csv (or real)
#              data/raw/SYNTHETIC_weather.csv
#              data/raw/SYNTHETIC_pm25.csv
# OUTPUT     : data/processed/traffic_clean.parquet
#              data/processed/weather_clean.parquet
#              data/processed/pm25_clean.parquet
#              data/processed/free_flow_speeds.csv
#              reports/step03_missing_heatmap.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (USE_SYNTHETIC, DATA_RAW_DIR, DATA_PROCESSED_DIR,
                    REPORTS_DIR, TIMEZONE, TRAIN_FRAC, RANDOM_SEED,
                    REAL_TRAFFIC_PATH, REAL_WEATHER_PATH)
from common import (remove_impossible_speeds, fill_short_gaps,
                    add_missing_mask, compute_free_flow_speed, add_speed_ratio)

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 03 — Clean Data")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Load raw data (same as step02, with tz fix)
# ===========================================================================
print("\n[A] Loading raw data ...")

def load_csv_tz(path, time_col):
    """Load CSV and make time column properly timezone-aware."""
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(TIMEZONE)
    return df

if USE_SYNTHETIC:
    traffic_df = load_csv_tz(os.path.join(DATA_RAW_DIR, "SYNTHETIC_traffic.csv"), "data_as_of")
    weather_df = load_csv_tz(os.path.join(DATA_RAW_DIR, "SYNTHETIC_weather.csv"), "time")
    pm25_df    = load_csv_tz(os.path.join(DATA_RAW_DIR, "SYNTHETIC_pm25.csv"),    "time")
    # Rename traffic time column
    traffic_df = traffic_df.rename(columns={"data_as_of": "time"})
else:
    print("  Mode: REAL NYC 2018-2019 dataset")
    raw_tr = pd.read_parquet(REAL_TRAFFIC_PATH, columns=["segment_id", "timestamp", "speed_mph"])
    traffic_df = pd.DataFrame({
        "link_id": raw_tr["segment_id"].astype(str),
        "time":    pd.to_datetime(raw_tr["timestamp"], utc=True).dt.tz_convert(TIMEZONE),
        "speed":   raw_tr["speed_mph"].astype(float),
        "borough": "Manhattan",
    })

    raw_w = pd.read_parquet(REAL_WEATHER_PATH)
    weather_df = pd.DataFrame({
        "time":     pd.to_datetime(raw_w["timestamp"], utc=True).dt.tz_convert(TIMEZONE),
        "temp":     (raw_w["temp_c"] * 9.0 / 5.0 + 32.0).astype(float),
        "humidity": raw_w["humidity_pct"].astype(float),
        "wind":     (raw_w["wind_kmh"] * 0.621371).astype(float),
        "precip":   (raw_w["precip_mm"] / 25.4).astype(float),
    })

    w_h = weather_df.sort_values("time").copy()
    w_h["pm25"] = (14.0 - 0.25 * w_h["wind"] - 18.0 * w_h["precip"]).clip(lower=2.0)
    pm25_df = w_h[["time", "pm25"]].copy()

print(f"  Traffic: {len(traffic_df):,} rows, {traffic_df['link_id'].nunique()} sensors")
print(f"  Weather: {len(weather_df):,} rows")
print(f"  PM2.5  : {len(pm25_df):,} rows")

# ===========================================================================
# SECTION B — Remove impossible speeds (before resampling)
# ===========================================================================
print("\n[B] Removing impossible speed values ...")
# Valid speeds: 0 to 120 mph. Outside this = sensor glitch.
traffic_df = remove_impossible_speeds(traffic_df, speed_col="speed", low=0, high=120)

# ===========================================================================
# SECTION C — Add a "was_missing" flag BEFORE we fill gaps
# ===========================================================================
print("\n[C] Recording which values were originally missing ...")
traffic_df = add_missing_mask(traffic_df, "speed")
# Count: how many are missing per sensor?
miss_before = traffic_df["speed"].isna().mean() * 100
print(f"  Missing speed before filling: {miss_before:.2f}%")

# ===========================================================================
# SECTION D — Resample traffic to 15-minute bins per sensor
# ===========================================================================
print("\n[D] Resampling traffic to 15-minute bins ...")

# Make time the index, then resample per sensor
traffic_df = traffic_df.set_index("time")

sensor_groups = []
for sensor_id, group in traffic_df.groupby("link_id"):
    # Resample: take mean of any readings within each 15-min window
    resampled = group["speed"].resample("15min").mean().rename("speed")
    
    # Add the missing flag: 1 if any original value was missing in that bin
    was_miss = group["speed_was_missing"].resample("15min").max()
    
    # Combine into a small dataframe
    s_df = pd.DataFrame({"speed": resampled, "speed_was_missing": was_miss})
    s_df["link_id"] = sensor_id
    s_df["borough"] = "Manhattan"
    sensor_groups.append(s_df)

traffic_15min = pd.concat(sensor_groups).reset_index()
traffic_15min = traffic_15min.rename(columns={"index": "time"})

print(f"  After resampling: {len(traffic_15min):,} rows "
      f"({traffic_15min['link_id'].nunique()} sensors)")

# ===========================================================================
# SECTION E — Fill short gaps (up to 1 hour = 4 steps of 15 min)
# ===========================================================================
print("\n[E] Filling short gaps (max 1 hour = 4 steps) ...")

traffic_15min = traffic_15min.sort_values(["link_id", "time"])

# Apply gap-filling per sensor
traffic_15min["speed"] = (traffic_15min
                          .groupby("link_id")["speed"]
                          .transform(lambda x: fill_short_gaps(x, max_gap_steps=4)))

miss_after = traffic_15min["speed"].isna().mean() * 100
print(f"  Missing speed after filling: {miss_after:.2f}%  (long gaps left as NaN)")

# ===========================================================================
# SECTION F — Resample weather and PM2.5 to 15-minute bins
# ===========================================================================
print("\n[F] Resampling weather and PM2.5 to 15-minute bins ...")

weather_df = weather_df.drop_duplicates("time").sort_values("time").set_index("time")
# Forward-fill weather (it only changes once per hour, so we repeat the hourly value)
weather_15min = weather_df.resample("15min").ffill().reset_index()
print(f"  Weather rows after resampling: {len(weather_15min):,}")

pm25_df = pm25_df.drop_duplicates("time").sort_values("time").set_index("time")
pm25_15min = pm25_df["pm25"].resample("15min").interpolate(method="linear").reset_index()
print(f"  PM2.5 rows after resampling: {len(pm25_15min):,}")

# ===========================================================================
# SECTION G — Compute free-flow speed (from TRAIN period only!)
# Rule 6: "No data leakage. Features use only past data."
# ===========================================================================
print("\n[G] Computing free-flow speed from TRAIN data only ...")

# Figure out where the train period ends
all_times  = traffic_15min["time"].sort_values()
n_total    = len(all_times.unique())
n_train    = int(n_total * TRAIN_FRAC)   # 70% of time steps = train
train_cutoff = all_times.unique()[n_train - 1]  # last time step of train period

print(f"  Train ends at: {train_cutoff}")

# Split traffic into train and the rest
train_mask   = traffic_15min["time"] <= train_cutoff
traffic_train = traffic_15min[train_mask]
print(f"  Train rows: {train_mask.sum():,}  |  Val+Test rows: {(~train_mask).sum():,}")

# Compute free-flow speed using ONLY train data
free_flow_dict = compute_free_flow_speed(traffic_train,
                                          speed_col="speed",
                                          sensor_col="link_id",
                                          percentile=85)

# ===========================================================================
# SECTION H — Add speed_ratio to ALL traffic data
# ===========================================================================
print("\n[H] Adding speed_ratio column ...")
traffic_15min = add_speed_ratio(traffic_15min,
                                 speed_col="speed",
                                 sensor_col="link_id",
                                 free_flow_dict=free_flow_dict)

congested_pct = (traffic_15min["speed_ratio"] < 0.5).mean() * 100
print(f"  speed_ratio < 0.5 (congested): {congested_pct:.1f}% of readings")
print(f"  speed_ratio stats: mean={traffic_15min['speed_ratio'].mean():.3f}, "
      f"min={traffic_15min['speed_ratio'].min():.3f}, "
      f"max={traffic_15min['speed_ratio'].max():.3f}")

# ===========================================================================
# SECTION I — Save cleaned data as Parquet (fast to load later)
# ===========================================================================
print("\n[I] Saving cleaned data ...")

t_path = os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet")
w_path = os.path.join(DATA_PROCESSED_DIR, "weather_clean.parquet")
p_path = os.path.join(DATA_PROCESSED_DIR, "pm25_clean.parquet")
ff_path = os.path.join(DATA_PROCESSED_DIR, "free_flow_speeds.csv")

traffic_15min.to_parquet(t_path, index=False)
weather_15min.to_parquet(w_path, index=False)
pm25_15min.to_parquet(p_path, index=False)

# Save free-flow speeds as a simple CSV for easy inspection
ff_df = pd.DataFrame(list(free_flow_dict.items()),
                     columns=["link_id", "free_flow_speed_mph"])
ff_df.to_csv(ff_path, index=False)

print(f"  Saved: {t_path}")
print(f"  Saved: {w_path}")
print(f"  Saved: {p_path}")
print(f"  Saved: {ff_path}")

# ===========================================================================
# SECTION J — Plot missing-data heatmap (shows which sensors/times are missing)
# ===========================================================================
print("\n[J] Plotting missing data heatmap ...")

# Pick first 10 sensors, first 7 days, for a readable heatmap
sensors_sample  = sorted(traffic_15min["link_id"].unique())[:10]
time_sample_end = traffic_15min["time"].min() + pd.Timedelta(days=7)

heatmap_data = traffic_15min[
    traffic_15min["link_id"].isin(sensors_sample) &
    (traffic_15min["time"] < time_sample_end)
].pivot_table(index="link_id", columns="time", values="speed_was_missing", fill_value=0)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("CITYPULSE — Step 03 Cleaning Results", fontsize=13, fontweight="bold")

# --- Plot 1: Missing data heatmap ---
im = axes[0].imshow(heatmap_data.values, aspect="auto", cmap="Reds",
                    interpolation="none", vmin=0, vmax=1)
axes[0].set_yticks(range(len(sensors_sample)))
axes[0].set_yticklabels(sensors_sample, fontsize=7)
axes[0].set_title("Missing data map — first 10 sensors, first 7 days (red = originally missing)")
axes[0].set_xlabel("Time steps (15-min bins)")
plt.colorbar(im, ax=axes[0], label="Was missing (1=yes)")

# --- Plot 2: Speed ratio distribution ---
axes[1].hist(traffic_15min["speed_ratio"].dropna(), bins=60,
             color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0.5, color="red", linestyle="--", label="Congested threshold (0.5)")
axes[1].axvline(1.0, color="green", linestyle="--", label="Free flow (1.0)")
axes[1].set_xlabel("Speed Ratio (speed / free-flow speed)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of speed_ratio — most readings are near free-flow (1.0)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "step03_cleaning_plots.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {plot_path}")

# ===========================================================================
# SECTION K — Append to step summaries
# ===========================================================================
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a") as f:
    f.write("## Step 03: Clean Data\n\n")
    f.write(f"**What we did:** Fixed timezones, removed impossible speeds, "
            f"resampled to 15-min bins, filled gaps ≤1 hour, computed "
            f"free-flow speed from train data, added speed_ratio.\n\n")
    f.write(f"**What we got:** Clean traffic with {len(traffic_15min):,} rows, "
            f"missing reduced from {miss_before:.1f}% to {miss_after:.1f}%.\n\n")
    f.write(f"**Key number:** **{congested_pct:.1f}%** of readings have "
            f"speed_ratio < 0.5 (congested). Train cutoff: `{train_cutoff}`.\n\n")
    f.write(f"**Plot:** `reports/step03_cleaning_plots.png`\n\n")
    f.write("---\n\n")

print(f"  Summary appended: {summary_path}")

print("\n" + "=" * 65)
print("  STEP 03 DONE.")
print(f"  → Traffic clean: {len(traffic_15min):,} rows | "
      f"Missing: {miss_before:.1f}% → {miss_after:.1f}%")
print(f"  → Congested readings (speed_ratio < 0.5): {congested_pct:.1f}%")
print(f"  → Train cutoff: {train_cutoff}")
print(f"  → Saved 3 parquet files + free_flow_speeds.csv")
print("=" * 65)
