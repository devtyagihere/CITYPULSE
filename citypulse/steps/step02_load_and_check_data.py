# =============================================================================
# STEP NAME  : step02 — Load and Check Data
# WHAT       : Loads the 3 raw CSV files, prints basic health statistics,
#              and plots one sensor's speed for one week.
# WHY        : Before doing any modelling, we must understand our data —
#              how many sensors, what dates, how much is missing.
# INPUT      : data/raw/SYNTHETIC_traffic.csv (or real traffic if USE_SYNTHETIC=False)
#              data/raw/SYNTHETIC_weather.csv
#              data/raw/SYNTHETIC_pm25.csv
# OUTPUT     : reports/step02_sensor_week_plot.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save plots to file without needing a screen
import matplotlib.pyplot as plt
from config import (USE_SYNTHETIC, DATA_RAW_DIR, REPORTS_DIR, TIMEZONE)
from common import load_real_traffic, load_real_weather, load_real_pm25

print("=" * 65)
print("  STEP 02 — Load and Check Data")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Load data (synthetic OR real, depending on the config switch)
# ===========================================================================
print("\n[A] Loading data ...")

if USE_SYNTHETIC:
    # --- Load synthetic files ---
    traffic_path = os.path.join(DATA_RAW_DIR, "SYNTHETIC_traffic.csv")
    weather_path = os.path.join(DATA_RAW_DIR, "SYNTHETIC_weather.csv")
    pm25_path    = os.path.join(DATA_RAW_DIR, "SYNTHETIC_pm25.csv")

    # utc=True: tells pandas to parse timezone-aware strings correctly
    traffic_df = pd.read_csv(traffic_path)
    traffic_df["data_as_of"] = pd.to_datetime(traffic_df["data_as_of"], utc=True)

    weather_df = pd.read_csv(weather_path)
    weather_df["time"] = pd.to_datetime(weather_df["time"], utc=True)

    pm25_df    = pd.read_csv(pm25_path)
    pm25_df["time"]    = pd.to_datetime(pm25_df["time"], utc=True)

    # Rename traffic time column to 'time' (consistent name from now on)
    traffic_df = traffic_df.rename(columns={"data_as_of": "time"})

else:
    # --- Load real files using the helper from common.py ---
    traffic_df = load_real_traffic(os.path.join(DATA_RAW_DIR, "DOT_Traffic_Speeds_NBE.csv"))
    weather_df = load_real_weather(os.path.join(DATA_RAW_DIR, "weather_knyc_2022.csv"))
    pm25_df    = load_real_pm25(os.path.join(DATA_RAW_DIR, "hourly_88101_2022.csv"))

print(f"  Traffic  : {len(traffic_df):>10,} rows loaded")
print(f"  Weather  : {len(weather_df):>10,} rows loaded")
print(f"  PM2.5    : {len(pm25_df):>10,} rows loaded")

# ===========================================================================
# SECTION B — Make timezone-aware (all times → America/New_York)
# ===========================================================================
print("\n[B] Fixing timezones ...")

def make_tz_aware(df, col):
    """Make a datetime column timezone-aware in New York time."""
    if df[col].dt.tz is None:
        # No timezone yet — assume UTC then convert to NY time
        df[col] = df[col].dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    else:
        df[col] = df[col].dt.tz_convert(TIMEZONE)
    return df

traffic_df = make_tz_aware(traffic_df, "time")
weather_df = make_tz_aware(weather_df, "time")
pm25_df    = make_tz_aware(pm25_df,    "time")

# ===========================================================================
# SECTION C — Print data health statistics
# ===========================================================================
print("\n[C] Data health check ...")
print("-" * 55)

# --- TRAFFIC ---
n_sensors  = traffic_df["link_id"].nunique()
date_start = traffic_df["time"].min()
date_end   = traffic_df["time"].max()
miss_speed = traffic_df["speed"].isna().mean() * 100
boroughs   = traffic_df["borough"].unique().tolist()

print(f"\n  TRAFFIC")
print(f"    Total rows     : {len(traffic_df):,}")
print(f"    Unique sensors : {n_sensors}")
print(f"    Date range     : {date_start.strftime('%Y-%m-%d')} to {date_end.strftime('%Y-%m-%d')}")
print(f"    Boroughs       : {boroughs}")
print(f"    Missing speed  : {miss_speed:.2f}%")
print(f"    Speed stats    :")
print(f"      min={traffic_df['speed'].min():.1f}, "
      f"max={traffic_df['speed'].max():.1f}, "
      f"mean={traffic_df['speed'].mean():.1f} mph")

# --- WEATHER ---
print(f"\n  WEATHER")
print(f"    Total rows     : {len(weather_df):,}")
print(f"    Date range     : {weather_df['time'].min().strftime('%Y-%m-%d')} "
      f"to {weather_df['time'].max().strftime('%Y-%m-%d')}")
for col in ["temp", "humidity", "wind", "precip"]:
    miss = weather_df[col].isna().mean() * 100
    print(f"    {col:<12} missing: {miss:.1f}%  "
          f"range: [{weather_df[col].min():.1f}, {weather_df[col].max():.1f}]")

# --- PM2.5 ---
print(f"\n  PM2.5")
print(f"    Total rows     : {len(pm25_df):,}")
print(f"    Date range     : {pm25_df['time'].min().strftime('%Y-%m-%d')} "
      f"to {pm25_df['time'].max().strftime('%Y-%m-%d')}")
miss_pm = pm25_df["pm25"].isna().mean() * 100
print(f"    Missing PM2.5  : {miss_pm:.2f}%")
print(f"    PM2.5 stats    : min={pm25_df['pm25'].min():.1f}, "
      f"max={pm25_df['pm25'].max():.1f}, mean={pm25_df['pm25'].mean():.2f} µg/m³")

print("-" * 55)

# ===========================================================================
# SECTION D — Plot one sensor's speed for one week
# ===========================================================================
print("\n[D] Plotting one sensor for one week ...")

# Pick the first sensor
sensor_id = traffic_df["link_id"].iloc[0]

# Pick the first week of data
week_start = traffic_df["time"].min()
week_end   = week_start + pd.Timedelta(days=7)

# Filter to that sensor and that week
mask = (traffic_df["link_id"] == sensor_id) & \
       (traffic_df["time"] >= week_start)   & \
       (traffic_df["time"] <  week_end)
week_df = traffic_df[mask].sort_values("time")

print(f"  Sensor: {sensor_id}   Week: {week_start.date()} to {week_end.date()}")
print(f"  Rows in this window: {len(week_df)}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(f"CITYPULSE — Step 02 Data Check\n(Sensor: {sensor_id}, First 7 days)",
             fontsize=13, fontweight="bold")

# --- Plot 1: Traffic speed ---
axes[0].plot(week_df["time"], week_df["speed"],
             color="steelblue", linewidth=0.8, label="Speed (mph)")
axes[0].set_ylabel("Speed (mph)")
axes[0].set_title("Traffic Speed — you can see morning and evening dips (rush hours)")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Weather (temp + precip) ---
weather_week = weather_df[
    (weather_df["time"] >= week_start) & (weather_df["time"] < week_end)
].sort_values("time")

ax2a = axes[1]
ax2b = ax2a.twinx()   # second y-axis for precip
ax2a.plot(weather_week["time"], weather_week["temp"],
          color="orange", linewidth=1.0, label="Temp (°F)")
ax2b.bar(weather_week["time"], weather_week["precip"],
         color="skyblue", alpha=0.5, label="Precip (in)", width=0.03)
ax2a.set_ylabel("Temp (°F)")
ax2b.set_ylabel("Precip (in)", color="skyblue")
ax2a.set_title("Weather — temperature and precipitation")
ax2a.legend(loc="upper left")
ax2a.grid(True, alpha=0.3)

# --- Plot 3: PM2.5 ---
pm25_week = pm25_df[
    (pm25_df["time"] >= week_start) & (pm25_df["time"] < week_end)
].sort_values("time")

axes[2].plot(pm25_week["time"], pm25_week["pm25"],
             color="tomato", linewidth=0.8, label="PM2.5 (µg/m³)")
axes[2].axhline(12, color="green",  linestyle="--", alpha=0.7, label="Good (<12)")
axes[2].axhline(35, color="orange", linestyle="--", alpha=0.7, label="Moderate (<35)")
axes[2].set_ylabel("PM2.5 (µg/m³)")
axes[2].set_title("PM2.5 Air Quality — spikes correspond to rush hours")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "step02_sensor_week_plot.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {plot_path}")

# ===========================================================================
# SECTION E — Append to step summaries
# ===========================================================================
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a") as f:
    f.write("## Step 02: Load and Check Data\n\n")
    f.write(f"**What we did:** Loaded all 3 datasets (traffic, weather, PM2.5), "
            f"checked for missing values, and plotted one sensor for one week.\n\n")
    f.write(f"**What we got:** {n_sensors} traffic sensors, date range "
            f"{date_start.strftime('%Y-%m-%d')} → {date_end.strftime('%Y-%m-%d')}, "
            f"speed missing = {miss_speed:.1f}%.\n\n")
    f.write(f"**Key number:** Total traffic rows = **{len(traffic_df):,}** "
            f"({n_sensors} sensors × time steps).\n\n")
    f.write(f"**Plot:** `reports/step02_sensor_week_plot.png`\n\n")
    f.write("---\n\n")

print(f"  Summary appended to: {summary_path}")

print("\n" + "=" * 65)
print("  STEP 02 DONE.")
print(f"  → {n_sensors} sensors, {len(traffic_df):,} traffic rows")
print(f"  → Missing speed: {miss_speed:.1f}%")
print(f"  → Plot saved: reports/step02_sensor_week_plot.png")
print("=" * 65)
