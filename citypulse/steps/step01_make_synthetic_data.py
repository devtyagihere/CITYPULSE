# =============================================================================
# STEP NAME  : step01 — Make Synthetic Data
# WHAT       : Creates fake but realistic traffic, weather, and PM2.5 data.
#              Saves them as CSV files in data/raw/.
# WHY        : Real data is not ready yet. We need something to test the full
#              pipeline. The fake data has the SAME column names as the real data.
# INPUT      : Nothing (we generate everything from scratch).
# OUTPUT     : data/raw/SYNTHETIC_traffic.csv
#              data/raw/SYNTHETIC_weather.csv
#              data/raw/SYNTHETIC_pm25.csv
# =============================================================================

import sys, os

# Go to the project root so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from config import (RANDOM_SEED, SYNTH_N_SENSORS, SYNTH_START,
                    SYNTH_END, SYNTH_FREQ, DATA_RAW_DIR, REPORTS_DIR)

# Fix random seed so we get the same fake data every time
np.random.seed(RANDOM_SEED)

# ===========================================================================
# BIG WARNING — always remind ourselves this is fake data
# ===========================================================================
print("=" * 65)
print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

# ===========================================================================
# STEP 1A — Build the time index (15-minute steps, 6 months)
# ===========================================================================
print("\n[1A] Building time index ...")
# Every 15 minutes from Jan 1 to Jun 30 2022
time_index = pd.date_range(start=SYNTH_START, end=SYNTH_END,
                            freq="15min", tz="America/New_York")
n_steps = len(time_index)
print(f"  Time steps: {n_steps:,}  ({SYNTH_START} to {SYNTH_END})")

# ===========================================================================
# STEP 1B — Helper: typical speed for a given hour and weekday
# ===========================================================================
def base_speed(hour, is_weekend):
    """
    Return a 'typical' road speed (mph) based on time of day.
    Real-world pattern: slow during rush hour (8-9 AM, 5-7 PM), fast at night.
    Weekends are smoother — no sharp rush hour.
    """
    if is_weekend:
        # Weekend: gentle dip in the afternoon
        if 12 <= hour <= 18:
            return 22.0
        return 28.0
    else:
        # Weekday: two rush-hour dips
        if 7 <= hour <= 9:
            return 12.0   # morning rush
        elif 16 <= hour <= 19:
            return 10.0   # evening rush
        elif 0 <= hour <= 5:
            return 32.0   # late night, roads empty
        else:
            return 20.0   # normal daytime

# ===========================================================================
# STEP 1C — Generate TRAFFIC data (50 sensors, 15-min steps)
# ===========================================================================
print("\n[1C] Generating traffic data ...")

sensor_ids = [f"SENSOR_{i:03d}" for i in range(SYNTH_N_SENSORS)]

# Give every sensor a "personality" — some sensors are on slower roads
sensor_base_offset = np.random.uniform(-5, 5, SYNTH_N_SENSORS)

rows = []   # we will collect every (sensor, time, speed) row here

for t_idx, t in enumerate(time_index):
    hour       = t.hour
    is_weekend = (t.weekday() >= 5)
    
    # Weather effect: sometimes it rains — reduces speed by ~20%
    # We use a simple "rain probability" that changes every few hours
    rain_factor = 1.0
    if np.random.rand() < 0.07:   # 7% chance of rain at any step
        rain_factor = np.random.uniform(0.75, 0.95)

    for s_idx, sensor in enumerate(sensor_ids):
        # Base speed for this hour
        bs = base_speed(hour, is_weekend) + sensor_base_offset[s_idx]
        bs = max(bs, 2.0)   # speed cannot be negative

        # Add random noise around the base speed
        speed = bs * rain_factor + np.random.normal(0, 2.5)
        speed = max(speed, 0.5)   # floor at 0.5 mph (almost stopped)

        # Occasionally add a sensor glitch (impossible value)
        if np.random.rand() < 0.003:
            speed = np.random.choice([-1, 999])  # bad reading

        # Occasionally make data missing (NaN)
        if np.random.rand() < 0.02:
            speed = np.nan

        rows.append({"link_id"    : sensor,
                     "data_as_of" : t,
                     "speed"      : round(speed, 2) if not np.isnan(speed) else np.nan,
                     "borough"    : "Manhattan"})

traffic_df = pd.DataFrame(rows)

# Save to CSV
traffic_path = os.path.join(DATA_RAW_DIR, "SYNTHETIC_traffic.csv")
traffic_df.to_csv(traffic_path, index=False)
print(f"  Saved: {traffic_path}")
print(f"  Rows: {len(traffic_df):,}   Sensors: {traffic_df['link_id'].nunique()}")
print(f"  Date range: {traffic_df['data_as_of'].min()} → {traffic_df['data_as_of'].max()}")
missing_pct = traffic_df["speed"].isna().mean() * 100
print(f"  Missing speed values: {missing_pct:.1f}%")

# ===========================================================================
# STEP 1D — Generate WEATHER data (hourly, then we'll resample later)
# ===========================================================================
print("\n[1D] Generating weather data ...")

weather_times = pd.date_range(start=SYNTH_START, end=SYNTH_END,
                               freq="h", tz="America/New_York")

temps     = []   # temperature in °F
humidity  = []
wind      = []   # wind speed in mph
precip    = []   # precipitation in inches

for t in weather_times:
    hour = t.hour
    month = t.month
    # Temperature: warmer in June, cooler in January; cooler at night
    base_temp = 30 + (month - 1) * 6 + (hour - 12) * (-0.5)
    temp = base_temp + np.random.normal(0, 3)

    # Humidity: higher at night, higher in summer
    hum = 60 + (month * 2) + np.random.normal(0, 8)
    hum = np.clip(hum, 20, 100)

    # Wind speed (mph)
    spd = abs(np.random.normal(8, 4))

    # Precipitation (inches): mostly 0, occasionally non-zero
    if np.random.rand() < 0.07:
        prec = np.random.exponential(0.1)   # small rain event
    else:
        prec = 0.0

    temps.append(round(temp, 1))
    humidity.append(round(hum, 1))
    wind.append(round(spd, 1))
    precip.append(round(prec, 3))

weather_df = pd.DataFrame({
    "time"     : weather_times,
    "temp"     : temps,
    "humidity" : humidity,
    "wind"     : wind,
    "precip"   : precip,
})

weather_path = os.path.join(DATA_RAW_DIR, "SYNTHETIC_weather.csv")
weather_df.to_csv(weather_path, index=False)
print(f"  Saved: {weather_path}")
print(f"  Rows: {len(weather_df):,}")
print(f"  Temp range: {weather_df['temp'].min():.1f} to {weather_df['temp'].max():.1f} °F")
print(f"  Rain hours: {(weather_df['precip'] > 0).sum()}")

# ===========================================================================
# STEP 1E — Generate PM2.5 data (hourly)
# ===========================================================================
print("\n[1E] Generating PM2.5 (air quality) data ...")

pm25_times = pd.date_range(start=SYNTH_START, end=SYNTH_END,
                            freq="h", tz="America/New_York")

pm25_values = []

for i, t in enumerate(pm25_times):
    hour       = t.hour
    is_weekend = (t.weekday() >= 5)

    # PM2.5 goes up during rush hour (cars emit more)
    if not is_weekend and (7 <= hour <= 9 or 16 <= hour <= 19):
        base_pm25 = 25.0   # higher during rush hour
    else:
        base_pm25 = 10.0   # normal background level

    # Wind reduces PM2.5 (blows pollution away)
    wind_effect = -wind[i] * 0.3 if i < len(wind) else 0
    
    # Rain also cleans the air
    rain_effect = -precip[i] * 20 if i < len(precip) else 0

    # Random noise
    pm25 = base_pm25 + wind_effect + rain_effect + np.random.normal(0, 3)
    pm25 = max(pm25, 1.0)    # PM2.5 is always >= 0

    # Occasionally add a spike (pollution event)
    if np.random.rand() < 0.02:
        pm25 += np.random.uniform(20, 60)

    pm25_values.append(round(pm25, 2))

pm25_df = pd.DataFrame({"time": pm25_times, "pm25": pm25_values})

pm25_path = os.path.join(DATA_RAW_DIR, "SYNTHETIC_pm25.csv")
pm25_df.to_csv(pm25_path, index=False)
print(f"  Saved: {pm25_path}")
print(f"  Rows: {len(pm25_df):,}")
print(f"  PM2.5 range: {pm25_df['pm25'].min():.1f} to {pm25_df['pm25'].max():.1f} µg/m³")
print(f"  Mean PM2.5: {pm25_df['pm25'].mean():.2f} µg/m³")

# ===========================================================================
# STEP 1F — Write step summary to reports/step_summaries.md
# ===========================================================================
os.makedirs(REPORTS_DIR, exist_ok=True)   # make sure reports/ folder exists

summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "w") as f:
    f.write("# CITYPULSE Step Summaries\n\n")
    f.write("---\n\n")
    f.write("## Step 01: Make Synthetic Data\n\n")
    f.write(f"**What we did:** Created realistic fake traffic, weather, and PM2.5 data "
            f"for {SYNTH_N_SENSORS} sensors over 6 months (Jan–Jun 2022).\n\n")
    f.write(f"**What we got:** 3 CSV files in data/raw/ with the same column names "
            f"as the real data, so swapping is easy.\n\n")
    f.write(f"**Key number:** Traffic table has **{len(traffic_df):,} rows** "
            f"({SYNTH_N_SENSORS} sensors × {n_steps:,} time steps). "
            f"About {missing_pct:.1f}% values are intentionally missing.\n\n")
    f.write("**Note:** ⚠️ SYNTHETIC DATA — only for pipeline testing. "
            "Re-run with real data for real results.\n\n")
    f.write("---\n\n")

print(f"\n[summary] Written to: {summary_path}")

# ===========================================================================
# FINAL PRINT
# ===========================================================================
print("\n" + "=" * 65)
print("  STEP 01 DONE. Summary of files created:")
print(f"  → data/raw/SYNTHETIC_traffic.csv  ({len(traffic_df):,} rows)")
print(f"  → data/raw/SYNTHETIC_weather.csv  ({len(weather_df):,} rows)")
print(f"  → data/raw/SYNTHETIC_pm25.csv     ({len(pm25_df):,} rows)")
print(f"  → reports/step_summaries.md")
print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)
