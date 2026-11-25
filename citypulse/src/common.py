# =============================================================================
# src/common.py — Shared cleaning and feature code
# Used the SAME way in training AND in the live pipeline (no copy-paste).
# Rule 7: "Use the same cleaning and feature code for training and live use."
# =============================================================================

import pandas as pd
import numpy as np

# ---- bring config ----
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TIMEZONE, CRISIS_CONGESTED_RATIO


# ===========================================================================
# SECTION 1 — Time helpers
# ===========================================================================

def to_local_time(df, time_col):
    """
    Make sure the time column is in America/New_York timezone.
    Why: all our data must share the same clock before we merge them.
    """
    df = df.copy()
    if df[time_col].dt.tz is None:
        # No timezone info yet — assume it is UTC, then convert
        df[time_col] = df[time_col].dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    else:
        df[time_col] = df[time_col].dt.tz_convert(TIMEZONE)
    return df


def resample_to_15min(df, time_col, value_col, sensor_col=None):
    """
    Resample any time series to 15-minute bins by averaging.
    If there is a sensor column, do it for each sensor separately.
    """
    df = df.set_index(time_col)
    if sensor_col:
        # Group by sensor, then resample
        resampled = (df.groupby(sensor_col)[value_col]
                       .resample("15T")
                       .mean()
                       .reset_index())
    else:
        resampled = df[value_col].resample("15T").mean().reset_index()
    return resampled


# ===========================================================================
# SECTION 2 — Cleaning helpers
# ===========================================================================

def remove_impossible_speeds(df, speed_col="speed", low=0, high=120):
    """
    Set speeds below 0 or above 120 mph to NaN (not real values).
    Why: sensors sometimes glitch; we do not want those numbers in the model.
    """
    df = df.copy()
    bad = (df[speed_col] < low) | (df[speed_col] > high)
    n_bad = bad.sum()
    df.loc[bad, speed_col] = np.nan   # mark as missing
    print(f"  [clean] Removed {n_bad} impossible speed values.")
    return df


def fill_short_gaps(series, max_gap_steps=4):
    """
    Fill short missing gaps by linear interpolation.
    max_gap_steps=4 means gaps up to 1 hour (4 × 15 min) are filled.
    Longer gaps are left as NaN (we do not guess if data is missing too long).
    """
    return series.interpolate(method="linear", limit=max_gap_steps)


def add_missing_mask(df, value_col):
    """
    Add a column that records 1 where data was originally missing, 0 otherwise.
    Why: the model can learn 'missing = unusual situation'.
    """
    df = df.copy()
    df[value_col + "_was_missing"] = df[value_col].isna().astype(int)
    return df


# ===========================================================================
# SECTION 3 — Free-flow speed (computed on TRAIN data only)
# ===========================================================================

def compute_free_flow_speed(train_df, speed_col="speed", sensor_col="link_id",
                             percentile=85):
    """
    Free-flow speed = the 85th-percentile speed for each sensor on TRAIN data.
    Why: this is the 'normal fast speed' — when roads are clear.
    We compute it on TRAIN only so we don't peek at the future.
    Returns a dict: {sensor_id -> free_flow_speed}
    """
    free_flow = (train_df.groupby(sensor_col)[speed_col]
                          .quantile(percentile / 100)
                          .to_dict())
    print(f"  [free-flow] Computed for {len(free_flow)} sensors "
          f"(85th percentile of training speeds).")
    return free_flow


def add_speed_ratio(df, speed_col="speed", sensor_col="link_id",
                    free_flow_dict=None):
    """
    speed_ratio = speed / free_flow_speed.
    1.0 = perfectly free,  0.5 = half speed (congested),  < 0.5 = very bad.
    Why: ratio is better than raw speed because roads have different speed limits.
    """
    df = df.copy()
    if free_flow_dict is None:
        raise ValueError("Pass the free_flow_dict computed from TRAIN data.")
    # Map each sensor's free-flow speed onto every row
    df["free_flow"] = df[sensor_col].map(free_flow_dict)
    # Avoid divide-by-zero (if free-flow is somehow 0, use 1)
    df["free_flow"] = df["free_flow"].replace(0, np.nan).fillna(1)
    df["speed_ratio"] = df[speed_col] / df["free_flow"]
    # Clip so ratio stays between 0 and 1.5 (small over-free-flow allowed)
    df["speed_ratio"] = df["speed_ratio"].clip(0, 1.5)
    return df


# ===========================================================================
# SECTION 4 — Feature helpers (used in step08 AND in live pipeline)
# ===========================================================================

def add_time_features(df, time_col="time"):
    """
    Add hour, day-of-week, and is_weekend columns.
    Why: traffic at 8 AM Monday is very different from 8 AM Sunday.
    """
    df = df.copy()
    df["hour"]       = df[time_col].dt.hour
    df["weekday"]    = df[time_col].dt.weekday      # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def add_lag_features(df, value_col, lags_steps, group_col=None):
    """
    Add past values as new columns.
    lags_steps = [1, 2, 4] means: 15 min ago, 30 min ago, 60 min ago.
    If group_col is given, lags are computed per sensor (no mixing sensors).
    """
    df = df.copy()
    for lag in lags_steps:
        col_name = f"{value_col}_lag{lag}"
        if group_col:
            df[col_name] = df.groupby(group_col)[value_col].shift(lag)
        else:
            df[col_name] = df[value_col].shift(lag)
    return df


def add_rolling_features(df, value_col, windows_steps, group_col=None):
    """
    Add rolling mean and std over past windows.
    windows_steps=[4] means: mean and std of the last 4 steps (1 hour).
    """
    df = df.copy()
    for w in windows_steps:
        mean_col = f"{value_col}_roll_mean_{w}"
        std_col  = f"{value_col}_roll_std_{w}"
        if group_col:
            df[mean_col] = (df.groupby(group_col)[value_col]
                              .transform(lambda x: x.shift(1).rolling(w).mean()))
            df[std_col]  = (df.groupby(group_col)[value_col]
                              .transform(lambda x: x.shift(1).rolling(w).std()))
        else:
            df[mean_col] = df[value_col].shift(1).rolling(w).mean()
            df[std_col]  = df[value_col].shift(1).rolling(w).std()
    return df


def pm25_to_aqi_category(pm25):
    """
    Convert a PM2.5 value (µg/m³) to an AQI word category.
    Returns e.g. 'Good', 'Moderate', 'Unhealthy', etc.
    """
    from config import AQI_BREAKPOINTS
    for low, high, label in AQI_BREAKPOINTS:
        if low <= pm25 <= high:
            return label
    return "Hazardous"


# ===========================================================================
# SECTION 5 — Real data loaders (rename columns to match our standard names)
# ===========================================================================

def load_real_traffic(filepath):
    """
    Load a REAL NYC DOT traffic CSV and rename columns to our standard names.
    Standard names: link_id, time, speed, borough
    Usage: df = load_real_traffic("data/raw/DOT_Traffic_Speeds_NBE.csv")
    """
    df = pd.read_csv(filepath, parse_dates=["DATA_AS_OF"])
    # Rename real column names to our names
    df = df.rename(columns={
        "LINK_ID"    : "link_id",
        "DATA_AS_OF" : "time",
        "SPEED"      : "speed",
        "BOROUGH"    : "borough",
    })
    # Keep only what we need
    df = df[["link_id", "time", "speed", "borough"]]
    print(f"  [real traffic] Loaded {len(df):,} rows from {filepath}")
    return df


def load_real_pm25(filepath):
    """
    Load a REAL EPA PM2.5 CSV (hourly_88101_YEAR.csv) and rename columns.
    Standard names: time, pm25
    """
    df = pd.read_csv(filepath)
    # Build a datetime from the Date and Time columns EPA provides
    df["time"] = pd.to_datetime(df["Date Local"] + " " + df["Time Local"])
    df = df.rename(columns={"Sample Measurement": "pm25"})
    df = df[["time", "pm25"]]
    print(f"  [real pm25] Loaded {len(df):,} rows from {filepath}")
    return df


def load_real_weather(filepath):
    """
    Load a REAL NOAA KNYC weather CSV and rename columns.
    Standard names: time, temp, humidity, wind, precip
    """
    df = pd.read_csv(filepath, parse_dates=["DATE"])
    df = df.rename(columns={
        "DATE"      : "time",
        "HourlyDryBulbTemperature"      : "temp",
        "HourlyRelativeHumidity"        : "humidity",
        "HourlyWindSpeed"               : "wind",
        "HourlyPrecipitation"           : "precip",
    })
    df = df[["time", "temp", "humidity", "wind", "precip"]]
    print(f"  [real weather] Loaded {len(df):,} rows from {filepath}")
    return df


# ===========================================================================
# SECTION 6 — Shared feature builder (used by step08 AND live pipeline)
# ===========================================================================

# Major US federal holidays (covers 2022 + general fixed/common dates)
US_HOLIDAY_DATES = {
    "2022-01-01", "2022-01-17", "2022-02-21", "2022-05-30",
    "2022-06-20", "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-25",
}


def add_holiday_feature(df, time_col="time"):
    """Add 1 if date is a US holiday, 0 otherwise."""
    df = df.copy()
    date_str = df[time_col].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = date_str.isin(US_HOLIDAY_DATES).astype(int)
    return df


def build_traffic_features(traffic_df, weather_df, adj_matrix=None,
                           sensor_order=None, usual_speed_df=None):
    """
    Build flat ML features for traffic prediction using ONLY past & current data.
    Used in step08 (training) AND step22 (live predictions) so there is NO mismatch.
    """
    df = traffic_df.sort_values(["link_id", "time"]).copy()

    # 1. Time & holiday features
    df = add_time_features(df, time_col="time")
    df = add_holiday_feature(df, time_col="time")

    # 2. Lag features: now (0), -15m (1), -30m (2), -60m (4)
    df["lag_0"] = df["speed_ratio"]
    df["lag_1"] = df.groupby("link_id")["speed_ratio"].shift(1)
    df["lag_2"] = df.groupby("link_id")["speed_ratio"].shift(2)
    df["lag_4"] = df.groupby("link_id")["speed_ratio"].shift(4)

    # 3. Rolling mean & std of the last hour (4 steps: t, t-15, t-30, t-45)
    df["roll_mean_1h"] = (df.groupby("link_id")["speed_ratio"]
                            .transform(lambda s: s.rolling(4, min_periods=1).mean()))
    df["roll_std_1h"]  = (df.groupby("link_id")["speed_ratio"]
                            .transform(lambda s: s.rolling(4, min_periods=1).std())
                            .fillna(0.0))

    # 4. Merge weather (temp, precip, wind) by timestamp
    w_cols = weather_df[["time", "temp", "precip", "wind"]].drop_duplicates("time")
    df = df.merge(w_cols, on="time", how="left")
    df["temp"]   = df["temp"].ffill().bfill()
    df["precip"] = df["precip"].fillna(0.0)
    df["wind"]   = df["wind"].ffill().bfill()

    # 5. Usual speed for (link_id, hour, weekday) — computed on TRAIN only!
    if usual_speed_df is not None:
        df = df.merge(usual_speed_df, on=["link_id", "hour", "weekday"], how="left")
        df["usual_speed_ratio"] = df["usual_speed_ratio"].fillna(df["lag_0"])

    # 6. Neighbour average speed_ratio at time t (from road adjacency matrix)
    if adj_matrix is not None and sensor_order is not None:
        # Pivot to (time x sensors) matrix, multiply by normalized adj_matrix
        wide = df.pivot(index="time", columns="link_id", values="speed_ratio")
        wide = wide.reindex(columns=sensor_order)
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        w_norm = adj_matrix / row_sums
        # Fill missing temporarily with row mean just for neighbour aggregation
        wide_filled = wide.T.fillna(wide.mean(axis=1)).T.fillna(0.7)
        nbr_vals = wide_filled.values @ w_norm.T
        nbr_df = pd.DataFrame(nbr_vals, index=wide.index, columns=sensor_order)
        nbr_long = nbr_df.reset_index().melt(
            id_vars="time", var_name="link_id", value_name="neighbour_speed_ratio"
        )
        df = df.merge(nbr_long, on=["time", "link_id"], how="left")
        df["neighbour_speed_ratio"] = df["neighbour_speed_ratio"].fillna(df["lag_0"])

    return df

