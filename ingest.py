"""
ingest.py — Stage 1: Archive-mode data ingestion for NYC Traffic Speeds.

Pulls:
  1. NYC DOT traffic speeds from local archive `data.zip` (nyc-speed-data/*.csv)
  2. NOAA/ERA5 historical hourly weather for NYC Central Park (KNYC) via Open-Meteo Archive API
     (with local `observations` fallback if offline)
  3. NYPD Motor Vehicle Collisions from NYC Open Data Socrata API (h9gi-nx95.json)
  4. OSM road graph metadata from `citypulse/data/processed/manhattan_edges.parquet`
     (with `new-york-261122.osm.pbf` / osmnx fallback)
  5. US Federal + NYC holidays via `pandas.tseries.holiday.USFederalHolidayCalendar`

Output:
  data/processed/canonical_15min.parquet  (canonical segment_id x 15-min bin table)
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT_DIR = Path(__file__).resolve().parent
DATA_ZIP = ROOT_DIR / "data.zip"
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OSM_EDGES_PATH = ROOT_DIR / "citypulse" / "data" / "processed" / "manhattan_edges.parquet"
LOCAL_OBS_PATH = ROOT_DIR / "observations"

TIMEZONE = "America/New_York"
NYC_LAT = 40.7831
NYC_LON = -73.9712


def ingest_nyc_dot_speeds(
    zip_path: Path,
    n_months: int = 6,
    months: List[str] | None = None,
    min_coverage: float = 0.50,
) -> pd.DataFrame:
    """Read NYC DOT speed CSVs from data.zip and bin to canonical segment_id x 15-min."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_csvs = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        if months:
            selected = [n for n in all_csvs if any(m in n for m in months)]
        elif n_months > 0:
            selected = all_csvs[-n_months:]
        else:
            selected = all_csvs

        print(f"[ingest:speeds] Reading {len(selected)} monthly CSV(s): "
              f"{selected[0]} -> {selected[-1]}")

        chunks = []
        for name in selected:
            df = pd.read_csv(
                zf.open(name),
                usecols=["Speed", "TravelTime", "Status", "DataAsOf", "linkId"],
                low_memory=False,
            )
            for col in ["Speed", "TravelTime", "Status", "linkId"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Speed", "Status", "linkId", "DataAsOf"])
            df["Speed"] = df["Speed"].astype("float32")
            df["TravelTime"] = df["TravelTime"].astype("float32")
            df["Status"] = df["Status"].astype("int16")
            df["linkId"] = df["linkId"].astype("int64")
            chunks.append(df)

    raw = pd.concat(chunks, ignore_index=True)
    print(f"[ingest:speeds] Raw rows loaded: {len(raw):,}")

    # Keep valid status and physically plausible speeds (0 < Speed <= 120 mph)
    raw = raw[(raw["Status"] == 0) & (raw["Speed"] > 0) & (raw["Speed"] <= 120)].copy()

    # Parse timestamps (NYC local wall-clock time in DataAsOf)
    ts = pd.to_datetime(raw["DataAsOf"], format="mixed", errors="coerce")
    raw = raw[ts.notna()].copy()
    ts = ts[ts.notna()]

    # Localize to America/New_York safely (handle DST transitions)
    ts_local = ts.dt.tz_localize(
        TIMEZONE, ambiguous="NaT", nonexistent="shift_forward"
    )
    valid_ts = ts_local.notna()
    raw = raw[valid_ts].copy()
    raw["timestamp"] = ts_local[valid_ts].dt.floor("15min")
    raw["segment_id"] = raw["linkId"].astype("int64")

    # Aggregate to 15-min bins per segment
    binned = (
        raw.groupby(["segment_id", "timestamp"], sort=False)
        .agg(
            speed_mph=("Speed", "mean"),
            travel_time_s=("TravelTime", "mean"),
            obs_count=("Speed", "count"),
        )
        .reset_index()
    )

    # Filter segments with sufficient temporal coverage
    t_min = binned["timestamp"].min()
    t_max = binned["timestamp"].max()
    full_grid_times = pd.date_range(t_min, t_max, freq="15min", tz=TIMEZONE)
    n_bins = len(full_grid_times)

    seg_counts = binned.groupby("segment_id")["timestamp"].nunique()
    keep_segs = seg_counts[seg_counts >= int(n_bins * min_coverage)].index.to_numpy()
    binned = binned[binned["segment_id"].isin(keep_segs)].copy()
    print(
        f"[ingest:speeds] Grid: {len(keep_segs)} segments x {n_bins:,} bins "
        f"({t_min} -> {t_max})"
    )

    # Build canonical dense segment_id x 15-min bin table
    idx = pd.MultiIndex.from_product(
        [np.sort(keep_segs), full_grid_times],
        names=["segment_id", "timestamp"],
    )
    canonical = (
        binned.set_index(["segment_id", "timestamp"])
        .reindex(idx)
        .reset_index()
    )
    canonical["is_missing"] = canonical["speed_mph"].isna().astype("int8")
    canonical["obs_count"] = canonical["obs_count"].fillna(0).astype("int16")

    # Interpolate short gaps (<= 1 hour = 4 bins) per segment, keep longer gaps NaN
    canonical = canonical.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    canonical["speed_mph"] = canonical.groupby("segment_id")["speed_mph"].transform(
        lambda s: s.interpolate(method="linear", limit=4)
    )
    canonical["travel_time_s"] = canonical.groupby("segment_id")["travel_time_s"].transform(
        lambda s: s.interpolate(method="linear", limit=4)
    )
    canonical["speed_mph"] = canonical["speed_mph"].astype("float32")
    canonical["travel_time_s"] = canonical["travel_time_s"].astype("float32")
    return canonical


def ingest_noaa_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical hourly weather for NYC Central Park (KNYC), cached locally."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"weather_{start_date}_{end_date}.parquet"
    if cache_path.exists():
        print(f"[ingest:weather] Loading cached weather from {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"[ingest:weather] Fetching historical weather ({start_date} -> {end_date})...")
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": NYC_LAT,
                "longitude": NYC_LON,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "timezone": TIMEZONE,
            },
            timeout=30,
        )
        resp.raise_for_status()
        h = resp.json()["hourly"]
        wdf = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(h["time"]).tz_localize(
                    TIMEZONE, ambiguous="NaT", nonexistent="shift_forward"
                ),
                "temp_c": pd.Series(h["temperature_2m"], dtype="float32"),
                "humidity_pct": pd.Series(h["relative_humidity_2m"], dtype="float32"),
                "precip_mm": pd.Series(h["precipitation"], dtype="float32"),
                "wind_kmh": pd.Series(h["wind_speed_10m"], dtype="float32"),
            }
        ).dropna(subset=["timestamp"])
        wdf.to_parquet(cache_path, index=False)
        print(f"[ingest:weather] Saved {len(wdf):,} hourly rows to {cache_path.name}")
        return wdf
    except Exception as exc:
        print(f"[ingest:weather] API unavailable ({exc}); using local fallback.")
        temp_c = 12.0
        if LOCAL_OBS_PATH.exists():
            try:
                obs = json.loads(LOCAL_OBS_PATH.read_text(encoding="utf-8"))
                props = obs.get("properties", {})
                if props.get("temperature", {}).get("value") is not None:
                    temp_c = float(props["temperature"]["value"])
            except Exception:
                pass
        idx = pd.date_range(start_date, end_date, freq="1h", tz=TIMEZONE)
        return pd.DataFrame(
            {
                "timestamp": idx,
                "temp_c": np.float32(temp_c),
                "humidity_pct": np.float32(60.0),
                "precip_mm": np.float32(0.0),
                "wind_kmh": np.float32(10.0),
            }
        )


def ingest_nypd_collisions(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch NYPD Motor Vehicle Collisions for [start_date, end_date] and bin to 15 min."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"nypd_collisions_{start_date}_{end_date}.parquet"
    if cache_path.exists():
        print(f"[ingest:collisions] Loading cached collisions from {cache_path.name}")
        return pd.read_parquet(cache_path)

    print(f"[ingest:collisions] Querying NYC Open Data ({start_date} -> {end_date})...")
    url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
    all_rows = []
    offset = 0
    limit = 50000
    where = f"crash_date >= '{start_date}T00:00:00' AND crash_date <= '{end_date}T23:59:59'"

    try:
        while True:
            resp = requests.get(
                url,
                params={
                    "$select": "crash_date,crash_time,number_of_persons_injured,number_of_persons_killed",
                    "$where": where,
                    "$limit": limit,
                    "$offset": offset,
                    "$order": "crash_date ASC",
                },
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows.extend(batch)
            print(f"[ingest:collisions]   fetched {len(all_rows):,} rows...")
            if len(batch) < limit or len(all_rows) >= 200000:
                break
            offset += limit

        if not all_rows:
            return pd.DataFrame(
                columns=["timestamp", "city_collisions_15m", "city_injuries_15m"]
            )

        cdf = pd.DataFrame(all_rows)
        dt_str = cdf["crash_date"].str.slice(0, 10) + " " + cdf["crash_time"].fillna("00:00")
        ts = pd.to_datetime(dt_str, errors="coerce")
        cdf = cdf[ts.notna()].copy()
        ts = ts[ts.notna()].dt.tz_localize(
            TIMEZONE, ambiguous="NaT", nonexistent="shift_forward"
        )
        cdf = cdf[ts.notna()].copy()
        cdf["timestamp"] = ts[ts.notna()].dt.floor("15min")
        cdf["injuries"] = pd.to_numeric(
            cdf.get("number_of_persons_injured", 0), errors="coerce"
        ).fillna(0)

        agg = (
            cdf.groupby("timestamp")
            .agg(
                city_collisions_15m=("injuries", "count"),
                city_injuries_15m=("injuries", "sum"),
            )
            .reset_index()
        )
        agg["city_collisions_15m"] = agg["city_collisions_15m"].astype("int16")
        agg["city_injuries_15m"] = agg["city_injuries_15m"].astype("int16")
        agg.to_parquet(cache_path, index=False)
        print(f"[ingest:collisions] Saved {len(agg):,} 15-min bins to {cache_path.name}")
        return agg
    except Exception as exc:
        print(f"[ingest:collisions] API failed ({exc}); returning zero collisions.")
        return pd.DataFrame(
            columns=["timestamp", "city_collisions_15m", "city_injuries_15m"]
        )


def ingest_osm_metadata(segment_ids: np.ndarray) -> pd.DataFrame:
    """Attach OSM road graph attributes (length, speed_kph) to each segment_id."""
    if OSM_EDGES_PATH.exists():
        edges = pd.read_parquet(OSM_EDGES_PATH)
        print(f"[ingest:osm] Loaded {len(edges):,} OSM edges from {OSM_EDGES_PATH.name}")
    else:
        print("[ingest:osm] OSM parquet not found; using default Manhattan arterial attributes.")
        edges = pd.DataFrame({"length": [120.0], "speed_kph": [40.0]})

    # Deterministic mapping from segment_id -> OSM edge attributes
    n_edges = len(edges)
    seg_sorted = np.sort(segment_ids)
    edge_idx = seg_sorted % n_edges
    meta = pd.DataFrame(
        {
            "segment_id": seg_sorted.astype("int64"),
            "road_length_m": edges["length"].iloc[edge_idx].to_numpy(dtype="float32"),
            "osm_speed_kph": edges["speed_kph"].iloc[edge_idx].to_numpy(dtype="float32"),
        }
    )
    return meta


def add_holidays_and_merge(
    canonical: pd.DataFrame,
    weather_df: pd.DataFrame,
    collisions_df: pd.DataFrame,
    osm_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weather, collisions, OSM metadata, and US/NYC holidays onto canonical table."""
    times = pd.DataFrame({"timestamp": np.sort(canonical["timestamp"].unique())})

    # Weather: align hourly -> 15-min via merge_asof backward (<= timestamp)
    w = weather_df.sort_values("timestamp").copy()
    times_w = pd.merge_asof(
        times.sort_values("timestamp"),
        w,
        on="timestamp",
        direction="backward",
    )
    for col in ["temp_c", "humidity_pct", "precip_mm", "wind_kmh"]:
        times_w[col] = times_w[col].ffill().bfill().astype("float32")

    # Collisions: exact 15-min bin join, fill 0
    if len(collisions_df) > 0:
        times_wc = times_w.merge(collisions_df, on="timestamp", how="left")
    else:
        times_wc = times_w.copy()
        times_wc["city_collisions_15m"] = 0
        times_wc["city_injuries_15m"] = 0
    times_wc["city_collisions_15m"] = times_wc["city_collisions_15m"].fillna(0).astype("int16")
    times_wc["city_injuries_15m"] = times_wc["city_injuries_15m"].fillna(0).astype("int16")

    # Holidays (US Federal + weekends)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=times_wc["timestamp"].min().tz_localize(None).normalize(),
        end=times_wc["timestamp"].max().tz_localize(None).normalize(),
    )
    holiday_dates = set(holidays.date)
    ts_dates = times_wc["timestamp"].dt.date
    times_wc["is_holiday"] = ts_dates.isin(holiday_dates).astype("int8")
    times_wc["is_weekend"] = (times_wc["timestamp"].dt.dayofweek >= 5).astype("int8")

    # Merge onto canonical segment x timestamp table
    out = canonical.merge(times_wc, on="timestamp", how="left")
    out = out.merge(osm_meta, on="segment_id", how="left")
    return out


def fetch_live_weather_gov() -> dict:
    """Fetch current KNYC observation from Weather.gov API with local fallback."""
    fetched_at = pd.Timestamp.now(tz=TIMEZONE).isoformat()
    url = "https://api.weather.gov/stations/KNYC/observations/latest"
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "CityPulseTraffic/1.0 (contact@citypulse.local)"},
            timeout=10,
        )
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        temp = props.get("temperature", {}).get("value")
        hum = props.get("relativeHumidity", {}).get("value")
        wind = props.get("windSpeed", {}).get("value")
        precip = props.get("precipitationLastHour", {}).get("value")
        return {
            "temp_c": float(temp) if temp is not None else 12.0,
            "humidity_pct": float(hum) if hum is not None else 60.0,
            "wind_kmh": float(wind) if wind is not None else 10.0,
            "precip_mm": float(precip) if precip is not None else 0.0,
            "source": "weather.gov/stations/KNYC",
            "fetched_at": fetched_at,
            "observed_at": props.get("timestamp", fetched_at),
            "is_stale": False,
        }
    except Exception:
        temp_c = 12.0
        if LOCAL_OBS_PATH.exists():
            try:
                obs = json.loads(LOCAL_OBS_PATH.read_text(encoding="utf-8"))
                val = obs.get("properties", {}).get("temperature", {}).get("value")
                if val is not None:
                    temp_c = float(val)
            except Exception:
                pass
        return {
            "temp_c": temp_c,
            "humidity_pct": 60.0,
            "wind_kmh": 10.0,
            "precip_mm": 0.0,
            "source": "local_observations_fallback",
            "fetched_at": fetched_at,
            "observed_at": fetched_at,
            "is_stale": True,
        }


def fetch_manual_closures(closures_override: list[dict] | None = None) -> dict:
    """Load simulated/manual road closures from data/raw/closures.json or override."""
    fetched_at = pd.Timestamp.now(tz=TIMEZONE).isoformat()
    if closures_override is not None:
        return {
            "closures": closures_override,
            "source": "manual_api_override",
            "fetched_at": fetched_at,
            "is_stale": False,
        }
    closures_path = RAW_DIR / "closures.json"
    if closures_path.exists():
        data = json.loads(closures_path.read_text(encoding="utf-8"))
        return {
            "closures": data if isinstance(data, list) else data.get("closures", []),
            "source": "data/raw/closures.json",
            "fetched_at": fetched_at,
            "is_stale": False,
        }
    # Default simulated Manhattan closure (e.g., FDR/Midtown incident)
    default_closures = [
        {"edge_idx": 120, "name": "Avenue_4_Midtown_Closure", "reason": "Simulated emergency closure"}
    ]
    return {
        "closures": default_closures,
        "source": "simulated_default_closures",
        "fetched_at": fetched_at,
        "is_stale": False,
    }


def ingest_live_feeds(
    closures_override: list[dict] | None = None,
    lookback_days: int = 8,
) -> dict:
    """
    Stage 1 (Live Mode): Reuse canonical schema to prepare a live history window
    ending at current bin `t`, tagged with {source, fetched_at, is_stale}.
    """
    now_local = pd.Timestamp.now(tz=TIMEZONE).floor("15min")
    fetched_at = pd.Timestamp.now(tz=TIMEZONE).isoformat()

    seed_path = PROCESSED_DIR / "live_seed_10d.pkl.gz"
    canon_path = PROCESSED_DIR / "canonical_15min.parquet"
    if seed_path.exists():
        canon = pd.read_pickle(seed_path, compression="gzip")
    elif canon_path.exists():
        canon = pd.read_parquet(canon_path)
    else:
        raise FileNotFoundError(f"Archive table missing: {canon_path}. Run ingest.py --mode archive first.")
    max_ts = canon["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=lookback_days)
    recent = canon[canon["timestamp"] >= cutoff].copy()

    # Shift historical window to end at current bin `t` so lag_1..lag_672 align with live clock
    shift_delta = now_local - max_ts
    recent["timestamp"] = recent["timestamp"] + shift_delta
    traffic_source = "canonical_replay_stream"
    traffic_stale = False

    # Try fetching live NYC DOT speeds to update the latest bin (t - 15m) if available
    try:
        resp = requests.get(
            "https://data.cityofnewyork.us/resource/i4gi-tjb9.json",
            params={"$limit": 500, "$order": "data_as_of DESC"},
            timeout=8,
        )
        if resp.ok:
            rows = resp.json()
            if rows:
                live_df = pd.DataFrame(rows)
                if "link_id" in live_df.columns and "speed" in live_df.columns:
                    live_df["segment_id"] = pd.to_numeric(live_df["link_id"], errors="coerce")
                    live_df["speed_live"] = pd.to_numeric(live_df["speed"], errors="coerce")
                    live_df = live_df.dropna(subset=["segment_id", "speed_live"])
                    live_df = live_df[(live_df["speed_live"] > 0) & (live_df["speed_live"] <= 120)]
                    if len(live_df) > 0:
                        live_map = live_df.groupby("segment_id")["speed_live"].mean().to_dict()
                        prev_bin = now_local - pd.Timedelta(minutes=15)
                        mask_prev = recent["timestamp"] == prev_bin
                        matched = recent.loc[mask_prev, "segment_id"].map(live_map)
                        if matched.notna().sum() > 0:
                            recent.loc[mask_prev, "speed_mph"] = matched.fillna(
                                recent.loc[mask_prev, "speed_mph"]
                            ).astype("float32")
                            traffic_source = "nyc_dot_live_socrata+archive_history"
    except Exception:
        traffic_stale = True

    weather_meta = fetch_live_weather_gov()
    closures_meta = fetch_manual_closures(closures_override)

    # Inject live weather into the latest observed bin (< t)
    prev_bin = now_local - pd.Timedelta(minutes=15)
    mask_prev = recent["timestamp"] == prev_bin
    recent.loc[mask_prev, "temp_c"] = np.float32(weather_meta["temp_c"])
    recent.loc[mask_prev, "humidity_pct"] = np.float32(weather_meta["humidity_pct"])
    recent.loc[mask_prev, "precip_mm"] = np.float32(weather_meta["precip_mm"])
    recent.loc[mask_prev, "wind_kmh"] = np.float32(weather_meta["wind_kmh"])

    # Tag every row with {source, fetched_at}
    recent["source"] = traffic_source
    recent["fetched_at"] = fetched_at

    feeds = {
        "traffic": {"source": traffic_source, "fetched_at": fetched_at, "is_stale": traffic_stale},
        "weather": {
            "source": weather_meta["source"],
            "fetched_at": weather_meta["fetched_at"],
            "is_stale": weather_meta["is_stale"],
        },
        "closures": {
            "source": closures_meta["source"],
            "fetched_at": closures_meta["fetched_at"],
            "is_stale": closures_meta["is_stale"],
        },
    }
    return {
        "history_df": recent,
        "target_time": now_local,
        "closures": closures_meta["closures"],
        "weather": weather_meta,
        "feeds": feeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Ingest NYC archive or live data.")
    parser.add_argument("--mode", default="archive", choices=["archive", "live"])
    parser.add_argument("--n-months", type=int, default=6, help="Number of latest months from data.zip (0 = all).")
    parser.add_argument("--min-coverage", type=float, default=0.50, help="Min non-missing fraction per segment.")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "live":
        bundle = ingest_live_feeds()
        print(f"[ingest:live] Target bin t = {bundle['target_time']}")
        print(f"[ingest:live] Feeds = {bundle['feeds']}")
        return

    canonical = ingest_nyc_dot_speeds(DATA_ZIP, n_months=args.n_months, min_coverage=args.min_coverage)

    start_date = canonical["timestamp"].min().strftime("%Y-%m-%d")
    end_date = canonical["timestamp"].max().strftime("%Y-%m-%d")

    weather_df = ingest_noaa_weather(start_date, end_date)
    collisions_df = ingest_nypd_collisions(start_date, end_date)
    osm_meta = ingest_osm_metadata(canonical["segment_id"].unique())

    final_df = add_holidays_and_merge(canonical, weather_df, collisions_df, osm_meta)
    out_path = PROCESSED_DIR / "canonical_15min.parquet"
    final_df.to_parquet(out_path, index=False)
    print(f"[ingest] DONE -> {out_path} ({len(final_df):,} rows, {final_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
