"""
crisis_rules.py — Stage 4: Aggregate segment predictions into citywide Congestion Index,
Crisis Type, and Crisis Window.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

CONGESTED_RATIO_THRESH = 0.60
CONGESTED_PROB_THRESH = 0.50


def evaluate_crisis_rules(
    preds_df: pd.DataFrame,
    weather: dict,
    closures: list[dict],
) -> Dict[str, object]:
    """Compute Congestion Index (0-100), Crisis Type, and Peak Window."""
    horizon_stats = {}
    for h in ["15", "30", "45", "60"]:
        ratio = preds_df[f"speed_ratio_{h}"].to_numpy()
        prob = preds_df[f"congestion_prob_{h}"].to_numpy()
        congested = (ratio < CONGESTED_RATIO_THRESH) | (prob >= CONGESTED_PROB_THRESH)
        share = float(np.mean(congested))
        severity = float(np.mean(np.clip(1.0 - ratio, 0.0, 1.0)))
        horizon_stats[h] = {
            "congested_share": round(share, 4),
            "mean_severity": round(severity, 4),
            "congested_count": int(congested.sum()),
        }

    peak_h = max(horizon_stats.keys(), key=lambda k: horizon_stats[k]["congested_share"])
    peak_share = horizon_stats[peak_h]["congested_share"]
    mean_share = float(np.mean([v["congested_share"] for v in horizon_stats.values()]))
    mean_sev = float(np.mean([v["mean_severity"] for v in horizon_stats.values()]))

    # 0-100 Congestion Index
    congestion_index = int(np.clip(round((0.65 * peak_share + 0.35 * mean_sev) * 140), 0, 100))

    precip = float(weather.get("precip_mm", 0.0) or 0.0)
    n_closures = len(closures)

    if congestion_index >= 70 or peak_share >= 0.45:
        crisis_type = "Citywide Gridlock"
    elif (precip >= 2.0 or n_closures >= 2) and congestion_index >= 40:
        crisis_type = "Weather/Incident Surge"
    elif congestion_index >= 45 or peak_share >= 0.25:
        crisis_type = "Corridor Slowdown"
    elif congestion_index >= 25 or peak_share >= 0.12:
        crisis_type = "Localized Bottleneck"
    else:
        crisis_type = "Normal"

    # Determine crisis window
    active_horizons = [int(h) for h, s in horizon_stats.items() if s["congested_share"] >= 0.15]
    if active_horizons:
        crisis_window = f"t+{min(active_horizons)}m to t+{max(active_horizons)}m"
    else:
        crisis_window = f"Peak at t+{peak_h}m (Normal)"

    return {
        "congestion_index": congestion_index,
        "crisis_type": crisis_type,
        "crisis_window": crisis_window,
        "peak_horizon_min": int(peak_h),
        "peak_congested_share": round(peak_share, 4),
        "mean_congested_share": round(mean_share, 4),
        "total_segments": int(len(preds_df)),
        "active_closures": n_closures,
        "precip_mm": precip,
        "horizon_breakdown": horizon_stats,
    }
