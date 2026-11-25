"""
main.py — Stage 6b: Live orchestration pipeline (Stages 1–6) -> outputs/live_snapshot.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from crisis_rules import evaluate_crisis_rules
from explainer import explain_high_risk_segments
from features import build_live_features
from ingest import ingest_live_feeds
from monitor import log_and_evaluate
from predictor import predict_live
from router import _get_graph_data, compute_routes

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = Path("/tmp/outputs") if os.environ.get("VERCEL") else ROOT_DIR / "outputs"


def _attach_segment_coords(preds_df: pd.DataFrame) -> List[dict]:
    """Attach representative Manhattan lat/lon coordinates to each segment for map rendering."""
    _, edges, coords = _get_graph_data()
    n_edges = len(edges)

    records = []
    for idx, r in enumerate(preds_df.itertuples(index=False)):
        edge = edges.iloc[(int(r.segment_id) + idx) % n_edges]
        u_lat, u_lon = coords[int(edge.u)]
        v_lat, v_lon = coords[int(edge.v)]
        records.append(
            {
                "segment_id": int(r.segment_id),
                "name": str(edge.name),
                "u_lat": float(u_lat),
                "u_lon": float(u_lon),
                "v_lat": float(v_lat),
                "v_lon": float(v_lon),
                "pred_speed_15": float(r.pred_speed_15),
                "pred_speed_30": float(r.pred_speed_30),
                "pred_speed_45": float(r.pred_speed_45),
                "pred_speed_60": float(r.pred_speed_60),
                "speed_ratio_15": float(r.speed_ratio_15),
                "speed_ratio_30": float(r.speed_ratio_30),
                "speed_ratio_45": float(r.speed_ratio_45),
                "speed_ratio_60": float(r.speed_ratio_60),
                "congestion_prob_15": float(r.congestion_prob_15),
                "congestion_prob_30": float(r.congestion_prob_30),
                "congestion_prob_45": float(r.congestion_prob_45),
                "congestion_prob_60": float(r.congestion_prob_60),
                "p90_speed_15": float(r.p90_speed_15),
                "is_stale": bool(r.is_stale),
                "fallback_used": bool(r.fallback_used),
            }
        )
    return records


def run_cycle(
    closures_override: Optional[List[dict]] = None,
    origin_node: int = 120,
    dest_node: int = 2150,
    lambda_penalty: float = 0.5,
    current_route_cost_s: Optional[float] = None,
) -> dict:
    """Execute Stages 1–6 and persist outputs/live_snapshot.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Live Feeds
    live_bundle = ingest_live_feeds(closures_override=closures_override)
    history_df = live_bundle["history_df"]
    target_time = live_bundle["target_time"]
    closures = live_bundle["closures"]
    weather = live_bundle["weather"]
    feeds = live_bundle["feeds"]

    # 2. Build Features (strictly < t)
    live_feat, _ = build_live_features(history_df, target_time)

    # 3. Predict Speeds & Calibrated Probabilities (+ Fallback)
    preds_df = predict_live(live_feat, feeds)

    # 4. Crisis Rules
    crisis = evaluate_crisis_rules(preds_df, weather, closures)

    # 5. Explainability (SHAP) + Time-Dependent A* Routing
    shap_drivers = explain_high_risk_segments(live_feat, preds_df, top_k_segments=5, top_k_features=5)
    routes = compute_routes(
        preds_df,
        closures=closures,
        origin_node=origin_node,
        dest_node=dest_node,
        lambda_penalty=lambda_penalty,
        current_route_cost_s=current_route_cost_s,
    )

    # 6. Persist & Monitor
    monitor_stats = log_and_evaluate(preds_df, history_df, target_time)
    segments = _attach_segment_coords(preds_df)

    snapshot = {
        "target_time": target_time.isoformat(),
        "feeds": feeds,
        "weather": weather,
        "closures": closures,
        "crisis": crisis,
        "shap_drivers": shap_drivers,
        "routes": routes,
        "monitor": monitor_stats,
        "segments": segments,
    }

    import math

    def _clean_json(obj):
        if isinstance(obj, dict):
            return {k: _clean_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_json(v) for v in obj]
        if hasattr(obj, "item"):
            obj = obj.item()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    snap_path = OUTPUTS_DIR / "live_snapshot.json"
    cleaned_snapshot = _clean_json(snapshot)
    snap_path.write_text(
        json.dumps(cleaned_snapshot, indent=2),
        encoding="utf-8",
    )
    print(
        f"[main] Snapshot saved -> {snap_path} | "
        f"Index={crisis['congestion_index']}/100 ({crisis['crisis_type']}) | "
        f"Routes: Current={routes['current_route']['time_min']}m vs "
        f"CITYPULSE={routes['citypulse_route']['time_min']}m"
    )
    return cleaned_snapshot


if __name__ == "__main__":
    run_cycle()
