"""
router.py — Stage 5b: Time-Dependent A* on Manhattan OSM graph with uncertainty penalty
and >10% live re-evaluation gate.

Edge cost:
  cost = P50_time + lambda_penalty * (P90_time - P50_time)
  Closed edges are removed from the graph before search.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
NODES_PKL_PATH = ROOT_DIR / "citypulse" / "data" / "processed" / "manhattan_nodes.pkl.gz"
EDGES_PKL_PATH = ROOT_DIR / "citypulse" / "data" / "processed" / "manhattan_edges.pkl.gz"
NODES_PATH = ROOT_DIR / "citypulse" / "data" / "processed" / "manhattan_nodes.parquet"
EDGES_PATH = ROOT_DIR / "citypulse" / "data" / "processed" / "manhattan_edges.parquet"

MPH_TO_MPS = 0.44704
DEFAULT_MAX_SPEED_MPS = 60.0 * MPH_TO_MPS  # admissible upper bound for A* heuristic
HORIZON_THRESHOLDS_S = ((900.0, "15"), (1800.0, "30"), (2700.0, "45"))

_GRAPH_CACHE: Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Tuple[float, float]]]] = None


def _get_graph_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Tuple[float, float]]]:
    """Load and cache Manhattan OSM nodes, edges, and coordinate lookup."""
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        if NODES_PKL_PATH.exists() and EDGES_PKL_PATH.exists():
            nodes = pd.read_pickle(NODES_PKL_PATH, compression="gzip")
            edges = pd.read_pickle(EDGES_PKL_PATH, compression="gzip")
        else:
            nodes = pd.read_parquet(NODES_PATH)
            edges = pd.read_parquet(EDGES_PATH)
        coords = {
            int(r.osmid): (float(r.y), float(r.x))
            for r in nodes.itertuples(index=False)
        }
        _GRAPH_CACHE = (nodes, edges, coords)
    return _GRAPH_CACHE


def _horizon_idx(elapsed_s: float) -> int:
    """Map elapsed trip time (seconds) to horizon index 0..3 (+15/+30/+45/+60m)."""
    if elapsed_s < 900.0:
        return 0
    if elapsed_s < 1800.0:
        return 1
    if elapsed_s < 2700.0:
        return 2
    return 3


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def should_reroute(current_cost_s: float, new_cost_s: float, min_improvement: float = 0.10) -> bool:
    """Switch on live trips only if the new route is >10% better."""
    if current_cost_s <= 0:
        return True
    improvement = (current_cost_s - new_cost_s) / current_cost_s
    return bool(improvement > min_improvement)


def _astar(
    adj: Dict[int, List[Tuple[int, float, float, float]]],
    coords: Dict[int, Tuple[float, float]],
    start: int,
    goal: int,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> Tuple[List[int], float, float, float]:
    """
    A* search minimizing edge cost (seconds).
    adj[u] = [(v, cost_s, base_time_s, length_m), ...]
    Returns (path_nodes, total_cost_s, total_time_s, total_dist_m).
    """
    g_lat, g_lon = coords[goal]

    def h(node: int) -> float:
        n_lat, n_lon = coords[node]
        return _haversine_m(n_lat, n_lon, g_lat, g_lon) / max_speed_mps

    pq: List[Tuple[float, float, int]] = [(h(start), 0.0, start)]
    g_cost: Dict[int, float] = {start: 0.0}
    parent: Dict[int, int] = {}
    edge_info: Dict[int, Tuple[float, float]] = {}

    while pq:
        _, cur_g, u = heapq.heappop(pq)
        if u == goal:
            path = []
            tot_time = 0.0
            tot_dist = 0.0
            curr = goal
            while curr in parent:
                path.append(curr)
                t_s, d_m = edge_info[curr]
                tot_time += t_s
                tot_dist += d_m
                curr = parent[curr]
            path.append(start)
            path.reverse()
            return path, cur_g, tot_time, tot_dist

        if cur_g > g_cost.get(u, float("inf")):
            continue

        for v, cost_s, time_s, dist_m in adj.get(u, ()):
            nxt_g = cur_g + cost_s
            if nxt_g < g_cost.get(v, float("inf")):
                g_cost[v] = nxt_g
                parent[v] = u
                edge_info[v] = (time_s, dist_m)
                heapq.heappush(pq, (nxt_g + h(v), nxt_g, v))

    return [], float("inf"), float("inf"), 0.0


def _td_astar(
    adj: Dict[int, List[Tuple[int, Tuple[float, float, float, float], Tuple[float, float, float, float], float]]],
    coords: Dict[int, Tuple[float, float]],
    start: int,
    goal: int,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> Tuple[List[int], float, float, float]:
    """
    Time-dependent A* search selecting edge cost/time by arrival horizon (+15/+30/+45/+60m).
    adj[u] = [(v, costs_by_h, times_by_h, length_m), ...]
    """
    g_lat, g_lon = coords[goal]

    def h(node: int) -> float:
        n_lat, n_lon = coords[node]
        return _haversine_m(n_lat, n_lon, g_lat, g_lon) / max_speed_mps

    pq: List[Tuple[float, float, int]] = [(h(start), 0.0, start)]
    g_cost: Dict[int, float] = {start: 0.0}
    parent: Dict[int, int] = {}
    edge_info: Dict[int, Tuple[float, float]] = {}

    while pq:
        _, cur_g, u = heapq.heappop(pq)
        if u == goal:
            path = []
            tot_time = 0.0
            tot_dist = 0.0
            curr = goal
            while curr in parent:
                path.append(curr)
                t_s, d_m = edge_info[curr]
                tot_time += t_s
                tot_dist += d_m
                curr = parent[curr]
            path.append(start)
            path.reverse()
            return path, cur_g, tot_time, tot_dist

        if cur_g > g_cost.get(u, float("inf")):
            continue

        h_idx = _horizon_idx(cur_g)
        for v, costs_by_h, times_by_h, dist_m in adj.get(u, ()):
            cost_s = costs_by_h[h_idx]
            time_s = times_by_h[h_idx]
            nxt_g = cur_g + cost_s
            if nxt_g < g_cost.get(v, float("inf")):
                g_cost[v] = nxt_g
                parent[v] = u
                edge_info[v] = (time_s, dist_m)
                heapq.heappush(pq, (nxt_g + h(v), nxt_g, v))

    return [], float("inf"), float("inf"), 0.0


def compute_routes(
    preds_df: pd.DataFrame,
    closures: List[dict],
    origin_node: int = 120,
    dest_node: int = 2150,
    lambda_penalty: float = 0.5,
    current_route_cost_s: Optional[float] = None,
) -> Dict[str, object]:
    """Compare Current-Traffic A* vs CITYPULSE Uncertainty-Aware Time-Dependent A*."""
    nodes, edges, coords = _get_graph_data()

    if origin_node not in coords:
        origin_node = int(nodes["osmid"].iloc[0])
    if dest_node not in coords:
        dest_node = int(nodes["osmid"].iloc[-1])

    closed_edge_indices = {int(c["edge_idx"]) for c in closures if "edge_idx" in c}
    closed_names = {str(c["name"]) for c in closures if "name" in c}

    n_segs = len(preds_df)
    p50_by_h = [
        preds_df[f"pred_speed_{h}"].to_numpy(dtype=float)
        for h in ("15", "30", "45", "60")
    ]
    p90_speeds_15 = preds_df["p90_speed_15"].to_numpy(dtype=float)
    cur_speeds = preds_df["speed_lag_1"].fillna(preds_df["hist_avg_mph"]).to_numpy(dtype=float)

    max_observed_mph = max(
        60.0,
        float(np.max(cur_speeds)) if n_segs else 60.0,
        max(float(np.max(arr)) for arr in p50_by_h) if n_segs else 60.0,
    )
    max_speed_mps = max_observed_mph * MPH_TO_MPS

    adj_current: Dict[int, List[Tuple[int, float, float, float]]] = {}
    adj_citypulse: Dict[
        int,
        List[Tuple[int, Tuple[float, float, float, float], Tuple[float, float, float, float], float]],
    ] = {}

    for idx, row in enumerate(edges.itertuples(index=False)):
        if idx in closed_edge_indices or str(row.name) in closed_names:
            continue  # Remove closed edges

        u, v = int(row.u), int(row.v)
        length_m = float(row.length)
        seg_i = idx % n_segs

        v_cur = max(cur_speeds[seg_i], 2.0) * MPH_TO_MPS
        t_cur = length_m / v_cur

        spread_ratio = max(0.0, p90_speeds_15[seg_i] - p50_by_h[0][seg_i]) / max(p50_by_h[0][seg_i], 2.0)
        times_h = tuple(length_m / (max(p50_by_h[hi][seg_i], 2.0) * MPH_TO_MPS) for hi in range(4))
        costs_h = tuple(t_h * (1.0 + lambda_penalty * spread_ratio) for t_h in times_h)

        adj_current.setdefault(u, []).append((v, t_cur, t_cur, length_m))
        adj_citypulse.setdefault(u, []).append((v, costs_h, times_h, length_m))

    path_cur, cost_cur, time_cur, dist_cur = _astar(adj_current, coords, origin_node, dest_node, max_speed_mps)
    path_cp, cost_cp, time_cp, dist_cp = _td_astar(adj_citypulse, coords, origin_node, dest_node, max_speed_mps)

    ref_cost = current_route_cost_s if current_route_cost_s is not None else cost_cur
    reroute_triggered = should_reroute(ref_cost, cost_cp, min_improvement=0.10)
    pct_diff = ((ref_cost - cost_cp) / ref_cost * 100.0) if ref_cost > 0 and math.isfinite(ref_cost) else 0.0

    return {
        "origin_node": origin_node,
        "dest_node": dest_node,
        "lambda_penalty": lambda_penalty,
        "closed_edges_removed": len(closed_edge_indices),
        "reroute_triggered": reroute_triggered,
        "pct_improvement": round(pct_diff, 2),
        "current_route": {
            "time_min": round(time_cur / 60.0, 2) if math.isfinite(time_cur) else None,
            "cost_s": round(cost_cur, 1) if math.isfinite(cost_cur) else None,
            "distance_km": round(dist_cur / 1000.0, 2),
            "coords": [list(coords[n]) for n in path_cur],
        },
        "citypulse_route": {
            "time_min": round(time_cp / 60.0, 2) if math.isfinite(time_cp) else None,
            "cost_s": round(cost_cp, 1) if math.isfinite(cost_cp) else None,
            "distance_km": round(dist_cp / 1000.0, 2),
            "coords": [list(coords[n]) for n in path_cp],
        },
    }
