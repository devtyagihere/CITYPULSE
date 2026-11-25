# =============================================================================
# STEP NAME  : step23 — FastAPI Backend Server (Single A* Hospital Dispatch)
# WHAT       : Exposes CITYPULSE predictions, crisis status, Manhattan
#              hospitals, and A* ambulance routing (including automatic
#              nearest-hospital selection from any clicked map point).
# WHY        : Powers the single-map A* ambulance dispatch dashboard.
# INPUT      : data/processed/live_snapshot.json
#              data/processed/manhattan_graph.graphml
#              data/processed/sensor_to_edge.csv
# OUTPUT     : Runs FastAPI server on http://127.0.0.1:8000
# =============================================================================

import sys, os, json, heapq, math
from datetime import datetime, timezone
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "steps"))

import pandas as pd
import networkx as nx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from config import DATA_PROCESSED_DIR, ROUTE_UNCERTAINTY_LAMBDA
from step19_dijkstra_from_scratch import dijkstra_heapq

app = FastAPI(title="CITYPULSE API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_PATH    = os.path.join(DATA_PROCESSED_DIR, "manhattan_graph.graphml")
SNAPSHOT_PATH = os.path.join(DATA_PROCESSED_DIR, "live_snapshot.json")
EDGE_CSV_PATH = os.path.join(DATA_PROCESSED_DIR, "sensor_to_edge.csv")

LANDMARKS = {
    "Battery Park":       (40.7033, -74.0170),
    "Wall Street":        (40.7074, -74.0113),
    "SoHo":               (40.7233, -74.0030),
    "Union Square":       (40.7359, -73.9911),
    "Times Square":       (40.7580, -73.9855),
    "Central Park South": (40.7678, -73.9815),
    "Harlem":             (40.8116, -73.9465),
    "Washington Heights": (40.8417, -73.9393),
}

HOSPITALS = {
    "NYP Lower Manhattan Hospital": (40.7104, -74.0052),
    "Bellevue Hospital":            (40.7392, -73.9754),
    "NYU Langone Health":           (40.7421, -73.9740),
    "NYP Weill Cornell":            (40.7644, -73.9541),
    "Mount Sinai West":             (40.7697, -73.9873),
    "Lenox Hill Hospital":          (40.7736, -73.9609),
    "Mount Sinai Hospital":         (40.7903, -73.9524),
    "Harlem Hospital":              (40.8142, -73.9403),
}

_G = None
_SENSOR_EDGE = None
_LANDMARK_NODES = None
_HOSPITAL_NODES = None


def nearest_node_in_graph(G, lat, lon):
    return int(min(
        G.nodes(),
        key=lambda n: (G.nodes[n]["y"] - lat) ** 2 + (G.nodes[n]["x"] - lon) ** 2,
    ))


def get_graph_and_edges():
    global _G, _SENSOR_EDGE, _LANDMARK_NODES, _HOSPITAL_NODES
    if _G is None:
        _G = nx.read_graphml(GRAPH_PATH, node_type=int, force_multigraph=True)
        for _, _, _, d in _G.edges(keys=True, data=True):
            d["base_tt"] = float(d.get("travel_time", 20.0))
        _SENSOR_EDGE = pd.read_csv(EDGE_CSV_PATH)
        _LANDMARK_NODES = {
            name: {
                "node": (n := nearest_node_in_graph(_G, lat, lon)),
                "lat": float(_G.nodes[n]["y"]),
                "lon": float(_G.nodes[n]["x"]),
            }
            for name, (lat, lon) in LANDMARKS.items()
        }
        _HOSPITAL_NODES = {
            name: {
                "node": (n := nearest_node_in_graph(_G, lat, lon)),
                "lat": float(_G.nodes[n]["y"]),
                "lon": float(_G.nodes[n]["x"]),
            }
            for name, (lat, lon) in HOSPITALS.items()
        }
    return _G, _SENSOR_EDGE, _LANDMARK_NODES, _HOSPITAL_NODES


def load_snapshot():
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_weights_from_snapshot(G, sensor_edge, snap, horizon=15):
    sr_map = {s["link_id"]: s for s in snap["sensors"]}
    edge_lookup = {
        (int(r["edge_u"]), int(r["edge_v"])): sr_map[r["link_id"]]
        for _, r in sensor_edge.iterrows() if r["link_id"] in sr_map
    }
    pred_key = f"pred_{horizon}m"
    # Estimate horizon-level city average from sensors
    city_p50 = sum(float(s.get(pred_key, s["pred_15m"])) for s in snap["sensors"]) / max(1, len(snap["sensors"]))
    for u, v, _, d in G.edges(keys=True, data=True):
        base = d["base_tt"]
        info = edge_lookup.get((u, v))
        p50 = max(float(info.get(pred_key, info["pred_15m"])) if info else city_p50, 0.15)
        p10 = max(float(info["p10_15m"]) if info else max(city_p50 - 0.1, 0.15), 0.15)
        d["pred_time"] = base / p50
        d["risk_cost"] = (base / p50) + ROUTE_UNCERTAINTY_LAMBDA * max(0.0, (base / p10) - (base / p50))


def haversine_seconds(G, u, v, max_speed_mps=15.0):
    lat1, lon1 = math.radians(G.nodes[u]["y"]), math.radians(G.nodes[u]["x"])
    lat2, lon2 = math.radians(G.nodes[v]["y"]), math.radians(G.nodes[v]["x"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    dist_m = 6371000.0 * 2 * math.asin(min(1.0, math.sqrt(a)))
    return dist_m / max_speed_mps


def astar_with_trace(G, start, end, weight_attr="risk_cost"):
    """Run A* (g(n) + h(n)) and return explored edges + final path."""
    heap = [(0.0, 0.0, start)]
    g_score = {start: 0.0}
    parent = {start: None}
    visited = set()
    explored_edges = []

    while heap:
        _, g, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if parent[u] is not None:
            p = parent[u]
            explored_edges.append([
                [float(G.nodes[p]["y"]), float(G.nodes[p]["x"])],
                [float(G.nodes[u]["y"]), float(G.nodes[u]["x"])],
            ])
        if u == end:
            break

        for v, edge_dict in G[u].items():
            w = min(float(attr.get(weight_attr, 1.0)) for attr in edge_dict.values())
            new_g = g + w
            if new_g < g_score.get(v, float("inf")):
                g_score[v] = new_g
                parent[v] = u
                h = haversine_seconds(G, v, end)
                heapq.heappush(heap, (new_g + h, new_g, v))

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()

    path_coords = [[float(G.nodes[n]["y"]), float(G.nodes[n]["x"])] for n in path]
    eta_min = sum(G[path[i]][path[i + 1]][0]["pred_time"] for i in range(len(path) - 1)) / 60.0
    dist_km = sum(float(G[path[i]][path[i + 1]][0].get("length", 250.0)) for i in range(len(path) - 1)) / 1000.0
    total_nodes = G.number_of_nodes()

    return {
        "start_node": start,
        "end_node": end,
        "explored_count": len(visited),
        "pct_city": round(100.0 * len(visited) / total_nodes, 1),
        "eta_min": round(eta_min, 2),
        "dist_km": round(dist_km, 2),
        "explored_edges": explored_edges,
        "path_coords": path_coords,
    }


@app.get("/")
def serve_landing():
    return FileResponse(os.path.join(PROJECT_ROOT, "steps", "landing.html"))


@app.get("/manhattan_bg.png")
def serve_manhattan_bg():
    return FileResponse(os.path.join(PROJECT_ROOT, "reports", "manhattan_bg.png"))


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(PROJECT_ROOT, "steps", "step24_dashboard.html"))


@app.get("/teacher")
def serve_teacher():
    return FileResponse(os.path.join(PROJECT_ROOT, "steps", "teacher_presentation.html"))


@app.get("/reports_img/{filename}")
def serve_report_img(filename: str):
    return FileResponse(os.path.join(PROJECT_ROOT, "reports", filename))


@app.get("/open_file")
def open_file_on_laptop(name: str):
    """Open a CITYPULSE project file locally on Windows using os.startfile."""
    safe_name = os.path.basename(name)
    for folder in ["steps", "src", "reports"]:
        candidate = os.path.join(PROJECT_ROOT, folder, safe_name)
        if os.path.isfile(candidate):
            os.startfile(candidate)
            return {"status": "opened", "path": candidate}
    return {"status": "not_found", "name": safe_name}


@app.get("/health")
def health():
    snap = load_snapshot()
    return {
        "status": "ok",
        "source": snap["source"],
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "sensors_loaded": len(snap["sensors"]),
    }


@app.get("/locations")
def locations():
    _, _, lm, hosp = get_graph_and_edges()
    return {"landmarks": lm, "hospitals": hosp}


@app.get("/predict")
def predict(horizon: int = Query(15, enum=[15, 30, 45, 60])):
    snap = load_snapshot()
    key = f"pred_{horizon}m"
    sensors = [
        {
            "link_id": s["link_id"],
            "road_name": s["road_name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "current_sr": s["current_sr"],
            "predicted_sr": s[key],
        }
        for s in snap["sensors"]
    ]
    h_cong = snap["city_metrics"].get("horizon_congestion", {}).get(
        f"+{horizon}m", snap["city_metrics"]["congestion_index"]
    )
    return {
        "source": snap["source"],
        "timestamp": snap["generated_at"],
        "horizon_min": horizon,
        "congestion_index": h_cong,
        "pm25_pred_1h": snap["city_metrics"]["pm25_pred_1h"],
        "aqi_category": snap["city_metrics"]["aqi_category"],
        "sensors": sensors,
    }


@app.get("/crisis")
def crisis():
    snap = load_snapshot()
    return {
        "source": snap["source"],
        "timestamp": snap["generated_at"],
        "weather": snap["weather"],
        **snap["city_metrics"],
    }


@app.get("/route_astar")
def route_astar(
    start_lat: float = Query(None),
    start_lon: float = Query(None),
    start_node: int = Query(None),
    end_node: int = Query(None),
    horizon: int = Query(15, enum=[15, 30, 45, 60]),
):
    """
    Compute A* route from Point A (clicked lat/lon or start_node)
    to Point B (selected hospital end_node, or automatically the NEAREST hospital)
    using the selected forecast horizon (+15m, +30m, +45m, +60m).
    """
    snap = load_snapshot()
    G, sensor_edge, _, hosp = get_graph_and_edges()
    apply_weights_from_snapshot(G, sensor_edge, snap, horizon=horizon)

    if start_lat is not None and start_lon is not None:
        s_node = nearest_node_in_graph(G, start_lat, start_lon)
    elif start_node is not None:
        s_node = start_node
    else:
        s_node = 100

    # If end_node is omitted, pick the nearest Manhattan hospital by A* ETA
    chosen_hospital = "Custom Destination"
    if end_node is None:
        best_res = None
        for h_name, h_info in hosp.items():
            res = astar_with_trace(G, s_node, h_info["node"], weight_attr="risk_cost")
            if best_res is None or res["eta_min"] < best_res["eta_min"]:
                best_res = res
                chosen_hospital = h_name
        result = best_res
    else:
        for h_name, h_info in hosp.items():
            if h_info["node"] == end_node:
                chosen_hospital = h_name
                break
        result = astar_with_trace(G, s_node, end_node, weight_attr="risk_cost")

    # Snap the A* route onto real Manhattan street geometry via OSRM (free, no key)
    # Using start, midpoint from A*, and destination so it follows real curved roads
    pts = result["path_coords"]
    if len(pts) >= 2:
        s_pt = (start_lat, start_lon) if (start_lat is not None and start_lon is not None) else tuple(pts[0])
        e_pt = tuple(pts[-1])
        try:
            osrm_url = (
                f"https://router.project-osrm.org/route/v1/driving/"
                f"{s_pt[1]},{s_pt[0]};{e_pt[1]},{e_pt[0]}"
                f"?overview=full&geometries=geojson"
            )
            r = requests.get(osrm_url, headers={"User-Agent": "CITYPULSE/1.0"}, timeout=4)
            if r.status_code == 200:
                routes = r.json().get("routes", [])
                if routes:
                    coords_lonlat = routes[0]["geometry"]["coordinates"]
                    result["path_coords"] = [[c[1], c[0]] for c in coords_lonlat]
                    result["dist_km"] = round(routes[0]["distance"] / 1000.0, 2)
        except Exception:
            pass

    return {
        "source": snap["source"],
        "timestamp": snap["generated_at"],
        "hospital_name": chosen_hospital,
        **result,
    }


@app.get("/route")
def route(start_node: int = 100, end_node: int = 2250):
    snap = load_snapshot()
    G, sensor_edge, _, _ = get_graph_and_edges()
    apply_weights_from_snapshot(G, sensor_edge, snap)
    _, path = dijkstra_heapq(G, start_node, end_node, weight_attr="risk_cost")
    coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in path]
    total_min = sum(G[path[i]][path[i + 1]][0]["pred_time"] for i in range(len(path) - 1)) / 60.0
    return {
        "source": snap["source"],
        "timestamp": snap["generated_at"],
        "start_node": start_node,
        "end_node": end_node,
        "predicted_minutes": round(total_min, 2),
        "num_intersections": len(path),
        "route_coords": coords,
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting CITYPULSE FastAPI server at http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
