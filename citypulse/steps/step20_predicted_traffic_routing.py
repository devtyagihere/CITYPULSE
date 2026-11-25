# =============================================================================
# STEP NAME  : step20 — Predicted Traffic & Uncertainty-Aware Routing
# WHAT       : Applies predicted +15m travel times and uncertainty penalties
#              (P90_time - P50_time) onto the Manhattan graph, then computes
#              3 alternative ambulance routes + an optional cleanest-air route.
# WHY        : An ambulance should avoid roads that are either predicted to be
#              slow OR highly unpredictable (high uncertainty).
# INPUT      : data/processed/manhattan_graph.graphml
#              data/processed/sensor_to_edge.csv
#              data/processed/test_quantiles_15m.parquet
#              models/aqi_lgb_1h.joblib
# OUTPUT     : reports/step20_alternative_routes.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "steps"))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (DATA_PROCESSED_DIR, REPORTS_DIR,
                    ROUTE_UNCERTAINTY_LAMBDA, RANDOM_SEED, USE_SYNTHETIC)
from step19_dijkstra_from_scratch import dijkstra_heapq

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 20 — Predicted Traffic & Uncertainty-Aware Routing")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)


def apply_traffic_to_graph(G, sensor_edge_df, time_slice_df,
                           lam=ROUTE_UNCERTAINTY_LAMBDA, pm25_pred=15.0):
    """
    Update every edge in G with:
      - 'pred_time'  : predicted travel time (s) using P50 speed_ratio
      - 'risk_cost'  : pred_time + lam * (P90_time - P50_time)
      - 'clean_cost' : risk_cost + small penalty proportional to PM2.5
    """
    # Map (u, v) -> (p50, p10) for edges that have sensors
    edge_lookup = {}
    sr_map = time_slice_df.set_index("link_id")[["p50", "p10"]].to_dict("index")
    for _, row in sensor_edge_df.iterrows():
        lid = row["link_id"]
        if lid in sr_map:
            edge_lookup[(int(row["edge_u"]), int(row["edge_v"]))] = sr_map[lid]

    city_p50 = float(time_slice_df["p50"].mean())
    city_p10 = float(time_slice_df["p10"].mean())

    for u, v, k, d in G.edges(keys=True, data=True):
        base_tt = float(d.get("base_travel_time", d.get("travel_time", 20.0)))
        d["base_travel_time"] = base_tt

        vals = edge_lookup.get((u, v), {"p50": city_p50, "p10": city_p10})
        p50 = max(float(vals["p50"]), 0.15)
        p10 = max(float(vals["p10"]), 0.15)

        pred_tt = base_tt / p50
        p90_tt  = base_tt / p10   # slower speed (P10) = longer travel time (P90)
        unc_pen = max(0.0, p90_tt - pred_tt)

        d["pred_time"]  = pred_tt
        d["risk_cost"]  = pred_tt + lam * unc_pen
        # Cleanest-air mode: penalize avenue/major corridors slightly when PM2.5 is elevated
        air_factor = 1.0 + (0.15 * (pm25_pred / 35.0) if d.get("highway") == "secondary" else 0.0)
        d["clean_cost"] = d["risk_cost"] * air_factor


def find_k_alternative_routes(G, start, end, weight_attr="risk_cost", k=3):
    """Find k diverse routes by softly penalizing edges used in earlier routes."""
    routes = []
    # Save original weights so we can restore them at the end
    orig_w = {(u, v, key): d[weight_attr] for u, v, key, d in G.edges(keys=True, data=True)}
    tmp_attr = "_tmp_route_cost"
    for u, v, key, d in G.edges(keys=True, data=True):
        d[tmp_attr] = d[weight_attr]

    for _ in range(k):
        _, path = dijkstra_heapq(G, start, end, weight_attr=tmp_attr)
        if not path:
            break
        # Compute true unpenalized cost and travel time
        true_cost = sum(orig_w[(path[i], path[i+1], 0)] for i in range(len(path) - 1))
        pred_time = sum(G[path[i]][path[i+1]][0]["pred_time"] for i in range(len(path) - 1))
        routes.append((true_cost, pred_time, path))
        # Penalize used edges by 1.4x so next route explores alternatives
        for i in range(len(path) - 1):
            G[path[i]][path[i+1]][0][tmp_attr] *= 1.4

    return routes


# ===========================================================================
# SECTION A — Run demo routing on one rush-hour test timestamp
# ===========================================================================
G = nx.read_graphml(os.path.join(DATA_PROCESSED_DIR, "manhattan_graph.graphml"),
                    node_type=int, force_multigraph=True)
sensor_edge_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "sensor_to_edge.csv"))
test_q         = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "test_quantiles_15m.parquet"))

# Pick a rush-hour timestamp in test (e.g. 17:00)
rush_times = test_q[test_q["time"].dt.hour == 17]["time"].unique()
sample_t   = rush_times[0] if len(rush_times) > 0 else test_q["time"].iloc[0]
slice_df   = test_q[test_q["time"] == sample_t]

apply_traffic_to_graph(G, sensor_edge_df, slice_df, lam=ROUTE_UNCERTAINTY_LAMBDA, pm25_pred=24.0)

# Pick a realistic Lower-to-Upper Manhattan trip
all_nodes = sorted(G.nodes())
src_node  = all_nodes[100]
dst_node  = all_nodes[-150]

alt_routes = find_k_alternative_routes(G, src_node, dst_node, weight_attr="risk_cost", k=3)
clean_cost, clean_path = dijkstra_heapq(G, src_node, dst_node, weight_attr="clean_cost")
clean_time = sum(G[clean_path[i]][clean_path[i+1]][0]["pred_time"] for i in range(len(clean_path) - 1))

print(f"\n[A] Routing at {sample_t} (from node {src_node} to {dst_node}):")
for i, (rcost, ptime, path) in enumerate(alt_routes, start=1):
    print(f"  Route {i}: pred_time = {ptime/60:.2f} min  |  risk_cost = {rcost/60:.2f} min  |  steps = {len(path)}")
print(f"  Cleanest-Air Route: pred_time = {clean_time/60:.2f} min  |  steps = {len(clean_path)}")

# Plot 3 alternative routes on Manhattan grid
fig, ax = plt.subplots(figsize=(8, 13))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

for u, v, _ in list(G.edges(data=True))[::5]:
    ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
            [G.nodes[u]["y"], G.nodes[v]["y"]],
            color="dimgray", linewidth=0.3, alpha=0.4)

colors = ["lime", "cyan", "orange"]
for idx, (_, ptime, path) in enumerate(alt_routes):
    xs = [G.nodes[n]["x"] for n in path]
    ys = [G.nodes[n]["y"] for n in path]
    ax.plot(xs, ys, color=colors[idx], linewidth=2.2 - idx * 0.4,
            label=f"Alt Route {idx+1} ({ptime/60:.1f} min)")

ax.scatter([G.nodes[src_node]["x"]], [G.nodes[src_node]["y"]], c="white", s=80, zorder=10, label="Start")
ax.scatter([G.nodes[dst_node]["x"]], [G.nodes[dst_node]["y"]], c="red",   s=80, zorder=10, label="Hospital")
ax.set_title("Step 20 — 3 Alternative Uncertainty-Aware Ambulance Routes", color="white", fontsize=11)
ax.legend(facecolor="black", labelcolor="white", loc="lower right")
ax.tick_params(colors="white")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "step20_alternative_routes.png"), dpi=120, bbox_inches="tight", facecolor="black")
plt.close()

summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 20: Predicted Traffic & Uncertainty-Aware Routing\n\n")
    f.write("**What we did:** Applied predicted +15m travel times and uncertainty penalties "
            "(P90 - P50) to Manhattan edges, generating 3 alternative routes plus a cleanest-air route.\n\n")
    f.write(f"**What we got:** Best route predicted time = **{alt_routes[0][1]/60:.2f} min** "
            f"(alternative 2 = {alt_routes[1][1]/60:.2f} min).\n\n")
    f.write(f"**Key number:** Best predicted ambulance route = **{alt_routes[0][1]/60:.2f} minutes**.\n\n")
    f.write("**Plot:** `reports/step20_alternative_routes.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 20 DONE.  Route 1 = {alt_routes[0][1]/60:.2f} min")
print("=" * 65)
