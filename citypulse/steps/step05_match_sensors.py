# =============================================================================
# STEP NAME  : step05 — Match Sensors to Road Edges
# WHAT       : Links each of our 50 traffic sensors to the nearest real road
#              edge on the Manhattan graph. For synthetic data, we randomly
#              pick 50 real road edges and assign sensor IDs to them.
# WHY        : ML models predict speed per sensor. The router works with roads.
#              We must know which road each sensor sits on so we can feed
#              predictions into the routing graph.
# INPUT      : data/processed/manhattan_graph.graphml
#              data/processed/manhattan_edges.parquet
#              data/processed/traffic_clean.parquet
# OUTPUT     : data/processed/sensor_to_edge.csv   (sensor_id → road edge)
#              reports/step05_sensor_map.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from config import (USE_SYNTHETIC, DATA_PROCESSED_DIR, REPORTS_DIR,
                    RANDOM_SEED, SYNTH_N_SENSORS)

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 05 — Match Sensors to Road Edges")
print("=" * 65)

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Load the graph and edge table
# ===========================================================================
print("\n[A] Loading Manhattan road graph ...")

graph_path = os.path.join(DATA_PROCESSED_DIR, "manhattan_graph.graphml")
edges_path = os.path.join(DATA_PROCESSED_DIR, "manhattan_edges.parquet")

G         = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
edges_df  = pd.read_parquet(edges_path)         # table version of edges

print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  Edges table: {len(edges_df):,} rows")

# ===========================================================================
# SECTION B — Load sensor list from cleaned traffic data
# ===========================================================================
print("\n[B] Loading sensor list ...")

traffic_path = os.path.join(DATA_PROCESSED_DIR, "traffic_clean.parquet")
traffic_df   = pd.read_parquet(traffic_path)
sensor_ids   = sorted(traffic_df["link_id"].unique())
print(f"  Sensors to match: {len(sensor_ids)}")

# ===========================================================================
# SECTION C — Place sensors on road edges
# For SYNTHETIC data: randomly assign each sensor to a real road edge.
# For REAL data: we would use nearest-edge matching by GPS coordinates.
# ===========================================================================
print("\n[C] Matching sensors to road edges ...")

if USE_SYNTHETIC:
    print("  Mode: SYNTHETIC → randomly picking real edges for each sensor")

    # Get all edges as a list of (u, v, key) tuples
    all_edge_tuples = list(G.edges(keys=True))

    # Pick SYNTH_N_SENSORS edges randomly (without replacement)
    chosen_indices = np.random.choice(
        len(all_edge_tuples),
        size=min(SYNTH_N_SENSORS, len(all_edge_tuples)),
        replace=False
    )
    chosen_edges = [all_edge_tuples[i] for i in chosen_indices]

    # Build the sensor-to-edge mapping table
    rows = []
    for sensor_id, (u, v, key) in zip(sensor_ids, chosen_edges):
        edge_data = G[u][v][key]   # get attributes of this edge

        # Get midpoint coordinates for the sensor's map location
        # We average the two endpoint node positions
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        mid_lon = (u_data["x"] + v_data["x"]) / 2
        mid_lat = (u_data["y"] + v_data["y"]) / 2

        rows.append({
            "link_id"       : sensor_id,
            "edge_u"        : u,           # start node of road edge
            "edge_v"        : v,           # end node of road edge
            "edge_key"      : key,
            "road_name"     : edge_data.get("name", "unknown"),
            "road_type"     : edge_data.get("highway", "unknown"),
            "length_m"      : edge_data.get("length", np.nan),
            "speed_kph"     : edge_data.get("speed_kph", np.nan),
            "travel_time_s" : edge_data.get("travel_time", np.nan),
            "sensor_lon"    : mid_lon,
            "sensor_lat"    : mid_lat,
        })

    sensor_edge_df = pd.DataFrame(rows)

else:
    # Real data: sensors have GPS coords from link_points column
    # We use osmnx.nearest_edges to snap each GPS point to its closest road
    # (Uncomment and fill in real lat/lon from your traffic CSV)
    raise NotImplementedError(
        "For real data: extract lat/lon from link_points column,\n"
        "then call: ox.nearest_edges(G, lons, lats, return_dist=True)"
    )

print(f"  Matched {len(sensor_edge_df)} sensors to real edges")
print(f"\n  Sample of sensor-edge mapping:")
print(sensor_edge_df[["link_id", "road_name", "road_type",
                       "length_m", "speed_kph"]].head(8).to_string(index=False))

# ===========================================================================
# SECTION D — Save the mapping table
# ===========================================================================
print("\n[D] Saving sensor-to-edge mapping ...")

mapping_path = os.path.join(DATA_PROCESSED_DIR, "sensor_to_edge.csv")
sensor_edge_df.to_csv(mapping_path, index=False)
print(f"  Saved: {mapping_path}")

# Count road types
road_type_counts = sensor_edge_df["road_type"].apply(
    lambda x: x[0] if isinstance(x, list) else str(x)
).value_counts()
print(f"\n  Road types our sensors are on:")
print(road_type_counts.to_string())

# ===========================================================================
# SECTION E — Plot sensors on the road map
# ===========================================================================
print("\n[E] Plotting sensors on the Manhattan road map ...")

fig, ax = plt.subplots(1, 1, figsize=(8, 14))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

# Plot all roads in white (faint)
for u, v, _ in list(G.edges(data=True))[::4]:
    x1, y1 = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
    x2, y2 = float(G.nodes[v]["x"]), float(G.nodes[v]["y"])
    ax.plot([x1, x2], [y1, y2], color="white", linewidth=0.3, alpha=0.4)

# Plot sensor locations as coloured dots
scatter = ax.scatter(
    sensor_edge_df["sensor_lon"],
    sensor_edge_df["sensor_lat"],
    c="red",
    s=20,
    zorder=5,
    alpha=0.9,
    label=f"Traffic sensors (n={len(sensor_edge_df)})"
)

ax.set_title(
    f"Manhattan Roads + {len(sensor_edge_df)} Traffic Sensors\n"
    f"(Synthetic: sensors placed randomly on real road edges)",
    color="white", fontsize=11, pad=10
)
ax.legend(loc="lower right", facecolor="black", labelcolor="white",
          fontsize=9)

plot_path = os.path.join(REPORTS_DIR, "step05_sensor_map.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight",
            facecolor="black", edgecolor="black")
plt.close()
print(f"  Plot saved: {plot_path}")

# ===========================================================================
# SECTION F — Append to step summaries
# ===========================================================================
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a") as f:
    f.write("## Step 05: Match Sensors to Road Edges\n\n")
    f.write(f"**What we did:** Assigned each of the {len(sensor_edge_df)} synthetic "
            f"sensors to a real road edge on the Manhattan graph. For real data, "
            f"GPS coordinates from `link_points` would be used instead.\n\n")
    f.write(f"**What we got:** A CSV table linking each sensor to its road edge "
            f"(u→v node pair, road name, speed limit, travel time).\n\n")
    f.write(f"**Key number:** **{len(sensor_edge_df)} sensors** matched to real "
            f"Manhattan road edges out of {G.number_of_edges():,} total edges.\n\n")
    f.write(f"**Plot:** `reports/step05_sensor_map.png`\n\n")
    f.write("---\n\n")

print(f"  Summary appended: {summary_path}")

print("\n" + "=" * 65)
print("  STEP 05 DONE.")
print(f"  → {len(sensor_edge_df)} sensors mapped to real Manhattan road edges")
print(f"  → Saved: data/processed/sensor_to_edge.csv")
print(f"  → Plot:  reports/step05_sensor_map.png")
print("=" * 65)
