# =============================================================================
# STEP NAME  : step04 — Build the Manhattan Road Graph
# WHAT       : Tries to download Manhattan's drivable road network via osmnx.
#              If the network is blocked or times out, builds a realistic
#              synthetic Manhattan grid graph with real GPS coordinates.
# WHY        : We need a road map to route ambulances. Roads form a graph —
#              intersections are nodes, roads are edges. Travel time on each
#              edge = length / speed.
# INPUT      : Internet (tries 3 OSM mirrors) OR synthetic Manhattan grid
# OUTPUT     : data/processed/manhattan_graph.graphml
#              data/processed/manhattan_edges.parquet
#              data/processed/manhattan_nodes.parquet
#              reports/step04_road_map.png
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
import networkx as nx
from config import DATA_PROCESSED_DIR, REPORTS_DIR, STUDY_PLACE, RANDOM_SEED

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 04 — Build the Manhattan Road Graph")
print("=" * 65)

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# A. Try OpenStreetMap mirrors (short 5-sec timeout so we fail fast)
# -----------------------------------------------------------------------
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

G = None
source_str = ""

print("\n[A] Trying OpenStreetMap mirrors ...")
try:
    import osmnx as ox
    for mirror in MIRRORS:
        host = mirror.split("/")[2]
        try:
            print(f"  Trying {host} ...", end=" ", flush=True)
            ox.settings.overpass_url = mirror
            ox.settings.requests_timeout = 5
            G = ox.graph_from_place(STUDY_PLACE, network_type="drive", retain_all=False)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            source_str = "Real OSM (" + host + ")"
            print("OK!")
            break
        except Exception as ex:
            print("failed (" + type(ex).__name__ + ")")
            G = None
except ImportError:
    print("  osmnx not installed.")

# -----------------------------------------------------------------------
# B. Fallback: realistic synthetic Manhattan grid
#    Use MultiDiGraph so it matches osmnx's exact graph type (u, v, key).
# -----------------------------------------------------------------------
if G is None:
    print("\n[B] Building synthetic Manhattan grid graph ...")
    print("  (Manhattan is a near-perfect grid — this is a realistic model)")
    print("  Re-run this step with internet to get the real OSM graph.")

    # Real Manhattan bounding box (approximate GPS)
    LON_MIN, LON_MAX = -74.02, -73.91   # west to east
    LAT_MIN, LAT_MAX = 40.70, 40.88     # south to north

    N_COLS = 30   # avenues (north-south corridors)
    N_ROWS = 80   # streets  (east-west corridors)

    lon_step = (LON_MAX - LON_MIN) / (N_COLS - 1)
    lat_step = (LAT_MAX - LAT_MIN) / (N_ROWS - 1)

    # Metres per degree at NYC (40.8 N latitude)
    M_PER_LON = 82000
    M_PER_LAT = 111000

    block_w = lon_step * M_PER_LON   # ~300 m between avenues
    block_h = lat_step * M_PER_LAT   # ~250 m between streets

    AVE_SPD = 48.0   # 30 mph -- typical NYC avenue
    STR_SPD = 40.0   # 25 mph -- typical NYC cross street

    G = nx.MultiDiGraph()
    nid = 0
    nmap = {}

    # Add one node per grid intersection
    for row in range(N_ROWS):
        for col in range(N_COLS):
            lon = LON_MIN + col * lon_step
            lat = LAT_MIN + row * lat_step
            G.add_node(nid, x=lon, y=lat, osmid=nid)
            nmap[(col, row)] = nid
            nid += 1

    # Add directed edges in both directions
    eid = 0
    for row in range(N_ROWS):
        for col in range(N_COLS):
            u = nmap[(col, row)]

            # North-south (avenue segment one block north)
            if row + 1 < N_ROWS:
                v = nmap[(col, row + 1)]
                tt = block_h / (AVE_SPD * 1000.0 / 3600.0)
                for src, dst in [(u, v), (v, u)]:
                    G.add_edge(src, dst, key=0, osmid=eid,
                               name="Avenue_" + str(col),
                               highway="secondary",
                               length=block_h, speed_kph=AVE_SPD, travel_time=tt)
                    eid += 1

            # East-west (street segment one block east)
            if col + 1 < N_COLS:
                v = nmap[(col + 1, row)]
                tt = block_w / (STR_SPD * 1000.0 / 3600.0)
                for src, dst in [(u, v), (v, u)]:
                    G.add_edge(src, dst, key=0, osmid=eid,
                               name="Street_" + str(row),
                               highway="residential",
                               length=block_w, speed_kph=STR_SPD, travel_time=tt)
                    eid += 1

    source_str = "Synthetic Manhattan grid (OSM unavailable)"
    print(f"  Grid: {N_COLS} cols x {N_ROWS} rows = {nid} nodes, {eid} directed edges")

# -----------------------------------------------------------------------
# C. Print graph stats
# -----------------------------------------------------------------------
print(f"\n[C] Graph ready — source: {source_str}")
print(f"  Nodes: {G.number_of_nodes():,}")
print(f"  Edges: {G.number_of_edges():,}")

travel_times = [d.get("travel_time", 0) for _, _, d in G.edges(data=True)]
speeds_kph   = [d.get("speed_kph",   0) for _, _, d in G.edges(data=True)]
lengths_m    = [d.get("length",       0) for _, _, d in G.edges(data=True)]

print(f"  Travel time (s): min={min(travel_times):.1f}  mean={np.mean(travel_times):.1f}  max={max(travel_times):.1f}")
print(f"  Speeds (kph): {sorted(set(round(s) for s in speeds_kph))}")
print(f"  Lengths (m) : mean={np.mean(lengths_m):.0f}")

# -----------------------------------------------------------------------
# D. Build DataFrames
# -----------------------------------------------------------------------
print("\n[D] Building node and edge tables ...")

nodes_data = [{"osmid": n, "x": d["x"], "y": d["y"]} for n, d in G.nodes(data=True)]
nodes_df = pd.DataFrame(nodes_data)

edges_data = []
for u, v, k, d in G.edges(keys=True, data=True):
    edges_data.append({
        "u": u, "v": v, "key": k,
        "osmid": str(d.get("osmid", 0)),
        "name": str(d.get("name", "")),
        "highway": str(d.get("highway", "")),
        "length": float(d.get("length", 0)),
        "speed_kph": float(d.get("speed_kph", 0)),
        "travel_time": float(d.get("travel_time", 0)),
    })
edges_df = pd.DataFrame(edges_data)

print(f"  Nodes table: {len(nodes_df):,} rows")
print(f"  Edges table: {len(edges_df):,} rows")
print(edges_df[["u", "v", "name", "length", "speed_kph", "travel_time"]].head(5).to_string(index=False))

# -----------------------------------------------------------------------
# E. Save
# -----------------------------------------------------------------------
print("\n[E] Saving ...")

graph_path = os.path.join(DATA_PROCESSED_DIR, "manhattan_graph.graphml")
nodes_path = os.path.join(DATA_PROCESSED_DIR, "manhattan_nodes.parquet")
edges_path = os.path.join(DATA_PROCESSED_DIR, "manhattan_edges.parquet")

nx.write_graphml(G, graph_path)
nodes_df.to_parquet(nodes_path, index=False)
edges_df.to_parquet(edges_path, index=False)

print(f"  Graph : {graph_path}")
print(f"  Nodes : {nodes_path}")
print(f"  Edges : {edges_path}")

# -----------------------------------------------------------------------
# F. Plot
# -----------------------------------------------------------------------
print("\n[F] Plotting ...")

fig, ax = plt.subplots(1, 1, figsize=(8, 14))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

all_edges = list(G.edges(data=True))
for u, v, data in all_edges[::4]:
    x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
    x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
    ax.plot([x1, x2], [y1, y2], color="white", linewidth=0.3, alpha=0.5)

ax.set_title(
    f"Manhattan Road Network\n({source_str})\n"
    f"{G.number_of_nodes():,} intersections   {G.number_of_edges():,} road segments",
    color="white", fontsize=10, pad=10
)
ax.set_xlabel("Longitude", color="white")
ax.set_ylabel("Latitude",  color="white")
ax.tick_params(colors="white")

plot_path = os.path.join(REPORTS_DIR, "step04_road_map.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight", facecolor="black")
plt.close()
print(f"  Plot saved: {plot_path}")

# -----------------------------------------------------------------------
# G. Step summary
# -----------------------------------------------------------------------
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 04: Build the Road Graph\n\n")
    f.write(f"**What we did:** Built Manhattan road graph. Source: *{source_str}*. "
            f"Added speed limits and travel times to every road segment.\n\n")
    f.write(f"**What we got:** {G.number_of_nodes():,} nodes (intersections) and "
            f"{G.number_of_edges():,} edges (road segments).\n\n")
    f.write(f"**Key number:** Avg travel time per edge = **"
            f"{np.mean(travel_times):.1f} seconds**.\n\n")
    if "Synthetic" in source_str:
        f.write("> NOTE: Synthetic grid used. Re-run with internet for real OSM graph.\n\n")
    f.write("**Plot:** `reports/step04_road_map.png`\n\n---\n\n")

print("  Summary appended.")

print("\n" + "=" * 65)
print("  STEP 04 DONE.")
print(f"  Source : {source_str}")
print(f"  Nodes  : {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}")
print(f"  Avg travel time: {np.mean(travel_times):.1f}s per edge")
print("  Saved  : data/processed/manhattan_graph.graphml")
print("=" * 65)
