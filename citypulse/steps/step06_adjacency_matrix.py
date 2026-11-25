# =============================================================================
# STEP NAME  : step06 — Build the Sensor Adjacency Matrix
# WHAT       : Builds a matrix that shows how close each sensor is to every
#              other sensor on the road network. Close sensors get a high
#              weight; far sensors get a weight near zero.
#              Uses a Gaussian kernel — the same method as METR-LA / DCRNN.
# WHY        : Graph neural networks need to know which sensors are neighbours.
#              Road distance is better than straight-line distance because
#              a sensor 500m away on a parallel street is not a neighbour
#              if there is no direct road connection.
# INPUT      : data/processed/manhattan_graph.graphml
#              data/processed/sensor_to_edge.csv
# OUTPUT     : data/processed/adj_matrix.npy     (N×N float32 weight matrix)
#              data/processed/sensor_order.csv    (which row = which sensor)
#              reports/step06_adjacency_heatmap.png
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
from config import (DATA_PROCESSED_DIR, REPORTS_DIR, RANDOM_SEED)

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 06 — Build Sensor Adjacency Matrix")
print("  (Uses road-network distance + Gaussian kernel, like DCRNN)")
print("=" * 65)

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ===========================================================================
# SECTION A — Load the graph and sensor mapping
# ===========================================================================
print("\n[A] Loading graph and sensor-edge mapping ...")

graph_path   = os.path.join(DATA_PROCESSED_DIR, "manhattan_graph.graphml")
mapping_path = os.path.join(DATA_PROCESSED_DIR, "sensor_to_edge.csv")

G              = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
sensor_edge_df = pd.read_csv(mapping_path)

n_sensors  = len(sensor_edge_df)
sensor_ids = list(sensor_edge_df["link_id"])

print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  Sensors: {n_sensors}")

# ===========================================================================
# SECTION B — Find the nearest graph node for each sensor
# ===========================================================================
print("\n[B] Finding the nearest graph node for each sensor ...")

# Each sensor sits on an edge (u → v). We pick the START node (u) as the
# sensor's representative node. This is good enough for distance computation.
sensor_nodes = []
for _, row in sensor_edge_df.iterrows():
    sensor_nodes.append(int(row["edge_u"]))   # start node of the sensor's edge

sensor_edge_df["graph_node"] = sensor_nodes
print(f"  Assigned one graph node per sensor (start of its road edge).")
print(f"  Unique nodes used: {len(set(sensor_nodes))}")

# ===========================================================================
# SECTION C — Compute road distances between ALL pairs of sensor nodes
# ===========================================================================
print(f"\n[C] Computing shortest road distances between {n_sensors} sensor nodes ...")
print(f"  This runs {n_sensors}×{n_sensors} = {n_sensors*n_sensors} pairs ...")
print(f"  Using 'travel_time' as distance (seconds) — better than metres for routing.")

# Build an N×N distance matrix, initialised to infinity
dist_matrix = np.full((n_sensors, n_sensors), np.inf, dtype=np.float32)

for i, node_i in enumerate(sensor_nodes):
    # Compute shortest path from node_i to ALL other nodes at once
    # This is much faster than computing each pair separately
    lengths = nx.single_source_dijkstra_path_length(
        G,
        source=node_i,
        weight="travel_time"   # use travel time (seconds) as edge weight
    )

    for j, node_j in enumerate(sensor_nodes):
        if node_j in lengths:
            dist_matrix[i, j] = lengths[node_j]  # travel time in seconds
        # if not reachable, stays as inf

    # Print progress every 10 sensors
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{n_sensors} sensors ...")

# Set diagonal to 0 (distance from a sensor to itself = 0)
np.fill_diagonal(dist_matrix, 0)

# Replace remaining inf (unreachable nodes) with a very large number
dist_matrix[np.isinf(dist_matrix)] = dist_matrix[~np.isinf(dist_matrix)].max() * 2

print(f"\n  Distance matrix: shape {dist_matrix.shape}")
print(f"  Distance range: {dist_matrix.min():.1f}s to {dist_matrix.max():.1f}s")
print(f"  Mean distance between sensors: {dist_matrix[dist_matrix > 0].mean():.1f}s")

# ===========================================================================
# SECTION D — Apply Gaussian kernel to convert distances to weights
# ===========================================================================
print("\n[D] Applying Gaussian kernel to get adjacency weights ...")

# The Gaussian kernel:  W[i,j] = exp( -distance² / sigma² )
# Close sensors (small distance) → weight near 1
# Far sensors (large distance)   → weight near 0
# This is exactly the DCRNN / METR-LA method.

# sigma = standard deviation of all distances (a common choice)
sigma = dist_matrix[dist_matrix > 0].std()
print(f"  Sigma (distance std): {sigma:.1f} seconds")

adj_matrix = np.exp(-(dist_matrix ** 2) / (sigma ** 2))

# Zero out the diagonal (a sensor is not a neighbour of itself in GCN)
np.fill_diagonal(adj_matrix, 0)

# Apply a threshold: very weak connections (< 0.1) are set to 0
# This makes the adjacency matrix sparse (faster computation)
threshold = 0.1
adj_matrix[adj_matrix < threshold] = 0.0

n_nonzero = np.count_nonzero(adj_matrix)
sparsity  = 1 - n_nonzero / (n_sensors * n_sensors)
print(f"  Threshold applied: {threshold}")
print(f"  Non-zero connections: {n_nonzero} / {n_sensors*n_sensors} "
      f"({sparsity*100:.1f}% sparse)")
print(f"  Weight range: {adj_matrix[adj_matrix>0].min():.4f} to {adj_matrix.max():.4f}")

# ===========================================================================
# SECTION E — Sanity check: does the matrix make sense?
# ===========================================================================
print("\n[E] Sanity check ...")

# Every sensor should have at least 1 neighbour
n_isolated = (adj_matrix.sum(axis=1) == 0).sum()
print(f"  Sensors with NO neighbours (isolated): {n_isolated}  (should be 0 or very small)")

# Average number of neighbours per sensor
avg_neighbours = (adj_matrix > 0).sum(axis=1).mean()
print(f"  Average neighbours per sensor: {avg_neighbours:.1f}")

# ===========================================================================
# SECTION F — Save adjacency matrix and sensor order
# ===========================================================================
print("\n[F] Saving adjacency matrix and sensor order ...")

adj_path   = os.path.join(DATA_PROCESSED_DIR, "adj_matrix.npy")
order_path = os.path.join(DATA_PROCESSED_DIR, "sensor_order.csv")

np.save(adj_path, adj_matrix)                      # save as numpy binary
print(f"  Adjacency matrix: {adj_path}")

# Save the sensor order (row i in the matrix = which sensor)
order_df = pd.DataFrame({
    "row_index"  : range(n_sensors),
    "link_id"    : sensor_ids,
    "graph_node" : sensor_nodes,
})
order_df.to_csv(order_path, index=False)
print(f"  Sensor order:     {order_path}")

# ===========================================================================
# SECTION G — Plot the adjacency matrix as a heatmap
# ===========================================================================
print("\n[G] Plotting adjacency matrix heatmap ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("CITYPULSE — Step 06: Sensor Adjacency Matrix",
             fontsize=13, fontweight="bold")

# --- Plot 1: Full heatmap ---
im = axes[0].imshow(adj_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
axes[0].set_title(f"Adjacency matrix ({n_sensors}×{n_sensors})\n"
                  f"Yellow = close neighbours, White = far apart")
axes[0].set_xlabel("Sensor index")
axes[0].set_ylabel("Sensor index")
plt.colorbar(im, ax=axes[0], label="Connection weight (0=far, 1=close)")

# --- Plot 2: Distribution of non-zero weights ---
nonzero_weights = adj_matrix[adj_matrix > 0]
axes[1].hist(nonzero_weights, bins=30, color="coral", edgecolor="white", alpha=0.8)
axes[1].set_xlabel("Connection weight")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Distribution of non-zero weights\n"
                  f"(Mean = {nonzero_weights.mean():.3f}, "
                  f"Non-zero connections = {n_nonzero})")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "step06_adjacency_heatmap.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Plot saved: {plot_path}")

# ===========================================================================
# SECTION H — Explain the adjacency matrix in simple words
# ===========================================================================
print("\n[H] What this matrix means (simple explanation):")
print("""
  Think of the matrix like a table of friendships between sensors:
  - Row 3, Column 7 = 0.85 → Sensor 3 and Sensor 7 are close on the road
  - Row 3, Column 42 = 0.0  → Sensor 3 and Sensor 42 are far apart

  The graph model uses this table to let nearby sensors "share information".
  If Sensor 7 is seeing a traffic jam, Sensor 3 (nearby) should also
  expect a jam soon. This is the whole idea behind graph neural networks!

  We used road distance (not straight-line distance) because:
  - Two sensors might be 100m apart as the crow flies
  - But if a river or block separates them, they might be 2km apart by road
  - Road distance captures the REAL influence between sensors.
""")

# ===========================================================================
# SECTION I — Append to step summaries
# ===========================================================================
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a") as f:
    f.write("## Step 06: Build Sensor Adjacency Matrix\n\n")
    f.write(f"**What we did:** Computed shortest road-travel-time distances "
            f"between all {n_sensors} sensor pairs, then applied a Gaussian "
            f"kernel (σ={sigma:.1f}s) to convert distances into connection "
            f"weights. Close sensors → high weight; far sensors → 0.\n\n")
    f.write(f"**What we got:** A {n_sensors}×{n_sensors} float32 weight matrix. "
            f"{n_nonzero} non-zero connections ({sparsity*100:.1f}% sparse). "
            f"Average {avg_neighbours:.1f} neighbours per sensor.\n\n")
    f.write(f"**Key number:** σ = **{sigma:.1f} seconds** road-travel-time. "
            f"This is the 'neighbourhood radius' — sensors reachable within "
            f"about {sigma:.0f}s of travel time are considered neighbours.\n\n")
    f.write(f"**Plot:** `reports/step06_adjacency_heatmap.png`\n\n")
    f.write("---\n\n")

print(f"  Summary appended: {summary_path}")

print("\n" + "=" * 65)
print("  STEP 06 DONE.")
print(f"  → Adjacency matrix: {n_sensors}×{n_sensors}, "
      f"{n_nonzero} non-zero connections")
print(f"  → Sigma: {sigma:.1f}s  |  Avg neighbours: {avg_neighbours:.1f}")
print(f"  → Saved: data/processed/adj_matrix.npy")
print("=" * 65)
