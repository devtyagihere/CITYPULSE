# =============================================================================
# STEP NAME  : step19 — Dijkstra's Shortest Path Algorithm From Scratch
# WHAT       : Implements Dijkstra's algorithm by hand using Python's `heapq`
#              priority queue, and verifies it matches `networkx` on 20 random
#              Manhattan trips.
# WHY        : We want to understand how ambulance routing works under the
#              hood instead of treating networkx as a black box.
# INPUT      : data/processed/manhattan_graph.graphml
# OUTPUT     : reports/step19_dijkstra_check.csv
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os
import heapq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import networkx as nx
from config import DATA_PROCESSED_DIR, REPORTS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 19 — Dijkstra's Algorithm From Scratch (with heapq)")
print("=" * 65)


def dijkstra_heapq(G, start_node, end_node, weight_attr="travel_time"):
    """
    Find the lowest-cost path from start_node to end_node in graph G.
    Uses a min-heap (heapq) so we always expand the closest unvisited node next.
    Returns: (total_cost, list_of_nodes_in_path)
    """
    # Min-heap stores tuples: (cost_so_far, current_node)
    heap = [(0.0, start_node)]
    best_cost = {start_node: 0.0}   # best known cost to reach each node
    parent    = {start_node: None}  # remembers how we got to each node
    visited   = set()

    while heap:
        cost, u = heapq.heappop(heap)

        # Skip if we already finalized a cheaper path to u
        if u in visited:
            continue
        visited.add(u)

        # Stop early as soon as we reach the destination!
        if u == end_node:
            break

        # Check all outgoing road edges u -> v
        for v, edge_dict in G[u].items():
            # For MultiDiGraph, pick minimum weight across parallel edges
            edge_w = min(float(attr.get(weight_attr, 1.0)) for attr in edge_dict.values())
            new_cost = cost + edge_w
            if new_cost < best_cost.get(v, float("inf")):
                best_cost[v] = new_cost
                parent[v]    = u
                heapq.heappush(heap, (new_cost, v))

    if end_node not in best_cost:
        return float("inf"), []

    # Reconstruct path backwards from end_node to start_node
    path = []
    cur = end_node
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return best_cost[end_node], path


# ===========================================================================
# SECTION A — Test against NetworkX on 20 random origin-destination trips
# ===========================================================================
graph_path = os.path.join(DATA_PROCESSED_DIR, "manhattan_graph.graphml")
G = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)

# Ensure numeric travel_time on every edge
for _, _, _, d in G.edges(keys=True, data=True):
    d["travel_time"] = float(d.get("travel_time", 20.0))

all_nodes = list(G.nodes())
checks = []

print("\n[A] Checking our heapq Dijkstra vs NetworkX on 20 random Manhattan trips ...")
for trip_i in range(1, 21):
    src, dst = np.random.choice(all_nodes, size=2, replace=False)
    src, dst = int(src), int(dst)

    my_cost, my_path = dijkstra_heapq(G, src, dst, weight_attr="travel_time")
    nx_cost = nx.shortest_path_length(G, src, dst, weight="travel_time")
    diff    = abs(my_cost - nx_cost)

    checks.append({
        "trip": trip_i, "src": src, "dst": dst,
        "my_time_s": round(my_cost, 3),
        "nx_time_s": round(nx_cost, 3),
        "diff_s": round(diff, 6),
        "stops": len(my_path),
    })

checks_df = pd.DataFrame(checks)
max_diff  = checks_df["diff_s"].max()
print(checks_df.head(8).to_string(index=False))
print(f"\n  Max difference across all 20 trips: {max_diff:.6f} seconds (exact match!)")

assert max_diff < 1e-4, "Mismatch with NetworkX!"
checks_df.to_csv(os.path.join(REPORTS_DIR, "step19_dijkstra_check.csv"), index=False)

summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 19: Dijkstra From Scratch\n\n")
    f.write("**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch "
            "using Python's `heapq` and tested it on 20 random Manhattan trips.\n\n")
    f.write(f"**What we got:** Exact match with NetworkX on all 20 trips "
            f"(max difference = **{max_diff:.6f} s**).\n\n")
    f.write(f"**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.\n\n---\n\n")

print("\n" + "=" * 65)
print("  STEP 19 DONE.  20/20 trips matched NetworkX exactly.")
print("=" * 65)
