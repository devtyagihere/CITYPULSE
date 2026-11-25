# =============================================================================
# STEP NAME  : step21 — Replay & Evaluate Ambulance Routes in Test Period
# WHAT       : Replays 60 ambulance trips across the test period and compares:
#              1) Current-Traffic Route (uses speed right now, lag_0)
#              2) CITYPULSE Route (uses predicted +15m speed + uncertainty penalty)
#              3) Oracle Route (uses actual future speeds — theoretical best)
# WHY        : Proves whether ML-predicted routing actually saves ambulance
#              travel time compared to naive current-traffic routing.
# INPUT      : data/processed/manhattan_graph.graphml
#              data/processed/sensor_to_edge.csv
#              data/processed/lgb_test.parquet
#              data/processed/test_quantiles_15m.parquet
# OUTPUT     : reports/step21_route_eval.csv
#              reports/step21_example_routes.png
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
print("  STEP 21 — Replay 60 Ambulance Trips (Route Evaluation)")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

G = nx.read_graphml(os.path.join(DATA_PROCESSED_DIR, "manhattan_graph.graphml"),
                    node_type=int, force_multigraph=True)
sensor_edge_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "sensor_to_edge.csv"))
lgb_test       = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "lgb_test.parquet"))
test_q         = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "test_quantiles_15m.parquet"))

# Merge current speed (lag_0) with predicted quantiles (p50, p10) and true future (target_15m)
merged = test_q.merge(lgb_test[["link_id", "time", "lag_0"]], on=["link_id", "time"], how="inner")

# Save base travel time per edge once
for _, _, _, d in G.edges(keys=True, data=True):
    d["base_tt"] = float(d.get("travel_time", 20.0))

sensor_edges = {(int(r["edge_u"]), int(r["edge_v"])): r["link_id"]
                for _, r in sensor_edge_df.iterrows()}

unique_times = np.sort(merged["time"].unique())
all_nodes    = sorted(G.nodes())
N_TRIPS      = 60

# Sample 60 timestamps evenly across test set
sample_times = unique_times[np.linspace(0, len(unique_times) - 1, N_TRIPS, dtype=int)]
trip_rows    = []
example_plots = []

print(f"\n[A] Replaying {N_TRIPS} trips in the test period ...")

for idx, t in enumerate(sample_times):
    sl = merged[merged["time"] == t].set_index("link_id")
    cur_map  = sl["lag_0"].to_dict()
    p50_map  = sl["p50"].to_dict()
    p10_map  = sl["p10"].to_dict()
    true_map = sl["target_15m"].to_dict()

    c_avg = float(sl["lag_0"].mean())
    p_avg = float(sl["p50"].mean())
    l_avg = float(sl["p10"].mean())
    t_avg = float(sl["target_15m"].mean())

    # Update edge weights for all 3 routing modes + ground truth travel time
    for u, v, _, d in G.edges(keys=True, data=True):
        base = d["base_tt"]
        lid  = sensor_edges.get((u, v))
        sr_cur  = max(float(cur_map.get(lid, c_avg)), 0.15)
        sr_p50  = max(float(p50_map.get(lid, p_avg)), 0.15)
        sr_p10  = max(float(p10_map.get(lid, l_avg)), 0.15)
        sr_true = max(float(true_map.get(lid, t_avg)), 0.15)

        d["w_current"]   = base / sr_cur
        d["w_citypulse"] = (base / sr_p50) + ROUTE_UNCERTAINTY_LAMBDA * max(0.0, (base / sr_p10) - (base / sr_p50))
        d["w_true"]      = base / sr_true

    # Pick random origin and destination separated by some distance
    src = int(np.random.choice(all_nodes[:len(all_nodes) // 3]))
    dst = int(np.random.choice(all_nodes[2 * len(all_nodes) // 3:]))

    _, path_cur  = dijkstra_heapq(G, src, dst, "w_current")
    _, path_cp   = dijkstra_heapq(G, src, dst, "w_citypulse")
    _, path_orac = dijkstra_heapq(G, src, dst, "w_true")

    # Measure ACTUAL future travel time of each chosen path using w_true
    def actual_time(path):
        return sum(G[path[i]][path[i+1]][0]["w_true"] for i in range(len(path) - 1)) / 60.0

    t_cur  = actual_time(path_cur)
    t_cp   = actual_time(path_cp)
    t_orac = actual_time(path_orac)

    trip_rows.append({
        "trip": idx + 1,
        "current_min": round(t_cur, 3),
        "citypulse_min": round(t_cp, 3),
        "oracle_min": round(t_orac, 3),
        "citypulse_beats_or_ties_current": t_cp <= t_cur,
    })

    if idx < 2:
        example_plots.append((idx + 1, src, dst, path_cur, path_cp, t_cur, t_cp))

eval_df = pd.DataFrame(trip_rows)
eval_df.to_csv(os.path.join(REPORTS_DIR, "step21_route_eval.csv"), index=False)

mean_cur  = eval_df["current_min"].mean()
mean_cp   = eval_df["citypulse_min"].mean()
mean_orac = eval_df["oracle_min"].mean()
p90_cur   = np.percentile(eval_df["current_min"], 90)
p90_cp    = np.percentile(eval_df["citypulse_min"], 90)
p90_orac  = np.percentile(eval_df["oracle_min"], 90)
win_rate  = (eval_df["citypulse_min"] <= eval_df["current_min"]).mean() * 100

print("\n  Route Evaluation Summary (60 replayed test trips):")
print(f"    Current-Traffic : mean = {mean_cur:.2f} min  |  P90 = {p90_cur:.2f} min")
print(f"    CITYPULSE       : mean = {mean_cp:.2f} min  |  P90 = {p90_cp:.2f} min")
print(f"    Oracle (future) : mean = {mean_orac:.2f} min  |  P90 = {p90_orac:.2f} min")
print(f"    CITYPULSE win/tie rate vs Current-Traffic: {win_rate:.1f}%")

# Plot 2 example trips side-by-side
fig, axes = plt.subplots(1, 2, figsize=(13, 10))
fig.patch.set_facecolor("black")
for ax, (t_num, src, dst, p_cur, p_cp, tc, tcp) in zip(axes, example_plots):
    ax.set_facecolor("black")
    for u, v, _ in list(G.edges(data=True))[::6]:
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color="dimgray", linewidth=0.3, alpha=0.35)
    ax.plot([G.nodes[n]["x"] for n in p_cur], [G.nodes[n]["y"] for n in p_cur],
            color="orange", linewidth=2.0, linestyle="--", label=f"Current ({tc:.1f}m)")
    ax.plot([G.nodes[n]["x"] for n in p_cp],  [G.nodes[n]["y"] for n in p_cp],
            color="lime",   linewidth=2.2, label=f"CITYPULSE ({tcp:.1f}m)")
    ax.scatter([G.nodes[src]["x"]], [G.nodes[src]["y"]], c="white", s=70, zorder=10)
    ax.scatter([G.nodes[dst]["x"]], [G.nodes[dst]["y"]], c="red",   s=70, zorder=10)
    ax.set_title(f"Example Trip #{t_num}", color="white")
    ax.legend(facecolor="black", labelcolor="white")
    ax.tick_params(colors="white")

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "step21_example_routes.png"), dpi=120, bbox_inches="tight", facecolor="black")
plt.close()

summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 21: Route Replay Evaluation\n\n")
    f.write("**What we did:** Replayed 60 ambulance trips in the test period comparing "
            "Current-Traffic, CITYPULSE (predicted + uncertainty), and Oracle routes.\n\n")
    f.write(f"**What we got:** Mean times — Current = {mean_cur:.2f}m, "
            f"CITYPULSE = **{mean_cp:.2f}m**, Oracle = {mean_orac:.2f}m "
            f"(P90: {p90_cur:.2f}m vs {p90_cp:.2f}m).\n\n")
    f.write(f"**Key number:** CITYPULSE win/tie rate vs Current-Traffic = **{win_rate:.1f}%**.\n\n")
    f.write("**Plot:** `reports/step21_example_routes.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 21 DONE.  CITYPULSE mean={mean_cp:.2f}m | win/tie={win_rate:.1f}%")
print("=" * 65)
