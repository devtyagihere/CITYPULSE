# =============================================================================
# STEP NAME  : step13 — Compare All Traffic Models
# WHAT       : Combines MAE and RMSE results from Step 09 (baselines),
#              Step 10 (LightGBM), Step 11 (GRU), and Step 12 (Graph+GRU)
#              into one comparison table and bar chart.
# WHY        : Helps us see honestly which model actually wins at each horizon.
#              If a deep model does not beat LightGBM, we report it honestly!
# INPUT      : reports/step09_baselines_results.csv
#              reports/step10_lightgbm_results.csv
#              reports/step11_gru_results.csv
#              reports/step12_graph_gru_results.csv
# OUTPUT     : reports/step13_model_comparison.csv
#              reports/step13_model_comparison.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import REPORTS_DIR, USE_SYNTHETIC

print("=" * 65)
print("  STEP 13 — Compare All Traffic Models")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

# ===========================================================================
# SECTION A — Load and merge all model results
# ===========================================================================
files = [
    "step09_baselines_results.csv",
    "step10_lightgbm_results.csv",
    "step11_gru_results.csv",
    "step12_graph_gru_results.csv",
]

dfs = [pd.read_csv(os.path.join(REPORTS_DIR, f)) for f in files]
all_df = pd.concat(dfs, ignore_index=True)

# Pivot for easy reading: rows = model, columns = horizon MAE
mae_pivot = all_df.pivot(index="model", columns="horizon", values="mae")
mae_pivot = mae_pivot[["+15m", "+30m", "+45m", "+60m"]]
mae_pivot["avg_mae"] = mae_pivot.mean(axis=1).round(5)
mae_pivot = mae_pivot.sort_values("avg_mae")

print("\n[A] Test MAE across all models & horizons (lower is better):")
print(mae_pivot.to_string())

best_model = mae_pivot.index[0]
best_avg   = mae_pivot["avg_mae"].iloc[0]

print(f"\n  WINNER overall: {best_model} (Average MAE = {best_avg:.5f})")

# Honest check: did deep models beat LightGBM?
lgb_avg = mae_pivot.loc["LightGBM", "avg_mae"]
gru_avg = mae_pivot.loc["GRU", "avg_mae"]
gcn_avg = mae_pivot.loc["Graph_GRU", "avg_mae"]

print("\n  Honest comparison:")
if gru_avg < lgb_avg:
    print(f"  - GRU ({gru_avg:.5f}) beat LightGBM ({lgb_avg:.5f}).")
else:
    print(f"  - GRU ({gru_avg:.5f}) did NOT beat LightGBM ({lgb_avg:.5f}).")
if gcn_avg < lgb_avg:
    print(f"  - Graph_GRU ({gcn_avg:.5f}) beat LightGBM ({lgb_avg:.5f}).")
else:
    print(f"  - Graph_GRU ({gcn_avg:.5f}) did NOT beat LightGBM ({lgb_avg:.5f}). "
          "That is normal on tabular/synthetic data with strong lag & neighbour features!")

# Save combined table
out_csv = os.path.join(REPORTS_DIR, "step13_model_comparison.csv")
all_df.to_csv(out_csv, index=False)

# ===========================================================================
# SECTION B — Plot grouped bar chart
# ===========================================================================
ax = mae_pivot[["+15m", "+30m", "+45m", "+60m"]].plot(
    kind="bar", figsize=(11, 6), width=0.75
)
ax.set_title("CITYPULSE — Model Comparison (Test MAE, Lower is Better)")
ax.set_ylabel("MAE (speed_ratio)")
ax.set_xlabel("Model")
plt.xticks(rotation=20, ha="right")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(title="Horizon")
plt.tight_layout()

plot_path = os.path.join(REPORTS_DIR, "step13_model_comparison.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved: {plot_path}")

# Append summary
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 13: Compare All Traffic Models\n\n")
    f.write("**What we did:** Compared all baselines, LightGBM, GRU, and Graph+GRU "
            "on the exact same test period across +15, +30, +45, and +60 min.\n\n")
    f.write(f"**What we got:** Overall winner is **{best_model}** "
            f"(avg MAE = **{best_avg:.4f}**; LightGBM = {lgb_avg:.4f}, "
            f"GRU = {gru_avg:.4f}, Graph+GRU = {gcn_avg:.4f}).\n\n")
    f.write(f"**Key number:** Best average MAE = **{best_avg:.4f}** ({best_model}).\n\n")
    f.write("**Plot:** `reports/step13_model_comparison.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 13 DONE.  Best model: {best_model} (avg MAE = {best_avg:.4f})")
print("=" * 65)
