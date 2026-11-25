# =============================================================================
# STEP NAME  : run_all — Run the Full CITYPULSE Pipeline (Steps 01 to 21)
# WHAT       : Runs every training, evaluation, and routing step (01–21)
#              in strict order and prints a final summary of results.
# WHY        : Lets you reproduce the entire project from raw data to
#              ambulance route evaluation with a single command.
# INPUT      : src/config.py (USE_SYNTHETIC=True or False)
# OUTPUT     : All files in data/processed/, models/, and reports/
# =============================================================================

import os
import sys
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR    = os.path.join(PROJECT_ROOT, "steps")

STEPS = [
    "step01_make_synthetic_data.py",
    "step02_load_and_check_data.py",
    "step03_clean_data.py",
    "step04_road_graph.py",
    "step05_match_sensors.py",
    "step06_adjacency_matrix.py",
    "step07_leakage_test.py",
    "step08_build_features.py",
    "step09_baselines.py",
    "step10_lightgbm.py",
    "step11_lstm_gru.py",
    "step12_graph_model.py",
    "step13_compare_models.py",
    "step14_uncertainty.py",
    "step15_calibration.py",
    "step16_explain_shap.py",
    "step17_crisis_model.py",
    "step18_aqi_model.py",
    "step19_dijkstra_from_scratch.py",
    "step20_predicted_traffic_routing.py",
    "step21_route_evaluation.py",
]


def main():
    print("=" * 65)
    print("  CITYPULSE — Running Full Pipeline (Steps 01 to 21)")
    print("=" * 65)
    t0 = time.time()

    for idx, step_file in enumerate(STEPS, start=1):
        step_path = os.path.join(STEPS_DIR, step_file)
        print(f"\n>>> [{idx}/{len(STEPS)}] Running {step_file} ...")
        res = subprocess.run([sys.executable, step_path], cwd=PROJECT_ROOT)
        if res.returncode != 0:
            print(f"\nERROR: {step_file} failed with exit code {res.returncode}.")
            sys.exit(res.returncode)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"  ALL 21 STEPS COMPLETED in {elapsed:.1f} seconds!")
    print("  Next: run live pipeline & API:")
    print("    python steps/step22_live_pipeline.py")
    print("    python steps/step23_api.py")
    print("    Open http://127.0.0.1:8000 in your browser")
    print("=" * 65)


if __name__ == "__main__":
    main()
