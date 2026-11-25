# CITYPULSE — Traffic, Air Quality & Ambulance Routing (Manhattan, NYC)

> ⚠️ **SYNTHETIC DATA NOTICE:** The current results below were generated with `USE_SYNTHETIC = True` to test the end-to-end pipeline on a normal laptop. Final real-world results must be re-run after placing real NYC DOT, EPA, and NOAA files in `data/raw/`.

---

## Flow Diagram

```text
[1. Data: Traffic + Weather + PM2.5]
        │  (clean, 15-min bins, America/New_York, train-only free-flow)
        ▼
[2. Manhattan Road Graph + Sensor Matching + Adjacency Matrix]
        │
        ▼
[3. Leakage-Free Features] ──► (a) Flat Table (LightGBM)
        │                      (b) 4D Sequences (GRU & Graph+GRU)
        ▼
[4. Traffic Models (+15/+30/+45/+60m)] + [5. P10/P50/P90 + Calibration + SHAP]
        │
        ├──► [6. City Crisis Warning & 0–100 Congestion Index]
        ├──► [7. Future PM2.5 (+1h/+3h/+6h) & AQI Category]
        └──► [8. Uncertainty-Aware Dijkstra Ambulance Routing]
                    │
                    ▼
             [9. Live Pipeline + FastAPI + Leaflet Dashboard]
```

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the full pipeline (Steps 01–21):**
   ```bash
   python run_all.py
   ```
3. **Run the live system & dashboard (Steps 22–24):**
   ```bash
   python steps/step22_live_pipeline.py
   python steps/step23_api.py
   ```
   Then open **http://127.0.0.1:8000** in your browser.

> **Google Colab Note:** Steps 11 (`GRU`) and 12 (`Graph+GRU`) automatically use a GPU (`cuda`) if available. On a normal laptop CPU they run 12 small epochs so everything finishes in minutes; on Google Colab you can increase `EPOCHS` in `step11_lstm_gru.py` and `step12_graph_model.py` for heavier training.

---

## Switching from Synthetic Data to Real Data

Open `src/config.py` and change **one line**:
```python
USE_SYNTHETIC = False
```
Place your real CSV files in `data/raw/`:
- NYC DOT Traffic Speeds (`DOT_Traffic_Speeds_NBE.csv`)
- NOAA KNYC Central Park Weather (`weather_knyc_2022.csv`)
- EPA Hourly PM2.5 (`hourly_88101_2022.csv`)

---

## What Each Step Does

| Step | File | Purpose |
|---|---|---|
| 01 | `steps/step01_make_synthetic_data.py` | Generates 6 months of realistic fake traffic, weather, and PM2.5 |
| 02 | `steps/step02_load_and_check_data.py` | Loads CSVs, checks missing %, plots 1 sensor for 1 week |
| 03 | `steps/step03_clean_data.py` | Fixes timezone, 15-min resample, fills ≤1h gaps, computes train-only `speed_ratio` |
| 04 | `steps/step04_road_graph.py` | Builds Manhattan drivable road graph (OSM or synthetic grid fallback) |
| 05 | `steps/step05_match_sensors.py` | Maps 50 sensors onto Manhattan road edges |
| 06 | `steps/step06_adjacency_matrix.py` | Builds road-distance Gaussian kernel adjacency matrix |
| 07 | `steps/step07_leakage_test.py` | Proves future-feature leakage drops error unrealistically (98.1% fake drop) |
| 08 | `steps/step08_build_features.py` | Builds past-only LightGBM tables + 4D deep arrays, split 70/10/20 by time |
| 09 | `steps/step09_baselines.py` | Evaluates Persistence, Historical Average, and Seasonal Naive |
| 10 | `steps/step10_lightgbm.py` | Trains 4 LightGBM models (+15, +30, +45, +60 min) |
| 11 | `steps/step11_lstm_gru.py` | Trains PyTorch GRU sequence model |
| 12 | `steps/step12_graph_model.py` | Trains PyTorch Graph Convolution + GRU with masked MAE |
| 13 | `steps/step13_compare_models.py` | Compares all models honestly across all horizons |
| 14 | `steps/step14_uncertainty.py` | Quantile LightGBM (`P10`, `P50`, `P90`) for 80% prediction intervals |
| 15 | `steps/step15_calibration.py` | Isotonic Regression for calibrated congestion probabilities |
| 16 | `steps/step16_explain_shap.py` | SHAP summary plot & top-5 human-readable drivers |
| 17 | `steps/step17_crisis_model.py` | LightGBM crisis classifier + Isolation Forest + 0–100 Congestion Index |
| 18 | `steps/step18_aqi_model.py` | Predicts PM2.5 at +1h, +3h, +6h + SHAP + AQI categories |
| 19 | `steps/step19_dijkstra_from_scratch.py` | Hand-written `heapq` Dijkstra verified against NetworkX |
| 20 | `steps/step20_predicted_traffic_routing.py` | Uncertainty-aware 3 alternative routes + cleanest-air route |
| 21 | `steps/step21_route_evaluation.py` | Replays 60 test trips comparing Current, CITYPULSE, and Oracle routes |
| 22 | `steps/step22_live_pipeline.py` | Fetches live KNYC weather + runs `Simulated Input` predictions |
| 23 | `steps/step23_api.py` | FastAPI server (`/health`, `/predict`, `/route`, `/crisis`) |
| 24 | `steps/step24_dashboard.html` | Leaflet map dashboard with +15/+30/+45/+60 slider |

---

## Final Results (Synthetic Pipeline Test)

| Task | Metric | Observed Result |
|---|---|---|
| Traffic (+15m) | Test MAE (`speed_ratio`) | **GRU: `0.07555`** \| LightGBM: `0.08150` \| Graph+GRU: `0.08398` \| Hist Avg: `0.08352` |
| Traffic (Avg +15..+60m) | Mean Test MAE | **GRU: `0.07641`** \| LightGBM: `0.08079` \| Graph+GRU: `0.08512` |
| Uncertainty (`P10`–`P90`) | Empirical Coverage | **`79.66%`** (target ~80%) |
| Congestion Calibration | Test Brier Score | **`0.05033`** |
| City Crisis (next 30m) | Test PR-AUC / F1 | **PR-AUC = `0.9814`**, **F1 = `0.9085`** |
| PM2.5 (+1h / +3h / +6h) | Test MAE (`µg/m³`) | **`2.373` / `3.016` / `2.986`** (vs Persistence `4.102` / `7.519` / `7.821`) |
| Route Replay (60 trips) | Win/Tie vs Current | **`61.7%`** (`35.18 min` vs `35.20 min`) |
