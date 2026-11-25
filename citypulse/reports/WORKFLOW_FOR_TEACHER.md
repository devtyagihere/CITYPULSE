# CITYPULSE — One-Page Project Summary for Teacher

> **Disclaimer:** This is a university **research prototype** tested on synthetic data (`USE_SYNTHETIC = True`). It is **not** intended for real emergency dispatch, and final accuracy must be re-evaluated on real NYC DOT / EPA / NOAA data.

## 1. Problem & Data
Emergency ambulances in Manhattan lose critical minutes when routed using only *current* traffic speeds, because traffic can worsen 15–30 minutes *after* dispatch. **CITYPULSE** forecasts future Manhattan road speeds (`+15`, `+30`, `+45`, `+60` min), air quality (`PM2.5` at `+1h`, `+3h`, `+6h`), and city-wide congestion crises, then routes ambulances using uncertainty-aware Dijkstra.

## 2. The 10 Phases in Simple Words
1. **Data (Steps 01–03):** Generate/load traffic, weather, and PM2.5; clean glitches; resample to 15-min bins; compute `speed_ratio = speed / train_free_flow`.
2. **Road Graph (Steps 04–06):** Build Manhattan's road network, pin sensors to edges, and compute a road-distance adjacency matrix.
3. **Features (Steps 07–08):** Verify zero data leakage, then build past-only lag/rolling/weather/neighbour tables (for LightGBM) and 4D windows (for PyTorch). Split strictly by time (`70% / 10% / 20%`).
4. **Traffic Models (Steps 09–13):** Compare 3 baselines, LightGBM, PyTorch GRU, and PyTorch Graph+GRU.
5. **Uncertainty & Explainability (Steps 14–16):** Predict `P10`/`P50`/`P90` bands, calibrate congestion probability (Isotonic), and explain drivers with SHAP.
6. **Crisis Model (Step 17):** Predict city-wide gridlock 30 min ahead (LightGBM + Isolation Forest) and compute a `0–100` Congestion Index.
7. **AQI Model (Step 18):** Forecast `PM2.5` at `+1h/+3h/+6h` using traffic + weather + PM2.5 lags.
8. **Routing (Steps 19–21):** Write `heapq` Dijkstra from scratch, add `lambda * (P90 - P50)` risk penalty, and replay 60 test trips.
9. **Live System (Steps 22–24):** Fetch live Central Park (`KNYC`) weather, serve `/predict`, `/route`, `/crisis`, `/health` via FastAPI, and display on a Leaflet map.
10. **Packaging:** Single-command `run_all.py` and documentation.

## 3. Models Used & Key Results (Synthetic Pipeline Test)
- **Traffic Forecasting:** **PyTorch GRU** (`0.07641` avg MAE) > **LightGBM** (`0.08079`) > **Graph+GRU** (`0.08512`) > Historical Average (`0.10200`) > Persistence (`0.12715`).
- **Uncertainty & Calibration:** `P10–P90` coverage = **`79.66%`** (target 80%); Isotonic Brier score = **`0.05033`**.
- **Crisis & AQI:** Crisis LightGBM **PR-AUC = `0.9814`** (`F1 = 0.9085`); PM2.5 +1h **MAE = `2.373 µg/m³`** (beats Persistence `4.102`).
- **Routing (60 replayed trips):** CITYPULSE matched or beat Current-Traffic routing on **`61.7%`** of trips.

## 4. Answers to Common Viva / Teacher Questions
- **"Why not just use Google Maps?"**  
  Google Maps is a closed commercial black box. It does not expose raw uncertainty intervals (`P90 - P50`), custom ambulance risk penalties, clean-air (`PM2.5`) routing, or hospital-specific crisis warnings.
- **"Why a graph model?"**  
  Roads are connected in a network, not isolated lines. A jam on 5th Ave spills onto cross streets. A graph adjacency matrix lets sensors share traffic state with road-connected neighbours.
- **"Is routing itself Machine Learning?"**  
  No — **Dijkstra is a deterministic graph algorithm**, not ML. The **ML part** is predicting future edge travel times and uncertainty (`P90 - P50`), which Dijkstra then uses as edge weights.
- **"What if the deep/graph model is not better than LightGBM?"**  
  We report it honestly! In our run, `Graph+GRU` (`0.08512`) did not beat `LightGBM` (`0.08079`). On tabular features with strong lags and neighbour averages, gradient-boosted trees are often stronger and much faster than small neural networks.
