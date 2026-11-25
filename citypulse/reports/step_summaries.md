# CITYPULSE Step Summaries

---

## Step 01: Make Synthetic Data

**What we did:** Created realistic fake traffic, weather, and PM2.5 data for 50 sensors over 6 months (Jan–Jun 2022).

**What we got:** 3 CSV files in data/raw/ with the same column names as the real data, so swapping is easy.

**Key number:** Traffic table has **863,850 rows** (50 sensors × 17,277 time steps). About 2.0% values are intentionally missing.

**Note:** ⚠️ SYNTHETIC DATA — only for pipeline testing. Re-run with real data for real results.

---

## Step 02: Load and Check Data

**What we did:** Loaded all 3 datasets (traffic, weather, PM2.5), checked for missing values, and plotted one sensor for one week.

**What we got:** 50 traffic sensors, date range 2022-01-01 → 2022-06-30, speed missing = 2.0%.

**Key number:** Total traffic rows = **863,850** (50 sensors × time steps).

**Plot:** `reports/step02_sensor_week_plot.png`

---

## Step 03: Clean Data

**What we did:** Fixed timezones, removed impossible speeds, resampled to 15-min bins, filled gaps ≤1 hour, computed free-flow speed from train data, added speed_ratio.

**What we got:** Clean traffic with 863,850 rows, missing reduced from 2.3% to 0.0%.

**Key number:** **22.1%** of readings have speed_ratio < 0.5 (congested). Train cutoff: `2022-05-07 00:00:00-04:00`.

**Plot:** `reports/step03_cleaning_plots.png`

---

## Step 04: Build the Road Graph

**What we did:** Built Manhattan road graph. Source: *Synthetic Manhattan grid (OSM unavailable)*. Added speed limits and travel times to every road segment.

**What we got:** 2,400 nodes (intersections) and 9,380 edges (road segments).

**Key number:** Avg travel time per edge = **23.4 seconds**.

> NOTE: Synthetic grid used. Re-run with internet for real OSM graph.

**Plot:** `reports/step04_road_map.png`

---

## Step 05: Match Sensors to Road Edges

**What we did:** Assigned each of the 50 synthetic sensors to a real road edge on the Manhattan graph. For real data, GPS coordinates from `link_points` would be used instead.

**What we got:** A CSV table linking each sensor to its road edge (u→v node pair, road name, speed limit, travel time).

**Key number:** **50 sensors** matched to real Manhattan road edges out of 9,380 total edges.

**Plot:** `reports/step05_sensor_map.png`

---

## Step 06: Build Sensor Adjacency Matrix

**What we did:** Computed shortest road-travel-time distances between all 50 sensor pairs, then applied a Gaussian kernel (σ=395.4s) to convert distances into connection weights. Close sensors → high weight; far sensors → 0.

**What we got:** A 50×50 float32 weight matrix. 902 non-zero connections (63.9% sparse). Average 18.0 neighbours per sensor.

**Key number:** σ = **395.4 seconds** road-travel-time. This is the 'neighbourhood radius' — sensors reachable within about 395s of travel time are considered neighbours.

**Plot:** `reports/step06_adjacency_heatmap.png`

---

## Step 07: Data Leakage Test

**What we did:** Trained an honest model (past data only) and a leaky model (deliberately given future speed at t+15 min).

**What we got:** Honest MAE = **0.0861**, whereas Leaky MAE dropped unrealistically to **0.0016**.

**Key number:** Leakage caused a **98.1%** fake error drop — proving why we must strictly split by time and never touch future rows.

**Plot:** `reports/step07_leakage_test.png`

---

## Step 08: Build Features

**What we did:** Built past-only features for LightGBM (lags, rolling stats, train-only usual speed, weather, neighbour speed) and 4D sequence arrays (12 past steps → 4 future steps) for deep models, split 70/10/20 by time.

**What we got:** LightGBM train/val/test = 604,448 / 86,400 / 172,600 rows; Deep arrays train = (12082, 12, 50, 5).

**Key number:** **172,600 test rows** and **3,452 test sequence windows** — zero future leakage.

---

## Step 09: Baseline Models

**What we did:** Tested 3 simple baselines (Persistence, Historical Average, Seasonal Naive) on the test set across +15, +30, +45, and +60 min.

**What we got:** Saved MAE and RMSE benchmark table to `reports/step09_baselines_results.csv`.

**Key number:** Best baseline at +15m is **Historical_Average** with MAE = **0.0835** (target for ML models to beat!).

---

## Step 10: LightGBM Traffic Models

**What we did:** Trained 4 LightGBM models (+15, +30, +45, +60 min) using early stopping on validation MAE.

**What we got:** Test MAE = **0.0815** (+15m) up to **0.0717** (+60m). Most important feature: `hour`.

**Key number:** LightGBM +15m MAE = **0.0815**.

**Plot:** `reports/step10_feature_importance.png`

---

## Step 11: Simple GRU Model (PyTorch)

**What we did:** Trained a 1-layer PyTorch GRU on 12 past time steps to forecast +15, +30, +45, and +60 min ahead.

**What we got:** Best validation MAE = **0.0755**. Test +15m MAE = **0.0756**.

**Key number:** GRU +15m MAE = **0.0756**.

**Plot:** `reports/step11_gru_loss_curve.png`

---

## Step 12: Graph + GRU Model (PyTorch)

**What we did:** Combined graph convolution (using the road adjacency matrix) with a GRU and masked MAE loss.

**What we got:** Best validation MAE = **0.0835**. Test +15m MAE = **0.0840**.

**Key number:** Graph+GRU +15m MAE = **0.0840**.

**Plot:** `reports/step12_graph_gru_loss_curve.png`

---

## Step 13: Compare All Traffic Models

**What we did:** Compared all baselines, LightGBM, GRU, and Graph+GRU on the exact same test period across +15, +30, +45, and +60 min.

**What we got:** Overall winner is **GRU** (avg MAE = **0.0764**; LightGBM = 0.0808, GRU = 0.0764, Graph+GRU = 0.0851).

**Key number:** Best average MAE = **0.0764** (GRU).

**Plot:** `reports/step13_model_comparison.png`

---

## Step 14: Uncertainty Estimation (Quantile LightGBM)

**What we did:** Trained P10, P50, and P90 Quantile LightGBM models for +15 min predictions.

**What we got:** **79.7%** of true test values fell inside [P10, P90] (expected ~80%). Average band width = 0.2600.

**Key number:** Empirical 80% interval coverage = **79.7%**.

**Plot:** `reports/step14_uncertainty_band.png`

---

## Step 15: Congestion Probability Calibration

**What we did:** Fitted Isotonic Regression on validation predictions to map predicted speed_ratio into probability of congestion (< 0.5).

**What we got:** Test Brier score = **0.0503**; predicted mean probability (22.3%) matches observed rate (22.6%).

**Key number:** Brier score = **0.0503**.

**Plot:** `reports/step15_calibration_curve.png`

---

## Step 16: SHAP Explanation

**What we did:** Computed TreeSHAP values on 2,000 test samples for the +15 min LightGBM model.

**What we got:** Top drivers are `usual_speed_ratio`, `hour`, `weekday`, `neighbour_speed_ratio`, `roll_mean_1h`.

**Key number:** #1 driver is **`usual_speed_ratio`** (mean |SHAP| = **0.1916**).

**Plot:** `reports/step16_shap_summary.png`

---

## Step 17: Crisis Model & Congestion Index

**What we did:** Labeled historical city-wide crises (>30% sensors congested for >=30 min), trained a LightGBM crisis classifier + Isolation Forest anomaly detector, and computed a 0–100 Congestion Index.

**What we got:** Test PR-AUC = **0.9814**, F1 = **0.9085** (Precision=0.853, Recall=0.971). Anomalies flagged = 19.8%.

**Key number:** Crisis classifier PR-AUC = **0.9814**.

**Plot:** `reports/step17_crisis_and_congestion.png`

---

## Step 18: AQI (PM2.5) Prediction Model

**What we did:** Trained LightGBM models to forecast PM2.5 at +1h, +3h, and +6h using PM2.5 lags, traffic speed_ratio, wind, rain, and temperature.

**What we got:** +1h LightGBM MAE = **2.37 µg/m³** vs Persistence MAE = 4.10 µg/m³.

**Key number:** +1h PM2.5 MAE = **2.37 µg/m³**.

**Plot:** `reports/step18_aqi_shap.png`

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 20: Predicted Traffic & Uncertainty-Aware Routing

**What we did:** Applied predicted +15m travel times and uncertainty penalties (P90 - P50) to Manhattan edges, generating 3 alternative routes plus a cleanest-air route.

**What we got:** Best route predicted time = **90.29 min** (alternative 2 = 90.53 min).

**Key number:** Best predicted ambulance route = **90.29 minutes**.

**Plot:** `reports/step20_alternative_routes.png`

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 21: Route Replay Evaluation

**What we did:** Replayed 60 ambulance trips in the test period comparing Current-Traffic, CITYPULSE (predicted + uncertainty), and Oracle routes.

**What we got:** Mean times — Current = 35.20m, CITYPULSE = **35.18m**, Oracle = 35.08m (P90: 64.92m vs 64.92m).

**Key number:** CITYPULSE win/tie rate vs Current-Traffic = **61.7%**.

**Plot:** `reports/step21_example_routes.png`

---

## Step 22: Live Pipeline

**What we did:** Built a live collector checking `api.weather.gov` (KNYC), replaying test traffic in `Simulated Input` mode through `common.py`.

**What we got:** Saved `live_snapshot.json` with source, timestamp, and 'unavailable' handling for offline weather.

**Key number:** Congestion Index = **0.0/100**, PM2.5 +1h = **8.14 µg/m³**.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 07: Data Leakage Test

**What we did:** Trained an honest model (past data only) and a leaky model (deliberately given future speed at t+15 min).

**What we got:** Honest MAE = **0.0861**, whereas Leaky MAE dropped unrealistically to **0.0016**.

**Key number:** Leakage caused a **98.1%** fake error drop — proving why we must strictly split by time and never touch future rows.

**Plot:** `reports/step07_leakage_test.png`

---

## Step 22: Live Pipeline

**What we did:** Built a live collector checking `api.weather.gov` (KNYC), replaying test traffic in `Simulated Input` mode through `common.py`.

**What we got:** Saved `live_snapshot.json` with source, timestamp, and 'unavailable' handling for offline weather.

**Key number:** Congestion Index = **20.8/100**, PM2.5 +1h = **10.26 µg/m³**.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

## Step 19: Dijkstra From Scratch

**What we did:** Wrote Dijkstra's shortest-path algorithm from scratch using Python's `heapq` and tested it on 20 random Manhattan trips.

**What we got:** Exact match with NetworkX on all 20 trips (max difference = **0.000000 s**).

**Key number:** **20/20 trips matched** NetworkX with 0.000000s error.

---

