# CITYPULSE — Live Traffic Forecasting, Crisis Detection & Uncertainty-Aware A* Routing

CITYPULSE is an end-to-end traffic speed forecasting, crisis monitoring, and time-dependent A\* routing system for Manhattan, NYC.

## Architecture

```text
[1. Ingest: ingest.py]
    NYC DOT Speeds + Open-Meteo Weather + NYPD Collisions + OSM Road Graph + US Holidays
        │
        ▼
[2. Features: features.py]
    Leakage-free lag/rolling/calendar/weather features (strictly < t)
        │
        ▼
[3. Predictor: predictor.py]
    LightGBM (+15/+30/+45/+60m) + P90 Quantile + Isotonic Calibration + Hist-Avg Fallback
        │
        ├──► [4. Crisis Rules: crisis_rules.py]  ──► 0–100 Congestion Index & Crisis Type
        ├──► [5a. Explainability: explainer.py]  ──► TreeSHAP top drivers for high-risk segments
        └──► [5b. Router: router.py]             ──► Time-Dependent A* with >10% reroute gate
                    │
                    ▼
[6. Orchestrator & Monitor: main.py, monitor.py] ──► outputs/live_snapshot.json
                    │
                    ▼
[7. API & Dashboard: api.py, static/index.html]   ──► FastAPI + Leaflet Dashboard
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run one live pipeline cycle:**
   ```bash
   python main.py
   ```
3. **Start the FastAPI server:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
   Open `http://127.0.0.1:8000` in your browser.

## API Endpoints

- `GET /` — Serves the live dashboard (`static/index.html`)
- `GET /api/health` — Health check and snapshot availability
- `GET /api/snapshot` — Returns the latest `outputs/live_snapshot.json` (or runs a cycle if missing)
- `POST /api/refresh` — Re-runs the full pipeline with optional `closures`, `origin_node`, `dest_node`, `lambda_penalty`, and `current_route_cost_s`

## Running Tests

```bash
pytest
```

## Docker

```bash
docker compose up --build
```
