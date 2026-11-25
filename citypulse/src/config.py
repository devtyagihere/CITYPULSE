# =============================================================================
# config.py — Central settings for the whole CITYPULSE project
# Change ONE thing here and everything else updates automatically.
# =============================================================================

import os

# ---- TOP SWITCH: flip to False when you have real data ----
USE_SYNTHETIC = False  # True = fake data,  False = real data

# ---- Real dataset paths (NYC 2018-10 to 2019-03) ----
WORKSPACE_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_TRAFFIC_PATH   = os.path.join(WORKSPACE_ROOT, "data", "processed", "canonical_15min.parquet")
REAL_WEATHER_PATH   = os.path.join(WORKSPACE_ROOT, "data", "raw", "weather_2018-10-01_2019-03-31.parquet")

# ---- Random seed so results are the same every time ----
RANDOM_SEED = 42

# ---- Project root (wherever this file lives, go one folder up) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Folder paths ----
DATA_RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR        = os.path.join(PROJECT_ROOT, "reports")

# ---- Synthetic data size (keep small so laptop is fast) ----
SYNTH_N_SENSORS    = 50          # number of fake road sensors
SYNTH_START        = "2022-01-01"
SYNTH_END          = "2022-06-30"
SYNTH_FREQ         = "15min"     # 15-minute intervals

# ---- Study area ----
STUDY_PLACE        = "Manhattan, New York City, New York, USA"

# ---- Time splits (by time, NOT random) ----
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
# TEST_FRAC  = 0.20  (whatever is left)

# ---- Timezone ----
TIMEZONE = "America/New_York"

# ---- Routing ----
ROUTE_UNCERTAINTY_LAMBDA = 0.5   # weight for P90-P50 uncertainty penalty

# ---- Crisis threshold ----
CRISIS_CONGESTED_RATIO   = 0.5   # speed_ratio below this = congested sensor
CRISIS_MIN_SHARE         = 0.30  # 30%+ sensors congested = potential crisis
CRISIS_MIN_DURATION_MIN  = 30    # must last 30 minutes

# ---- AQI breakpoints (PM2.5 µg/m³) ----
AQI_BREAKPOINTS = [
    (0,   12.0,  "Good"),
    (12.1, 35.4, "Moderate"),
    (35.5, 55.4, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4,"Unhealthy"),
    (150.5, 250.4,"Very Unhealthy"),
    (250.5, 500,  "Hazardous"),
]
