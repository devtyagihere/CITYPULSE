"""
explainer.py — Stage 5a: SHAP feature attribution for high-risk segments.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "speed_model_v1.pkl"


@lru_cache(maxsize=1)
def _get_explainer_bundle():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model_15 = bundle["models"]["y_15"]
    return shap.TreeExplainer(model_15), bundle["feature_cols"]


def explain_high_risk_segments(
    live_feat: pd.DataFrame,
    preds_df: pd.DataFrame,
    top_k_segments: int = 5,
    top_k_features: int = 5,
) -> List[Dict[str, object]]:
    """Compute TreeSHAP drivers for the highest congestion-risk segments (+15m)."""
    explainer, feature_cols = _get_explainer_bundle()

    ranked = preds_df.sort_values("congestion_prob_15", ascending=False).head(top_k_segments)
    idx_list = ranked.index.to_numpy()
    X_sub = live_feat.loc[idx_list, feature_cols]

    shap_vals = explainer.shap_values(X_sub)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    results = []
    for pos, row_idx in enumerate(idx_list):
        sv = shap_vals[pos]
        top_feat_idx = np.argsort(np.abs(sv))[::-1][:top_k_features]
        drivers = []
        for fi in top_feat_idx:
            fname = feature_cols[fi]
            val = float(X_sub.iloc[pos, fi])
            impact = float(sv[fi])
            drivers.append(
                {
                    "feature": fname,
                    "value": None if np.isnan(val) else round(val, 3),
                    "shap_mph": 0.0 if np.isnan(impact) else round(impact, 3),
                    "direction": "slower" if impact < 0 else "faster",
                }
            )

        row_pred = preds_df.loc[row_idx]
        results.append(
            {
                "segment_id": int(row_pred["segment_id"]),
                "pred_speed_15": float(row_pred["pred_speed_15"]),
                "speed_ratio_15": float(row_pred["speed_ratio_15"]),
                "congestion_prob_15": float(row_pred["congestion_prob_15"]),
                "drivers": drivers,
            }
        )
    return results
