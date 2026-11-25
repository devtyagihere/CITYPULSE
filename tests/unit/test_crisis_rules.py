"""Unit tests for crisis_rules.py."""

import pandas as pd

from crisis_rules import evaluate_crisis_rules


def _make_preds(ratio: float, prob: float, n: int = 10) -> pd.DataFrame:
    data = {"segment_id": list(range(n))}
    for h in ("15", "30", "45", "60"):
        data[f"speed_ratio_{h}"] = [ratio] * n
        data[f"congestion_prob_{h}"] = [prob] * n
    return pd.DataFrame(data)


def test_crisis_rules_normal():
    df = _make_preds(ratio=0.90, prob=0.05)
    res = evaluate_crisis_rules(df, weather={"precip_mm": 0.0}, closures=[])
    assert res["crisis_type"] == "Normal"
    assert res["congestion_index"] < 25


def test_crisis_rules_gridlock():
    df = _make_preds(ratio=0.35, prob=0.85)
    res = evaluate_crisis_rules(df, weather={"precip_mm": 0.0}, closures=[])
    assert res["crisis_type"] == "Citywide Gridlock"
    assert res["congestion_index"] >= 70
    assert res["peak_congested_share"] == 1.0


def test_crisis_rules_weather_surge():
    df = _make_preds(ratio=0.55, prob=0.40)
    res = evaluate_crisis_rules(df, weather={"precip_mm": 5.0}, closures=[{"name": "5 Av"}])
    assert res["crisis_type"] in {"Weather/Incident Surge", "Citywide Gridlock"}
    assert res["precip_mm"] == 5.0
    assert res["active_closures"] == 1
