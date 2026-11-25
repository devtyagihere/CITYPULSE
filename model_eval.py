"""
model_eval.py — Stage 4: Test-set evaluation, Gate Check, Report & Reliability Plots.

Gate check:
  LightGBM MUST beat Historical Average MAE on the held-out test set.

Exports:
  - outputs/model_evaluation_report.md
  - outputs/reliability_diagram.png
  - outputs/horizon_mae_comparison.png
  - outputs/p90_uncertainty_band.png
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from baselines import HORIZONS, compute_baselines, get_time_splits

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "speed_model_v1.pkl"
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    models = bundle["models"]
    p90_model = bundle["p90_model"]
    calibrator = bundle["calibrator"]
    free_flow_map = bundle["free_flow_map"]
    congestion_ratio = bundle["congestion_ratio"]
    feature_cols = bundle["feature_cols"]
    split_info = bundle["split_info"]

    feat = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")
    df = feat.merge(labels, on=["segment_id", "timestamp"])

    train_df, val_df, test_df, _ = get_time_splits(df)

    # Load or compute baselines
    baseline_csv = OUTPUTS_DIR / "baseline_metrics.csv"
    if baseline_csv.exists():
        base_df = pd.read_csv(baseline_csv)
    else:
        base_df = compute_baselines(train_df, val_df, test_df)
        base_df.to_csv(baseline_csv, index=False)

    test_base = base_df[base_df["split"] == "test"].copy()

    # Evaluate LightGBM on test across all horizons
    lgb_rows = []
    for horizon in HORIZONS:
        te = test_df[test_df[horizon].notna()]
        y_true = te[horizon].to_numpy()
        y_pred = models[horizon].predict(te[feature_cols])
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        lgb_rows.append(
            {
                "split": "test",
                "model": "LightGBM",
                "horizon": horizon,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n_samples": len(y_true),
            }
        )
    lgb_df = pd.DataFrame(lgb_rows)
    all_test = pd.concat([test_base, lgb_df], ignore_index=True)

    # ---- Gate Check: LightGBM MUST beat Historical Average MAE on test ----
    lgb_mae_15 = float(lgb_df.loc[lgb_df["horizon"] == "y_15", "mae"].iloc[0])
    hist_mae_15 = float(
        test_base.loc[
            (test_base["model"] == "Historical Average") & (test_base["horizon"] == "y_15"),
            "mae",
        ].iloc[0]
    )
    lgb_mean_mae = float(lgb_df["mae"].mean())
    hist_mean_mae = float(
        test_base.loc[test_base["model"] == "Historical Average", "mae"].mean()
    )

    gate_15_pass = lgb_mae_15 < hist_mae_15
    gate_mean_pass = lgb_mean_mae < hist_mean_mae
    gate_pass = gate_15_pass and gate_mean_pass

    print("\n" + "=" * 65)
    print(f"  GATE CHECK: LightGBM (+15m MAE={lgb_mae_15:.4f}) vs Historical Avg ({hist_mae_15:.4f})")
    print(f"  GATE CHECK: LightGBM (Mean MAE={lgb_mean_mae:.4f}) vs Historical Avg ({hist_mean_mae:.4f})")
    print(f"  STATUS    : {'PASS' if gate_pass else 'FAIL'}")
    print("=" * 65)

    # ---- Calibration & P90 evaluation on test (y_15) ----
    te15 = test_df[test_df["y_15"].notna()].copy()
    y15_true = te15["y_15"].to_numpy()
    y15_pred = models["y_15"].predict(te15[feature_cols])
    y15_p90 = p90_model.predict(te15[feature_cols])
    p90_coverage = float(np.mean(y15_true <= y15_p90))

    te_ff = te15["segment_id"].map(free_flow_map).fillna(25.0).to_numpy()
    y_cong = (y15_true < congestion_ratio * te_ff).astype(int)
    raw_prob = np.clip(1.0 - (y15_pred / te_ff), 0.0, 1.0)
    cal_prob = calibrator.predict(raw_prob)

    brier_raw = float(brier_score_loss(y_cong, raw_prob))
    brier_cal = float(brier_score_loss(y_cong, cal_prob))
    ece_raw = compute_ece(y_cong, raw_prob)
    ece_cal = compute_ece(y_cong, cal_prob)
    auc_cal = float(roc_auc_score(y_cong, cal_prob)) if y_cong.sum() > 0 else float("nan")

    # ---- Plot 1: Reliability Diagram ----
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    prob_true_raw, prob_pred_raw = calibration_curve(y_cong, raw_prob, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_cong, cal_prob, n_bins=10, strategy="uniform")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.plot(prob_pred_raw, prob_true_raw, "o-", alpha=0.75, label=f"Uncalibrated (Brier={brier_raw:.4f})")
    ax.plot(prob_pred_cal, prob_true_cal, "s-", linewidth=2, label=f"Isotonic Calibrated (Brier={brier_cal:.4f})")
    ax.set_xlabel("Predicted Congestion Probability")
    ax.set_ylabel("Observed Congestion Frequency")
    ax.set_title("Reliability Diagram — Test Set (+15m Congestion)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    rel_plot = OUTPUTS_DIR / "reliability_diagram.png"
    fig.savefig(rel_plot, dpi=150)
    plt.close(fig)

    # ---- Plot 2: Horizon MAE Comparison ----
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    horizon_labels = {"y_15": "+15m", "y_30": "+30m", "y_45": "+45m", "y_60": "+60m"}
    for m_name in ["Persistence", "Seasonal Naive", "Historical Average", "LightGBM"]:
        sub = all_test[all_test["model"] == m_name].set_index("horizon").loc[HORIZONS]
        lw = 2.5 if m_name == "LightGBM" else 1.5
        ax.plot([horizon_labels[h] for h in HORIZONS], sub["mae"], marker="o", linewidth=lw, label=m_name)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Test MAE (mph)")
    ax.set_title("Test MAE by Horizon: LightGBM vs Baselines")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    mae_plot = OUTPUTS_DIR / "horizon_mae_comparison.png"
    fig.savefig(mae_plot, dpi=150)
    plt.close(fig)

    # ---- Plot 3: P90 Uncertainty Band on sample segment ----
    seg0 = te15["segment_id"].value_counts().idxmax()
    mask0 = (te15["segment_id"] == seg0).to_numpy()
    sub_ts = te15.loc[mask0, "timestamp"].iloc[:96]  # 24 hours
    sub_y = y15_true[mask0][:96]
    sub_pred = y15_pred[mask0][:96]
    sub_p90 = y15_p90[mask0][:96]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sub_ts, sub_y, color="black", linewidth=1.3, label="Actual Speed (mph)")
    ax.plot(sub_ts, sub_pred, color="#1f77b4", linewidth=1.3, label="LightGBM Forecast (+15m)")
    ax.plot(sub_ts, sub_p90, color="#ff7f0e", linestyle="--", linewidth=1.2, label="P90 Upper Bound")
    ax.fill_between(sub_ts, sub_pred, sub_p90, color="#ff7f0e", alpha=0.18)
    ax.set_title(f"Test Forecast & P90 Bound — Segment {seg0} (24h Sample)")
    ax.set_ylabel("Speed (mph)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    p90_plot = OUTPUTS_DIR / "p90_uncertainty_band.png"
    fig.savefig(p90_plot, dpi=150)
    plt.close(fig)

    # ---- Top Feature Importances ----
    imp = pd.Series(models["y_15"].feature_importances_, index=feature_cols).sort_values(ascending=False)
    top10_md = "\n".join(f"| `{k}` | {int(v):,} |" for k, v in imp.head(10).items())

    # ---- Markdown Evaluation Report ----
    pivot_mae = all_test.pivot(index="model", columns="horizon", values="mae")[HORIZONS]
    pivot_mae["Mean_MAE"] = pivot_mae.mean(axis=1)
    pivot_mae = pivot_mae.sort_values("Mean_MAE")

    table_lines = ["| Model | +15m MAE | +30m MAE | +45m MAE | +60m MAE | Mean MAE |",
                   "|---|---:|---:|---:|---:|---:|"]
    for m_name, row in pivot_mae.iterrows():
        table_lines.append(
            f"| {m_name} | {row['y_15']:.4f} | {row['y_30']:.4f} | {row['y_45']:.4f} | {row['y_60']:.4f} | {row['Mean_MAE']:.4f} |"
        )
    table_md = "\n".join(table_lines)

    report_md = f"""# NYC Traffic Speed Model Evaluation Report (`speed_model_v1`)

## 1. Gate Check
- **Requirement**: LightGBM MUST beat Historical Average MAE on the held-out test set.
- **+15m Test MAE**: LightGBM `{lgb_mae_15:.4f}` mph vs Historical Average `{hist_mae_15:.4f}` mph (**{"PASS" if gate_15_pass else "FAIL"}**)
- **Mean Test MAE (+15..+60m)**: LightGBM `{lgb_mean_mae:.4f}` mph vs Historical Average `{hist_mean_mae:.4f}` mph (**{"PASS" if gate_mean_pass else "FAIL"}**)
- **Overall Gate Status**: **{"PASS" if gate_pass else "FAIL"}**

## 2. Chronological Data Split
| Split | Start | End | Rows |
|---|---|---|---:|
| Train | `{split_info['train_start']}` | `{split_info['train_end']}` | {split_info['n_train']:,} |
| Val | `{split_info['val_start']}` | `{split_info['val_end']}` | {split_info['n_val']:,} |
| Test | `{split_info['test_start']}` | `{split_info['test_end']}` | {split_info['n_test']:,} |

## 3. Test Set MAE Comparison (mph)
{table_md}

## 4. Calibration & Uncertainty (Test Set, +15m)
- **Congestion Definition**: `speed < {congestion_ratio:.0%}` of segment train 85th-percentile free-flow speed (test base rate: `{y_cong.mean():.2%}`)
- **Brier Score**: `{brier_raw:.5f}` (uncalibrated) -> **`{brier_cal:.5f}`** (Isotonic Regression)
- **ECE (10 bins)**: `{ece_raw:.5f}` (uncalibrated) -> **`{ece_cal:.5f}`** (Isotonic Regression)
- **ROC-AUC (calibrated)**: `{auc_cal:.4f}`
- **P90 Quantile Empirical Coverage**: **`{p90_coverage:.2%}`** (target: `90.00%`)

## 5. Top 10 Features (`y_15` LightGBM Split Importance)
| Feature | Importance |
|---|---:|
{top10_md}

## 6. Artifacts Generated
- Model Bundle: `models/speed_model_v1.pkl`
- Reliability Plot: `outputs/reliability_diagram.png`
- Horizon Comparison: `outputs/horizon_mae_comparison.png`
- P90 Uncertainty Band: `outputs/p90_uncertainty_band.png`
"""
    report_path = OUTPUTS_DIR / "model_evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"[eval] Saved report -> {report_path}")
    print(f"[eval] Saved plots  -> {rel_plot.name}, {mae_plot.name}, {p90_plot.name}")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
