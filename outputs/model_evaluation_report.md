# NYC Traffic Speed Model Evaluation Report (`speed_model_v1`)

## 1. Gate Check
- **Requirement**: LightGBM MUST beat Historical Average MAE on the held-out test set.
- **+15m Test MAE**: LightGBM `3.2885` mph vs Historical Average `5.7855` mph (**PASS**)
- **Mean Test MAE (+15..+60m)**: LightGBM `3.8849` mph vs Historical Average `5.9835` mph (**PASS**)
- **Overall Gate Status**: **PASS**

## 2. Chronological Data Split
| Split | Start | End | Rows |
|---|---|---|---:|
| Train | `2018-10-01 02:45:00-04:00` | `2019-02-05 09:00:00-05:00` | 1,515,528 |
| Val | `2019-02-05 09:15:00-05:00` | `2019-03-04 15:45:00-05:00` | 324,756 |
| Test | `2019-03-04 16:00:00-05:00` | `2019-03-31 23:45:00-04:00` | 324,880 |

## 3. Test Set MAE Comparison (mph)
| Model | +15m MAE | +30m MAE | +45m MAE | +60m MAE | Mean MAE |
|---|---:|---:|---:|---:|---:|
| LightGBM | 3.2885 | 3.7661 | 4.1118 | 4.3731 | 3.8849 |
| Persistence | 3.8096 | 4.5268 | 5.0954 | 5.6052 | 4.7592 |
| Historical Average | 5.7855 | 5.8576 | 6.0271 | 6.2636 | 5.9835 |
| Seasonal Naive | 5.8913 | 6.1542 | 6.4454 | 6.7496 | 6.3101 |

## 4. Calibration & Uncertainty (Test Set, +15m)
- **Congestion Definition**: `speed < 60%` of segment train 85th-percentile free-flow speed (test base rate: `22.21%`)
- **Brier Score**: `0.08056` (uncalibrated) -> **`0.06453`** (Isotonic Regression)
- **ECE (10 bins)**: `0.08429` (uncalibrated) -> **`0.00509`** (Isotonic Regression)
- **ROC-AUC (calibrated)**: `0.9500`
- **P90 Quantile Empirical Coverage**: **`88.73%`** (target: `90.00%`)

## 5. Top 10 Features (`y_15` LightGBM Split Importance)
| Feature | Importance |
|---|---:|
| `segment_idx` | 4,268 |
| `speed_lag_96` | 2,451 |
| `speed_lag_672` | 2,313 |
| `speed_lag_1` | 2,192 |
| `hour` | 1,934 |
| `speed_delta_15m` | 1,909 |
| `hour_sin` | 1,492 |
| `network_mean_speed_lag_1` | 1,467 |
| `collisions_roll_3h` | 1,417 |
| `speed_delta_1h` | 1,358 |

## 6. Artifacts Generated
- Model Bundle: `models/speed_model_v1.pkl`
- Reliability Plot: `outputs/reliability_diagram.png`
- Horizon Comparison: `outputs/horizon_mae_comparison.png`
- P90 Uncertainty Band: `outputs/p90_uncertainty_band.png`
