# Depth-8 XGBoost + Kalman candidate: test evaluation

## Decision

The depth-8 candidate was **not worse than the previous XGBoost + Kalman result**: after refitting on train+validation and applying the validation-selected Kalman settings, it improved every reported metric on both the common-cadence comparison and the full test set. It is a promising candidate for further evaluation, not a replacement backed by an independent final benchmark.

The candidate still trails the LSTM on the common-cadence comparison and has a larger booster. No production artifacts were replaced.

## Candidate and protocol

- XGBoost objective: `reg:absoluteerror`
- Maximum depth: 8
- Estimators: 160
- Learning rate: 0.05
- Minimum child weight: 10
- Kalman settings: `[0.002, 0.00001, 0.30]`, chosen during the earlier train/validation-only search
- Final refit: all 795,686 rows from the 560 train and 120 validation engines
- Test: all 143,904 rows from 120 engine-disjoint test engines
- Common-cadence comparison: 27,416 rows, matching LSTM test prediction timestamps
- Serialized candidate booster: 2,722,601 bytes in UBJ format, excluding preprocessing and Kalman artifacts

The candidate settings were chosen based on validation results, then refit using train+validation. The test partition was not used to select the settings or Kalman parameters. However, this same test cohort has been inspected in previous exploratory model runs; these results are therefore exploratory, not an untouched one-shot benchmark.

## Common-cadence comparison

| Model | MAE (h) ↓ | RMSE (h) ↓ | R² ↑ | Within 1 h ↑ | Within 2 h ↑ | NASA-style score ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Previous XGBoost + Kalman | 1.6810 | 2.3113 | 85.14% | 44.20% | 69.63% | 4,457.90 |
| Depth-8 XGBoost + Kalman candidate | **1.6456** | **2.2694** | **85.67%** | **45.22%** | **70.66%** | **4,358.63** |
| LSTM (validation-calibrated) | 1.5808 | 2.2283 | 86.19% | 48.29% | 71.93% | 4,069.70 |

Relative to the previous XGBoost + Kalman model, the candidate improves all six common-cadence metrics: MAE decreases by 0.0354 h, RMSE by 0.0419 h, R² rises by 0.53 percentage points, within-one-hour by 1.02 points, within-two-hours by 1.02 points, and NASA-style score decreases by 99.27.

The candidate does **not** beat the LSTM on this comparison.

## Full-test XGBoost + Kalman

| Metric | Previous XGBoost + Kalman | Depth-8 candidate |
|---|---:|---:|
| Rows | 143,904 | 143,904 |
| MAE (h) | 1.7497 | **1.7124** |
| RMSE (h) | 2.4069 | **2.3610** |
| R² | 85.16% | **85.72%** |
| Within 1 h | 43.02% | **43.98%** |
| Within 2 h | 68.07% | **69.06%** |
| NASA-style score | 24,466.90 | **23,930.75** |

All six full-test metrics improved over the previous XGBoost + Kalman result.

## Size and promotion caveat

The candidate's booster is approximately 2.60 MiB in UBJ format, larger than the prior approximately 0.79 MB validation booster. The previous complete serialized production inference artifacts totalled about 1.25 MiB; a deployable package size for this candidate was not built or measured, so the booster-only size is not directly comparable to that complete-package figure.

Keep the current production artifact until this candidate is evaluated once on a newly generated/frozen independent cohort and deployment package size, runtime memory, and target-hardware latency are measured. Synthetic benchmark improvements are not evidence of field performance.
