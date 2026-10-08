# Final experiment report

## Dataset

- 800 independent simulated engine histories.
- 939,590 one-minute telemetry rows.
- Eight publicly documented powerpack monitoring channels are used as the core model inputs: engine RPM, transmission RPM, engine coolant temperature, transmission-oil temperature, engine-oil pressure, clutch pressure, solenoid current and vehicle speed.
- Additional simulation context/metadata is retained for audit; metadata such as failure mode and environment is excluded from model features.
- Multiple operating regimes, harsh-environment stress profiles, degradation modes, missing blocks and sensor spikes.

## Production split

Chronological engine-cohort split:
- 560 engines train
- 120 engines validation
- 120 engines test

No engine occurs in more than one split. Model and Kalman selection use the validation engines. The LSTM's early stopping and its within-one-hour calibration also use validation predictions only.

## Training data and model selection

The XGBoost trainer computes causal features at full one-minute resolution and uses all training rows (`xgb_max_rows_per_engine: 0`). The final XGBoost model is fit on 795,686 train+validation rows. Validation and test rows are never sampled. The numeric `service_hours_since_overhaul` field is excluded from both XGBoost and LSTM inputs because it is a strong age proxy for the simulated RUL target.

XGBoost uses the absolute-error objective. Kalman parameters are selected on validation predictions by maximizing within-one-hour accuracy, with MAE and RMSE as tie-breakers. The selected parameters are `[0.002, 0.00001, 0.30]`.

The LSTM can form 124,309 training windows from the train engines (sequence length 60, stride 5); the current cap uses 80,000. All 27,089 validation windows are evaluated each epoch for early stopping. Training stopped after epoch 8, retaining the best validation-loss checkpoint from epoch 5. Validation windows guide checkpoint selection but are not used for gradient updates with `refit_final: false`.

LSTM predictions are calibrated using validation-only scale 0.975 and offset −0.45 hours to optimize within-one-hour accuracy. The transform is stored in `models/prediction_calibration.json`. This improves the target metric at the cost of MAE/RMSE/R²; the tradeoff is reported rather than presenting the within-one-hour change alone.

## XGBoost + Kalman full held-out test

These metrics cover every one-minute row from the 120 held-out test engines (143,904 rows). The age proxy is excluded from both models.

- MAE: 1.7497 h
- RMSE: 2.4069 h
- R²: 85.16%
- Within 1 h: 43.02%
- Within 2 h: 68.07%
- NASA PHM08-style score: 24,466.90

The selected production tree uses 128 features, depth 6 and 160 estimators. The serialized XGBoost + preprocessing + Kalman artifacts total 1,311,536 bytes (~1.25 MiB), excluding the LSTM baseline. Deployment latency and runtime memory still require measurement on the target ECU.

## Common XGBoost/LSTM benchmark

For an apples-to-apples comparison, the LSTM benchmark evaluates every 5th minute of each test engine after its 60-minute sequence warm-up; the same timestamps are selected from XGBoost predictions.

| Model | MAE (h) | RMSE (h) | R² | Within 1 h | Within 2 h |
|---|---:|---:|---:|---:|---:|
| XGBoost | 1.7130 | 2.3494 | 84.65% | 43.22% | 68.81% |
| XGBoost + Kalman | 1.6810 | 2.3113 | 85.14% | 44.20% | 69.63% |
| LSTM (validation-calibrated) | 1.5808 | 2.2283 | 86.19% | 48.29% | 71.93% |

There are 27,416 common evaluation rows across all 120 test engines. XGBoost + Kalman does not outperform the calibrated LSTM on this comparison. Relative to the preceding exploratory run, within-one-hour accuracy increased from 42.96% to 44.20% for XGBoost + Kalman, and from 47.09% to 48.29% for LSTM. For LSTM, this gain trades against worse MAE/RMSE/R² than the earlier uncalibrated run.

The 5-minute comparison cadence is fixed before evaluation to keep LSTM prediction cost bounded and is not selected using test results.

## Production inference

The online path is:

`Observed telemetry history -> causal feature builder -> train-fitted preprocessing -> XGBoost -> causal Kalman tracker -> RUL`

The production XGBoost + Kalman path does not require the LSTM. Hardware-in-the-loop measurement is required before making actual vehicle ECU latency/memory claims.

## Leakage controls and evaluation caveat

No centered windows, backfill, cross-engine filling, target-derived features, latent health variables, future timestamps, test-based preprocessing or test-based model/tracker selection is used. Calibration and tracker choices are selected on validation engines only.

The held-out test cohort was evaluated in multiple exploratory pipeline runs during this optimization task. Candidate choices were based on train/validation results, but repeated inspection means this cohort is no longer a pristine one-shot benchmark. Treat these test results as exploratory and use a newly generated, frozen cohort for a definitive final performance claim.
