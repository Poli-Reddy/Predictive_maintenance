# Final experiment report

## Dataset

- 1,200 independent simulated engine histories (800 original + 400 extra train-only engines 801–1200, same generator/seed stream; first 800 identical).
- 1,408,077 one-minute telemetry rows.
- Eight publicly documented powerpack monitoring channels are used as the core model inputs: engine RPM, transmission RPM, engine coolant temperature, transmission-oil temperature, engine-oil pressure, clutch pressure, solenoid current and vehicle speed.
- Additional simulation context/metadata is retained for audit; metadata such as failure mode and environment is excluded from model features.
- Multiple operating regimes, harsh-environment stress profiles, degradation modes, missing blocks and sensor spikes.

## Production split

Extended chronological engine-cohort split (test/val frozen from the original 800):
- 960 engines train (1–560 plus 801–1200)
- 120 engines validation (561–680)
- 120 engines test (681–800)

No engine occurs in more than one split. Model and Kalman selection use the validation engines. The LSTM's early stopping and its within-one-hour calibration also use validation predictions only.

## Training data and model selection

The trainer applies causal median-5 despiking, then computes causal features at full one-minute resolution. Operating hours (`service_hours_since_overhaul`) is allowed as a feature: it is known at inference (tank logs hours), causal past-only, and a single float. `timestamp_min` stays banned as its exact duplicate.

Final production model (all decisions on validation only, test frozen):
- 129 features, XGBoost `reg:squarederror`, depth 6, lr 0.03, 400 trees, `min_child_weight` 12, `reg_lambda` 10.0, `colsample_bytree` 0.7, `subsample` 0.8, `max_bin` 128, early-life weighting (RUL > 10 h ×2).
- Dev validation (142,273 rows): raw 1.6299/2.1464/87.40% → Kalman 1.6144/2.1286/87.61% → linear cal (scale 1.02, offset +0.10 h, RMSE-selected) 1.6353/2.1015/87.92%.
- Kalman `(0.01, 1e-05, 0.60)` selected on validation RMSE (MAE tie-break). Linear calibration fitted on validation tracked predictions only, stored in `models/prediction_calibration.json`.
- Final refit: sampled 324,000 rows from 1,080 train+validation engines. Booster 1,963,610 B UBJ (~1.87 MiB); full deploy package 1,985,758 B (~1.89 MiB) with UBJ model. Edge latency ~0.017 s per 360-row history (feat 0.0116 + XGB 0.0026 + Kalman 0.0032, dev machine; HIL still required).

The LSTM baseline is unchanged (train-only checkpoint, epoch 5, val-calibrated scale 0.975 / offset −0.45 h).

## XGBoost + Kalman full held-out test

One-shot-style evaluation on every one-minute row from the 120 held-out test engines (143,904 rows):

- Raw: MAE 1.7580 h, RMSE 2.3816 h, MSE 5.6722, R² 85.47%, within 1 h 41.55%, within 2 h 67.66%, NASA 24,866.20
- +Kalman: MAE 1.7459 h, RMSE 2.3691 h, MSE 5.6127, R² 85.62%, within 1 h 41.69%, within 2 h 67.97%, NASA 24,649.40
- +Kalman+cal (production best): MAE 1.7229 h, RMSE 2.3148 h, MSE 5.3584, R² 86.27%, within 1 h 41.30%, within 2 h 68.58%, NASA 24,876.70

This improves on the previous production result (1.7497/2.4069/85.16%): RMSE −3.8%, MSE −7.5%, R² +1.11pp.

## Common XGBoost/LSTM benchmark

For an apples-to-apples comparison, the LSTM benchmark evaluates every 5th minute of each test engine after its 60-minute sequence warm-up; the same timestamps are selected from XGBoost predictions.

| Model | MAE (h) | RMSE (h) | R² | Within 1 h | Within 2 h |
|---|---:|---:|---:|---:|---:|
| XGBoost | 1.6988 | 2.3056 | 85.21% | 42.50% | 69.15% |
| XGBoost + Kalman | 1.6850 | 2.2909 | 85.40% | 42.66% | 69.54% |
| LSTM (validation-calibrated) | 1.5808 | 2.2283 | 86.19% | 48.29% | 71.93% |

There are 27,416 common evaluation rows across all 120 test engines. XGBoost + Kalman does not outperform the calibrated LSTM on this comparison.

The 5-minute comparison cadence is fixed before evaluation to keep LSTM prediction cost bounded and is not selected using test results.

## Production inference

The online path is:

`Observed telemetry history -> causal median-5 despike -> causal feature builder -> train-fitted preprocessing -> XGBoost (UBJ) -> causal scalar Kalman tracker -> linear calibration -> RUL`

The production XGBoost + Kalman path does not require the LSTM. Hardware-in-the-loop measurement is required before making actual vehicle ECU latency/memory claims.

## Leakage controls and evaluation caveat

No centered windows, backfill, cross-engine filling, target-derived features, latent health variables, future timestamps, test-based preprocessing or test-based model/tracker selection is used. Despiking uses only the previous 5 observations of the same engine. Calibration and tracker choices are selected on validation engines only.

The held-out test cohort was evaluated in multiple exploratory pipeline runs during this optimization task (now a 3rd test touch including the extended-data push). Candidate choices were based on train/validation results, but repeated inspection means this cohort is no longer a pristine one-shot benchmark. Treat these test results as exploratory and use a newly generated, frozen cohort for a definitive final performance claim.
