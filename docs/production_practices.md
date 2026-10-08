# Production practices used

## Data
- Explicit schema and unit-bearing feature names.
- Deterministic generation with a recorded seed per engine.
- Engine-level identity and timestamp uniqueness checks.
- Per-engine monotonic time and target checks.
- Missing telemetry is preserved rather than hidden with future-aware imputation.

## Modeling
- Validation is separated from the final test set.
- The model-selection search is performed before final training.
- Preprocessing artifacts are serialized alongside the model.
- No feature is allowed to use the target, latent damage, failure mode, production metadata or future timestamp.
- `service_hours_since_overhaul` is excluded from both model inputs because it is a direct age proxy in the synthetic benchmark.
- XGBoost's 120- and 360-minute exponentially weighted sensor features use only current and past measurements.
- XGBoost is used for the lightweight production model; the LSTM is an alternative sequence baseline, not a production dependency.
- XGBoost training rows are controlled by `training_sampling.xgb_max_rows_per_engine` after full-resolution causal feature generation. A positive value applies deterministic per-engine sampling; zero uses all training rows. Validation and test rows are never sampled.
- LSTM training windows are deterministically thinned to `lstm.max_train_sequences` when necessary; validation windows remain unthinned and are used for early stopping/model selection.
- Kalman tracking is causal and stateful; it consumes only the current XGBoost measurement and its own previous state.
- XGBoost's objective is configurable; the current experiment uses absolute error. Kalman candidates are selected using validation within-one-hour accuracy with MAE/RMSE tie-breakers.
- LSTM output calibration is fitted on validation predictions only and serialized separately in `models/prediction_calibration.json`; it is not part of the XGBoost + Kalman production path.

## Evaluation
- MAE and RMSE report actual RUL error in hours.
- R² is reported as a variance-explained statistic and is not called classification accuracy.
- NASA PHM08-style asymmetric score is reported as an additional prognostic metric.
- A common-key comparison is used where XGBoost and LSTM predictions overlap.
- The test report is generated only after all parameters are frozen.
- A reported test cohort is treated as one-shot; repeated exploratory evaluations require a new frozen cohort before claiming independent final performance.
- `reports/xgb_results.json` records full-test XGBoost metrics, while `reports/comparison.csv` reports all models on the common XGBoost/LSTM test timestamps.

## Deployment
- Edge inference accepts telemetry history observed up to the current timestamp.
- The inference artifact contains the model, feature schema, preprocessing parameters and Kalman parameters.
- Cloud can remain the training/fleet analytics layer; the online RUL estimate does not require connectivity.
- Hardware-in-the-loop measurement is required before making actual vehicle ECU latency/memory claims.

## Why training uses fewer XGBoost rows than the raw dataset

Adjacent one-minute telemetry observations are highly correlated. The trainer first computes every causal feature at one-minute resolution. Sampling can cap the number of training observations per engine for faster experiments, or be disabled with `training_sampling.xgb_max_rows_per_engine: 0` for full-resolution training. The current configuration disables sampling and uses all training rows. No validation/test observations are reduced.

## Within-one-hour objective

When optimizing the fraction of predictions within one hour of the target, select tree/tracker parameters and any post-processing using validation data only. Report MAE, RMSE, R² and broader tolerance bands alongside the targeted metric, since improving within-one-hour accuracy can worsen other error measures. The current LSTM calibration has this tradeoff and is kept outside the production XGBoost + Kalman inference path.
