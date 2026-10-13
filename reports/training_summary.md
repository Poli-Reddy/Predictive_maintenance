# Training summary

Dataset rows: generated CSV.GZ

Production split: 960 train (1–560 plus extra 801–1200) / 120 validation (561–680) / 120 test (681–800) engines, chronological cohorts.
Feature count: 129 (service_hours_since_overhaul allowed as known operating-hours input; timestamp_min stays banned as its duplicate).
XGBoost training sample: sampled 324,000 rows from 1,080 train+validation engines for the final refit (dev: 288,000 from 960 train engines); validation and test rows are never sampled.
Operating hours are a legitimate known-at-inference input (tank logs hours, causal past-only); failure mode, environment and latent health stay excluded.
LSTM training-window cap: 80000; maximum epochs: 12; early-stopping patience: 3.
LSTM validation windows are used for early stopping/checkpoint selection, not gradient updates when refit_final is false; per-epoch window counts and selected epoch are in reports/lstm_training_history.json.
Preprocessing is fit only on training data during development and on the final sample for refit. Causal median-5 despiking precedes feature building in all paths.
LSTM refit_final: False.
XGBoost objective: reg:squarederror (RMSE target). Early-life weighting: RUL > 10 h ×2.
Kalman parameters are selected on validation RMSE (MAE tie-break); selected parameters: [0.01, 1e-05, 0.60]. Scalar 1-D update, first output equals raw measurement (matches online path).
XGBoost validation-only linear calibration: scale 1.02, offset +0.10 hours, RMSE-selected on validation tracked predictions.
Production params: depth 6, lr 0.03, 400 trees, min_child_weight 12, reg_lambda 10.0, colsample 0.7, subsample 0.8, max_bin 128. Booster 1,963,610 B UBJ; deploy package 1,985,758 B (~1.89 MiB, UBJ model).
Common test rows: 27,416. Within 1 h: XGBoost 42.50%, XGBoost + Kalman 42.66%, calibrated LSTM 48.29%.
Full test rows: 143,904. Production best (XGBoost + Kalman + cal): MAE 1.7229 h, RMSE 2.3148 h, MSE 5.3584, R² 86.27%, within 1 h 41.30%, within 2 h 68.58%.
The test partition is never used for model, calibration, or tracker selection. This cohort has been inspected in prior exploratory runs and is not a pristine one-shot benchmark.
All online features are causal and use only present/past telemetry.

## Validation-only Pareto search

Seven compact XGBoost configurations were compared with three configured causal Kalman settings on 27,089 validation endpoints matching the LSTM comparison cadence. Test engines were not used, and existing production model artifacts were not replaced. Candidate-level pooled and macro-engine metrics and Pareto status are in `reports/validation_pareto.csv` and `reports/validation_pareto.json`.

The current depth-6 absolute-error baseline with Kalman achieved MAE 1.5858 h, RMSE 2.1329 h, R² 86.42%, within-1h 44.10%, and within-2h 71.28%. A depth-8 absolute-error candidate (160 trees, min_child_weight 10) improved those validation measures to 1.5591 h, 2.1155 h, 86.64%, 44.91%, and 72.32%, respectively; its NASA-style score also improved from 4,138.25 to 4,071.47. Average per-engine MAE/RMSE and within-1h/within-2h also improved. Its UBJ booster is 2.69 MB versus 0.79 MB for the baseline, so the accuracy gain comes with roughly 3.4x booster size.

This is a promising validation candidate, not a final model selection: its result is measured on the same validation cohort used for screening. Do not replace the production artifact or claim test improvement until the choice is frozen and evaluated once on a new independent test cohort.
