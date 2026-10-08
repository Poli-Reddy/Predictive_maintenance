# Training summary

Dataset rows: generated CSV.GZ

Production split: 560 train / 120 validation / 120 test engines.
Feature count: 128.
XGBoost training sample: all training rows after full-resolution causal feature generation.
service_hours_since_overhaul is excluded from XGBoost and LSTM inputs as an age proxy.
LSTM training-window cap: 80000; maximum epochs: 12; early-stopping patience: 3.
LSTM validation windows are used for early stopping/checkpoint selection, not gradient updates when refit_final is false; per-epoch window counts and selected epoch are in reports/lstm_training_history.json.
Preprocessing is fit only on training data during development and on train+validation for final refit.
LSTM refit_final: False.
XGBoost objective: reg:absoluteerror.
Kalman parameters are selected on validation within_1h_percent (MAE/RMSE tie-breakers); selected parameters: [0.002, 0.00001, 0.30].
LSTM validation-only calibration: scale 0.975, offset -0.45 hours; this improves within-1h accuracy but trades off MAE/RMSE/R2.
Common test rows: 27,416. Within 1 h: XGBoost 43.22%, XGBoost + Kalman 44.20%, calibrated LSTM 48.29%.
Full test rows: 143,904. XGBoost + Kalman within 1 h: 43.02%.
The test partition is never used for model, calibration, or tracker selection. This cohort has been inspected in prior exploratory runs and is not a pristine one-shot benchmark.
All online features are causal and use only present/past telemetry.
