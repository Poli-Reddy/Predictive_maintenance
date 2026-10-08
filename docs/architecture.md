# Production-oriented architecture

```text
                PUBLIC EVIDENCE + EXPLICIT SIMULATION ASSUMPTIONS
                                  |
                                  v
                        Synthetic Powerpack Generator
                                  |
                                  v
                 Raw Telemetry Data Contract + Validation
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
        Engine-disjoint production split     Audit / provenance
        (train / validation / test)          manifests + hashes
                 |
                 v
        Partition-local causal features
    (lags / rolling / EWMs / rates / ratios)
                 |
          +------+------+
          |             |
          v             v
    XGBoost RUL       LSTM baseline
          |             |
          |             v
          |       Validation early stopping
          |             |
          |             v
          |       Frozen train-only model
          |
   Validation-only model selection
          |
          v
   Frozen Kalman parameters
          |
          v
   Train+Val final XGBoost refit
          |
          v
      Frozen held-out test evaluation
          |
          +------------------+
                             |
                             v
              XGBoost RUL + causal Kalman RUL
                             |
                             v
                  Production edge inference
```

## Leakage controls

- Entire engine histories are assigned to one partition only.
- Test engines are never used for hyperparameter, feature, imputer, scaler or tracker selection.
- Feature windows are backward-looking and causal; no centered rolling windows or back-fills.
- `service_hours_since_overhaul` is excluded from both XGBoost and LSTM inputs because it is an age proxy for the synthetic RUL target.
- XGBoost also uses 120- and 360-minute exponentially weighted sensor levels and deviations; these are computed causally from present/past telemetry only.
- Cross-engine forward filling is prohibited.
- Training quantiles/medians are fitted only on development training data during selection.
- Final preprocessing is refit on train+validation only after all model-selection decisions are frozen.
- Kalman parameters are chosen using validation predictions only and then frozen before test.
- The current Kalman selection target is validation within-one-hour accuracy, with MAE/RMSE tie-breakers. The LSTM's optional output calibration is also fitted on validation predictions only and is recorded as a separate artifact.
- LSTM early stopping uses validation only. With the current `refit_final: false` setting, the selected train-only checkpoint is retained; train+validation refitting is optional and must be explicitly enabled.
- The LSTM training-window cap and epoch/patience limits are configured under `lstm` in `config/config.yaml`; history records the actual train/validation window counts and selected epoch.
- The test set is evaluated only after the final artifacts are frozen.
- Once reported, the test cohort is not used to choose another candidate; repeated exploratory test runs invalidate a strict one-shot final benchmark.
- XGBoost training-row sampling is controlled by `training_sampling.xgb_max_rows_per_engine`; a value of zero uses all training rows. Any sampling is applied only after full-resolution causal feature generation and never touches validation/test rows.
