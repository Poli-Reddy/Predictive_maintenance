# Arjun Powerpack RUL Hybrid v3

A production-oriented research pipeline for **Remaining Useful Life (RUL) estimation** on an Arjun-like armoured-vehicle powerpack telemetry benchmark.

> **Important:** This repository does not contain or claim official/operational/classified Arjun telemetry. The public sources establish the monitored powerpack channel types and some platform anchors; the numerical telemetry histories are an explicitly documented synthetic benchmark.

## Research objective

Compare a sequence baseline (LSTM) against a lightweight engineering-oriented pipeline:

**causal temporal features -> XGBoost RUL estimator -> causal Kalman tracking**

The goal is to test whether a small model can achieve strong held-out prognostic accuracy without using a GPU or future observations during inference.

## Public basis

The CVRDE health-monitoring publication publicly identifies RTD temperature sensing, pressure sensing, solenoid current sensing, and magnetic pickups for the powerpack parameters used here. See `docs/public_sensor_evidence.md` for exact source links and the simulation boundary.

## Reproducibility

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate the dataset:

```bash
python src/simulate_dataset.py --config config/config.yaml
```

3. Validate the raw dataset:

```bash
python src/validate_dataset.py --config config/config.yaml
```

4. Run leakage/unit tests:

```bash
python -m pytest -q
```

5. Train the XGBoost+Kalman system and LSTM baseline:

```bash
python src/train_pipeline.py --config config/config.yaml
```

6. Rebuild the final comparison report:

```bash
python src/evaluate.py --config config/config.yaml
```

7. Benchmark edge inference:

```bash
python src/benchmark_edge.py
```

8. Run online inference on a single engine's observed history:

```bash
python src/predict_edge.py --history reports/edge_demo_history.csv
```

## Dataset design

- 800 independent engine histories.
- 720-1600 one-minute observations per engine.
- Publicly documented powerpack-monitoring channels only, plus simulation metadata.
- Multiple operating regimes: idle, cruise, stop-go, acceleration, hill climb, rough terrain.
- Multiple environmental profiles: temperate, hot-desert, cold, high-altitude stress testing.
- Multiple degradation/failure modes.
- Causal missing telemetry blocks and transient sensor spikes.
- Exact RUL from the simulated run-to-failure clock is the label.

## Production-style split

The split is by complete engine histories and follows a chronological production-cohort policy. An engine can appear in **exactly one** of train, validation or test.

Model selection and preprocessing fitting happen before the final test evaluation. The final model is retrained on train+validation after those decisions are frozen; the test engines remain untouched until evaluation.

## What this is NOT

- not an official Arjun sensor dataset;
- not a calibrated digital twin;
- not a proof of field/operational accuracy;
- not a certification or safety-critical deployment.

For a real deployment, authorized telemetry, sensor calibration/uncertainty characterization, failure definitions, fleet maintenance records, hardware-in-the-loop tests, and independent operational validation are required.
