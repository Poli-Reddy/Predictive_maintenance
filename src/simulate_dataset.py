from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Publicly documented powerpack-monitored channels used by the simulator.
# Source evidence and assumptions are recorded in docs/public_sensor_evidence.md.
SENSOR_COLUMNS = [
    "engine_rpm",
    "transmission_rpm",
    "engine_coolant_temp_c",
    "transmission_oil_temp_c",
    "engine_oil_pressure_bar",
    "clutch_pressure_bar",
    "solenoid_current_a",
    "vehicle_speed_kph",
]

OPERATING_MODES = {
    "idle": 0.05,
    "cruise": 0.42,
    "stop_go": 0.35,
    "acceleration": 0.72,
    "hill_climb": 0.88,
    "rough_terrain": 0.64,
}

ENVIRONMENTS = {
    # Environmental values here are simulation envelopes, not official Arjun Mk1 sensor specifications.
    # The extreme cold/hot/high-altitude profiles are stress-test cases informed by public reporting on
    # the later DATRAN engine program; see docs/public_sensor_evidence.md.
    "temperate": {"ambient_c": (15.0, 32.0), "altitude_m": (0, 1000), "dust": (0.05, 0.25)},
    "hot_desert": {"ambient_c": (35.0, 55.0), "altitude_m": (0, 1200), "dust": (0.45, 0.95)},
    "cold": {"ambient_c": (-40.0, 5.0), "altitude_m": (0, 2500), "dust": (0.02, 0.15)},
    "high_altitude": {"ambient_c": (-15.0, 22.0), "altitude_m": (2500, 5000), "dust": (0.05, 0.35)},
}

FAILURE_MODES = {
    # Latent failure modes shape the sensors. The failure mode is metadata only and is never an input feature.
    "thermal_aging": 1.00,
    "lubrication_degradation": 0.92,
    "transmission_wear": 0.88,
    "mixed_aging": 1.08,
}


@dataclass(frozen=True)
class EngineScenario:
    engine_id: int
    production_batch: int
    environment: str
    failure_mode: str
    lifetime_min: int
    seed: int
    severity: float


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def deterministic_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def choose_scenario(engine_id: int, batch: int, rng: np.random.Generator, lifetime_min: int, base_seed: int, severity: float) -> EngineScenario:
    environment = rng.choice(list(ENVIRONMENTS), p=[0.48, 0.24, 0.14, 0.14])
    failure_mode = rng.choice(list(FAILURE_MODES), p=[0.28, 0.27, 0.22, 0.23])
    return EngineScenario(
        engine_id=engine_id,
        production_batch=batch,
        environment=str(environment),
        failure_mode=str(failure_mode),
        lifetime_min=lifetime_min,
        seed=base_seed + engine_id * 7919,
        severity=float(severity),
    )


def operating_sequence(n: int, rng: np.random.Generator) -> np.ndarray:
    """Piecewise operating mode process with persistence, not random IID rows."""
    modes = list(OPERATING_MODES)
    out: list[str] = []
    current = "idle"
    remaining = 0
    transition_probs = {
        "idle": [0.20, 0.42, 0.16, 0.12, 0.05, 0.05],
        "cruise": [0.05, 0.55, 0.16, 0.10, 0.08, 0.06],
        "stop_go": [0.16, 0.42, 0.24, 0.10, 0.03, 0.05],
        "acceleration": [0.05, 0.45, 0.16, 0.10, 0.16, 0.08],
        "hill_climb": [0.03, 0.33, 0.09, 0.08, 0.40, 0.07],
        "rough_terrain": [0.05, 0.37, 0.14, 0.07, 0.13, 0.24],
    }
    while len(out) < n:
        if remaining <= 0:
            probs = transition_probs[current]
            current = str(rng.choice(modes, p=probs))
            mean_duration = {"idle": 18, "cruise": 35, "stop_go": 26, "acceleration": 8, "hill_climb": 25, "rough_terrain": 28}[current]
            remaining = int(max(4, rng.lognormal(math.log(mean_duration), 0.35)))
        take = min(remaining, n - len(out))
        out.extend([current] * take)
        remaining -= take
    return np.asarray(out, dtype=object)


def gaussian_ar1(n: int, rng: np.random.Generator, phi: float = 0.96, sigma: float = 1.0) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def simulate_engine(sc: EngineScenario) -> pd.DataFrame:
    rng = deterministic_rng(sc.seed)
    n = sc.lifetime_min
    t = np.arange(n, dtype=float)
    time_h = t / 60.0
    progress = np.clip(t / max(n - 1, 1), 0.0, 1.0)
    rul_h = np.clip((n - 1 - t) / 60.0, 0.0, None)

    env = ENVIRONMENTS[sc.environment]
    ambient = rng.uniform(*env["ambient_c"])
    ambient_walk = gaussian_ar1(n, rng, phi=0.997, sigma=0.12)
    ambient_series = np.clip(ambient + ambient_walk, env["ambient_c"][0], env["ambient_c"][1])
    altitude = rng.uniform(*env["altitude_m"])
    dust = rng.uniform(*env["dust"])

    modes = operating_sequence(n, rng)
    load = np.array([OPERATING_MODES[str(m)] for m in modes])

    # Production-batch drift represents calibration, manufacturing and fleet-generation variation.
    batch_drift = 1.0 + rng.normal(0.0, 0.015)
    sensor_bias = rng.normal(0.0, 1.0, size=len(SENSOR_COLUMNS))

    # Hidden health state is used only to generate physically coherent telemetry and is never saved.
    failure_strength = FAILURE_MODES[sc.failure_mode]
    # Faster-aging engines start drifting earlier; slower-aging engines remain closer to nominal until later life.
    # This creates a realistic sensor-to-RUL relationship without exposing the target or future life length.
    severity_shape = np.clip((sc.severity - 0.72) / 0.70, 0.0, 1.0)
    exponent = 1.95 - 0.55 * severity_shape + rng.normal(0, 0.04)
    nonlinear = progress ** max(1.25, exponent)
    cumulative_load = np.cumsum(load) / max(n, 1)
    health_loss = (
        0.73 * nonlinear
        + 0.14 * (cumulative_load ** 1.35)
        + 0.07 * progress * dust
        + 0.06 * np.maximum(0, altitude / 5000.0) * progress
    )
    health_loss *= failure_strength
    # Normalize only the final end-of-life scale; the shape still carries severity information.
    end_scale = max(float(health_loss[-1]), 1e-6)
    health_loss = np.clip(health_loss / end_scale, 0, 0.999)
    degradation_walk = gaussian_ar1(n, rng, phi=0.9985, sigma=0.0015)
    health_loss = np.clip(health_loss + np.maximum(0, degradation_walk.cumsum()) * 0.010, 0, 0.995)

    # Transient workload dynamics.
    demand = np.clip(load + gaussian_ar1(n, rng, phi=0.85, sigma=0.035), 0.0, 1.0)
    demand_change = np.diff(demand, prepend=demand[0])

    # Engine speed is bounded by the public rated value used as a simulation anchor.
    idle_rpm = 650.0 + rng.normal(0, 12)
    target_rpm = idle_rpm + demand * (2400.0 - idle_rpm) * (1.0 - 0.08 * health_loss)
    engine_rpm = target_rpm + 20 * np.sin(t / 13.0) + rng.normal(0, 16, n)
    engine_rpm += 55 * demand_change
    engine_rpm = np.clip(engine_rpm, 550.0, 2420.0)

    # Transmission output follows engine speed/load through a scenario-dependent ratio and slip.
    ratio = np.where(demand < 0.25, 0.52, np.where(demand < 0.55, 0.68, np.where(demand < 0.78, 0.80, 0.92)))
    slip = 1.0 - 0.03 * health_loss - 0.02 * (demand > 0.75)
    transmission_rpm = np.clip(engine_rpm * ratio * slip + rng.normal(0, 14, n), 0.0, 2450.0)

    # Vehicle speed is derived from rpm/load and gear-like ratios, then bounded around the public road-speed anchor.
    speed_factor = np.where(demand < 0.22, 0.02, np.where(demand < 0.50, 0.72, np.where(demand < 0.78, 0.82, 0.60)))
    vehicle_speed = np.clip(transmission_rpm / 2400.0 * 70.0 * speed_factor + rng.normal(0, 1.8, n), 0.0, 71.0)
    vehicle_speed = np.where(modes == "idle", np.maximum(vehicle_speed - 3.0, 0.0), vehicle_speed)

    # Temperatures rise with load, ambient conditions, and degradation-induced cooling inefficiency.
    heat_factor = 1.0 + 0.75 * health_loss + 0.15 * dust
    coolant = (
        73.0 + 34.0 * demand + 0.43 * (ambient_series - 20.0) + 7.0 * heat_factor * demand
        + 2.0 * np.sin(t / 25.0) + rng.normal(0, 1.25, n)
    )
    coolant = np.clip(coolant, 62.0, 125.0)

    trans_oil = (
        64.0 + 31.0 * demand + 0.35 * (ambient_series - 20.0) + 5.0 * (0.55 * health_loss + 0.7 * dust) * demand
        + rng.normal(0, 1.2, n)
    )
    trans_oil = np.clip(trans_oil, 55.0, 125.0)

    # Lubrication pressure decays with heat/degradation and depends on speed/load.
    oil_pressure = (
        2.1 + 3.2 * np.sqrt(np.clip(engine_rpm / 2400.0, 0, 1)) - 0.65 * health_loss
        - 0.012 * np.maximum(coolant - 95.0, 0)
        + rng.normal(0, 0.09, n)
    )
    oil_pressure *= batch_drift
    oil_pressure = np.clip(oil_pressure, 0.8, 6.2)

    # Clutch pressure depends mainly on commanded load and degradation/slip.
    clutch_pressure = 3.0 + 13.0 * demand + 1.5 * np.maximum(demand_change, 0) - 2.0 * health_loss * (0.3 + demand)
    clutch_pressure += rng.normal(0, 0.16, n)
    clutch_pressure = np.clip(clutch_pressure, 1.5, 18.5)

    # Solenoid current tracks duty/actuation; aging and dust increase current slightly.
    solenoid_current = 0.18 + 1.75 * np.clip(demand, 0, 1) + 0.28 * health_loss + 0.12 * dust + rng.normal(0, 0.045, n)
    solenoid_current += 0.08 * (np.abs(demand_change) > 0.10)
    solenoid_current = np.clip(solenoid_current, 0.05, 2.6)

    # Short sensor faults: dropout blocks, clipped spikes and recovery. All affect only telemetry, not labels.
    values = {
        "engine_rpm": engine_rpm,
        "transmission_rpm": transmission_rpm,
        "engine_coolant_temp_c": coolant,
        "transmission_oil_temp_c": trans_oil,
        "engine_oil_pressure_bar": oil_pressure,
        "clutch_pressure_bar": clutch_pressure,
        "solenoid_current_a": solenoid_current,
        "vehicle_speed_kph": vehicle_speed,
    }
    for idx, c in enumerate(SENSOR_COLUMNS):
        arr = values[c].copy()
        # ~0.15% independent missing points.
        miss = rng.random(n) < 0.0015
        arr[miss] = np.nan
        # 4-8 causal burst gaps per long engine history.
        n_blocks = int(rng.integers(1, 5))
        for _ in range(n_blocks):
            start = int(rng.integers(0, max(1, n - 20)))
            length = int(rng.integers(4, 18))
            arr[start : min(n, start + length)] = np.nan
        # Rare sensor spikes.
        spike = rng.random(n) < 0.0007
        if c in {"engine_oil_pressure_bar", "clutch_pressure_bar"}:
            arr[spike] *= rng.choice([0.45, 1.55])
        elif c.endswith("_temp_c"):
            arr[spike] += rng.choice([-12.0, 15.0], size=spike.sum())
        else:
            arr[spike] += rng.normal(0, 80 if "rpm" in c else 0.35, spike.sum())
        values[c] = arr

    # Direction is a derived operational state from motion; reverse is uncommon.
    reverse_event = np.zeros(n, dtype=np.int8)
    moving = vehicle_speed > 2.0
    reverse_event[moving] = (rng.random(moving.sum()) < 0.015).astype(np.int8)

    direction = np.where(reverse_event == 1, "reverse", np.where(moving, "forward", "stationary"))

    # Measurement uncertainty metadata is useful for audit but not model input.
    sensor_quality = np.clip(1.0 - 0.5 * (np.isnan(pd.DataFrame(values, columns=SENSOR_COLUMNS)).sum(axis=1).to_numpy()), 0, 1)

    frame = pd.DataFrame(values)
    frame.insert(0, "engine_id", sc.engine_id)
    frame.insert(1, "production_batch", sc.production_batch)
    frame.insert(2, "timestamp_min", t * 1.0)
    frame["environment"] = sc.environment
    frame["failure_mode_sim"] = sc.failure_mode
    frame["operation_mode_sim"] = modes
    frame["direction"] = direction
    frame["ambient_context_c"] = ambient_series
    frame["altitude_context_m"] = altitude
    frame["dust_context"] = dust
    frame["sensor_quality"] = sensor_quality
    frame["service_hours_since_overhaul"] = time_h.astype(np.float32)
    frame["rul_hours"] = rul_h
    return frame


def build_engine_manifest(cfg: dict) -> list[EngineScenario]:
    rng = deterministic_rng(int(cfg["project"]["seed"]))
    n_engines = int(cfg["dataset"]["n_engines"])
    min_steps = int(cfg["dataset"]["min_steps"])
    max_steps = int(cfg["dataset"]["max_steps"])
    scenarios: list[EngineScenario] = []
    nominal = 1160.0
    for eid in range(1, n_engines + 1):
        batch = (eid - 1) // 20
        # Severity is a latent unit-level property. It controls both life length and degradation shape.
        # The model never sees it; it only sees the resulting telemetry.
        severity = float(np.clip(rng.lognormal(mean=0.0, sigma=0.16), 0.68, 1.42))
        lifetime = int(np.clip(nominal / severity + rng.normal(0, 45), min_steps, max_steps))
        scenarios.append(choose_scenario(eid, batch, rng, lifetime, int(cfg["project"]["seed"]), severity))
    return scenarios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    out = root / cfg["dataset"]["output_path"]
    out.parent.mkdir(parents=True, exist_ok=True)

    scenarios = build_engine_manifest(cfg)
    frames: list[pd.DataFrame] = []
    manifest_rows = []
    for sc in scenarios:
        df = simulate_engine(sc)
        frames.append(df)
        manifest_rows.append({
            "engine_id": sc.engine_id,
            "production_batch": sc.production_batch,
            "environment": sc.environment,
            "failure_mode_sim": sc.failure_mode,
            "lifetime_min": sc.lifetime_min,
            "seed": sc.seed,
            "severity_latent": sc.severity,
        })
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["engine_id", "timestamp_min"]).reset_index(drop=True)

    # Stable dtypes for smaller, faster downstream training.
    for c in [*SENSOR_COLUMNS, "ambient_context_c", "altitude_context_m", "dust_context", "sensor_quality", "rul_hours", "timestamp_min"]:
        full[c] = full[c].astype("float32")
    full["engine_id"] = full["engine_id"].astype("int32")
    full["production_batch"] = full["production_batch"].astype("int16")

    full.to_csv(out, index=False, compression="gzip", float_format="%.6g")
    manifest_path = root / "data/processed/engine_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metadata = {
        "rows": int(len(full)),
        "engines": int(full.engine_id.nunique()),
        "sensors": SENSOR_COLUMNS,
        "environments": sorted(full.environment.unique().tolist()),
        "failure_modes": sorted(full.failure_mode_sim.unique().tolist()),
        "rul_definition": "RUL = simulated run-to-failure time minus current timestamp; no future telemetry is exposed to model features.",
        "simulation_only": True,
    }
    (root / "data/processed/dataset_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {out} ({out.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
