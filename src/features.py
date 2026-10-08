from __future__ import annotations

import numpy as np
import pandas as pd

CORE_SENSORS = [
    "engine_rpm",
    "transmission_rpm",
    "engine_coolant_temp_c",
    "transmission_oil_temp_c",
    "engine_oil_pressure_bar",
    "clutch_pressure_bar",
    "solenoid_current_a",
    "vehicle_speed_kph",
]

BANNED_FEATURES = {
    "rul_hours", "failure_mode_sim", "latent_damage", "health_state", "future_rul",
    "environment", "operation_mode_sim", "direction", "engine_id", "timestamp_min",
    "production_batch", "ambient_context_c", "altitude_context_m", "dust_context", "sensor_quality",
    "service_hours_since_overhaul",
}


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _rolling_stats(x: np.ndarray, w: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x)
    xv = np.where(valid, x, 0.0).astype(np.float64)
    cv = valid.astype(np.int32)
    cs = np.concatenate(([0.0], np.cumsum(xv)))
    c2 = np.concatenate(([0.0], np.cumsum(xv * xv)))
    cc = np.concatenate(([0], np.cumsum(cv)))
    end = np.arange(1, len(x) + 1)
    start = np.maximum(0, end - w)
    sums = cs[end] - cs[start]
    sums2 = c2[end] - c2[start]
    counts = cc[end] - cc[start]
    mean = np.full(len(x), np.nan, dtype=np.float32)
    std = np.full(len(x), np.nan, dtype=np.float32)
    ok = counts >= min_periods
    mean[ok] = (sums[ok] / counts[ok]).astype(np.float32)
    var_ok = counts > 1
    var = np.zeros(len(x), dtype=np.float64)
    var[var_ok] = (sums2[var_ok] - (sums[var_ok] ** 2) / counts[var_ok]) / (counts[var_ok] - 1)
    std[ok & var_ok] = np.sqrt(np.maximum(var[ok & var_ok], 0.0)).astype(np.float32)
    return mean, std


def _build_one_engine(
    g: pd.DataFrame,
    lags: tuple[int, ...],
    windows: tuple[int, ...],
    ewm_spans: tuple[int, ...],
) -> pd.DataFrame:
    g = g.sort_values("timestamp_min").reset_index(drop=True).copy()
    out: dict[str, np.ndarray] = {}
    for c in CORE_SENSORS:
        x = g[c].to_numpy(np.float32)
        out[f"{c}__missing"] = np.isnan(x).astype(np.int8)
        for lag in lags:
            out[f"{c}__lag{lag}"] = _lag(x, lag)
        for w in windows:
            minp = max(3, int(np.ceil(w * 0.25)))
            mean, std = _rolling_stats(x, w, minp)
            out[f"{c}__mean{w}"] = mean
            out[f"{c}__std{w}"] = std
            lagw = _lag(x, w)
            out[f"{c}__slope{w}"] = (x - lagw) / float(w)
        for span in ewm_spans:
            minp = max(3, int(np.ceil(span * 0.1)))
            smooth = pd.Series(x).ewm(span=span, adjust=False, min_periods=minp).mean().to_numpy(dtype=np.float32)
            out[f"{c}__ewm{span}"] = smooth
            out[f"{c}__ewm_delta{span}"] = x - smooth

    # Causal physics-inspired features.
    rpm = g.engine_rpm.to_numpy(np.float32)
    trpm = g.transmission_rpm.to_numpy(np.float32)
    coolant = g.engine_coolant_temp_c.to_numpy(np.float32)
    toil = g.transmission_oil_temp_c.to_numpy(np.float32)
    op = g.engine_oil_pressure_bar.to_numpy(np.float32)
    cp = g.clutch_pressure_bar.to_numpy(np.float32)
    sol = g.solenoid_current_a.to_numpy(np.float32)
    speed = g.vehicle_speed_kph.to_numpy(np.float32)
    load_proxy = np.clip(rpm / 2400.0, 0, 1).astype(np.float32)
    denom = np.maximum(np.abs(rpm) / 1000.0, 0.35)
    d = {
        "engine_trans_rpm_gap": rpm - trpm,
        "coolant_trans_oil_delta_c": coolant - toil,
        "oil_pressure_per_krpm": op / denom,
        "clutch_pressure_per_krpm": cp / denom,
        "speed_per_krpm": speed / denom,
        "transmission_slip_proxy": (rpm - trpm) / np.maximum(np.abs(rpm), 300.0),
        "coolant_excess_vs_load": coolant - (73.0 + 34.0 * load_proxy),
        "trans_oil_excess_vs_load": toil - (64.0 + 31.0 * load_proxy),
        "oil_pressure_deficit_vs_rpm": op - (2.1 + 3.2 * np.sqrt(load_proxy)),
        "clutch_pressure_residual_vs_load": cp - (3.0 + 13.0 * load_proxy),
        "solenoid_residual_vs_load": sol - (0.18 + 1.75 * load_proxy),
        "rpm_accel_proxy": np.diff(np.diff(rpm, prepend=np.nan), prepend=np.nan),
        "coolant_rate": np.diff(coolant, prepend=np.nan),
        "oil_pressure_rate": np.diff(op, prepend=np.nan),
        "solenoid_rate": np.diff(sol, prepend=np.nan),
        "load_proxy": load_proxy,
    }
    out.update({k: np.asarray(v, dtype=np.float32) for k, v in d.items()})
    result = pd.concat([g, pd.DataFrame(out)], axis=1)
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].astype(np.float32)
    return result


def build_features(raw: pd.DataFrame, lags=(1, 15), windows=(15, 60), ewm_spans=(120, 360)) -> pd.DataFrame:
    """Strictly causal features; implemented per engine to prevent cross-engine state and accelerate generation."""
    required = {"engine_id", "timestamp_min", *CORE_SENSORS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    base = raw.sort_values(["engine_id", "timestamp_min"]).reset_index(drop=True)
    pieces = [
        _build_one_engine(g, tuple(lags), tuple(windows), tuple(ewm_spans))
        for _, g in base.groupby("engine_id", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True)


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    banned = BANNED_FEATURES | {"split"}
    return [c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]


def assert_no_future_dependency(raw_a: pd.DataFrame, raw_b: pd.DataFrame, probe_engine: int, probe_index: int) -> None:
    fa = build_features(raw_a[raw_a.engine_id == probe_engine])
    fb = build_features(raw_b[raw_b.engine_id == probe_engine])
    if probe_index >= len(fa):
        raise ValueError("probe_index out of range")
    cols = model_feature_columns(fa)
    aa = fa.loc[:probe_index, cols].fillna(0).to_numpy()
    bb = fb.loc[:probe_index, cols].fillna(0).to_numpy()
    if not np.allclose(aa, bb, atol=1e-6, rtol=1e-6):
        raise AssertionError("Feature builder is not causal")
