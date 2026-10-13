from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd


@dataclass
class TrainFittedImputer:
    medians: dict[str, float]

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "TrainFittedImputer":
        med = X.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return cls({str(k): float(v) for k, v in med.items()})

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = list(self.medians)
        aligned = X.reindex(columns=cols).replace([np.inf, -np.inf], np.nan)
        for c, v in self.medians.items():
            aligned[c] = aligned[c].fillna(v)
        return aligned.astype(np.float32).to_numpy()


def fit_quantile_clipper(X: pd.DataFrame, low=0.005, high=0.995) -> tuple[pd.Series, pd.Series]:
    qlo = X.quantile(low, numeric_only=True)
    qhi = X.quantile(high, numeric_only=True)
    qlo = qlo.replace([np.inf, -np.inf], np.nan).fillna(X.min(numeric_only=True))
    qhi = qhi.replace([np.inf, -np.inf], np.nan).fillna(X.max(numeric_only=True))
    return qlo, qhi


def clip(X: pd.DataFrame, qlo: pd.Series, qhi: pd.Series) -> pd.DataFrame:
    cols = list(qlo.index)
    out = X.reindex(columns=cols).copy()
    return out.clip(qlo, qhi, axis="columns")


def save_preprocessor(directory, qlo, qhi, imputer, feature_columns):
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(qlo, directory / "xgb_qlo.joblib")
    joblib.dump(qhi, directory / "xgb_qhi.joblib")
    joblib.dump(imputer, directory / "xgb_imputer.joblib")
    joblib.dump(feature_columns, directory / "feature_columns.joblib")


SPIKE_THRESHOLDS = {
    "engine_rpm": 120.0, "transmission_rpm": 120.0,
    "engine_coolant_temp_c": 8.0, "transmission_oil_temp_c": 8.0,
    "engine_oil_pressure_bar": 0.8, "clutch_pressure_bar": 1.2,
    "solenoid_current_a": 0.3, "vehicle_speed_kph": 8.0,
}


def causal_despike(raw: pd.DataFrame) -> pd.DataFrame:
    """Causal median-5 spike rejection. Past-only, per engine, C-trivial (sort 5).
    Replaces |x - median(prev5)| > thresh with median. Preserves NaNs."""
    from features import CORE_SENSORS
    df = raw.sort_values(["engine_id", "timestamp_min"]).reset_index(drop=True)
    out = df.copy()
    for c in CORE_SENSORS:
        if c not in df.columns:
            continue
        th = SPIKE_THRESHOLDS.get(c)
        if th is None:
            continue
        med = df.groupby("engine_id")[c].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).median()
        ).to_numpy(dtype=np.float32)
        x = df[c].to_numpy(dtype=np.float32)
        bad = np.isfinite(x) & np.isfinite(med) & (np.abs(x - med) > th)
        xc = x.copy()
        xc[bad] = med[bad]
        out[c] = xc.astype(np.float32)
    return out
