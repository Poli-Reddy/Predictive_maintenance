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
