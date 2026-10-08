from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM08-style asymmetric score: early predictions are penalized differently from late ones."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    score_terms = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(np.sum(score_terms))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE_hours": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_hours": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2),
        "R2_percent": float(100.0 * r2),
        "within_1h_percent": float(100.0 * np.mean(np.abs(y_pred - y_true) <= 1.0)),
        "within_2h_percent": float(100.0 * np.mean(np.abs(y_pred - y_true) <= 2.0)),
        "nasa_phm08_score": nasa_phm08_score(y_true, y_pred),
    }
