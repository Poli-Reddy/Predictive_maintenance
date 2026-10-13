from __future__ import annotations

import numpy as np


class CausalRULKalman:
    """Two-state [RUL, dRUL/dt] Kalman tracker. State update uses only current/past predictions."""

    def __init__(self, q_pos: float, q_vel: float, measurement_var: float):
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.measurement_var = float(measurement_var)
        self.x = None
        self.P = None

    def reset(self, initial_rul: float) -> None:
        self.x = np.array([max(float(initial_rul), 0.0), -1.0 / 60.0], dtype=np.float64)
        self.P = np.diag([4.0, 0.25]).astype(np.float64)

    def update(self, measurement: float) -> float:
        if self.x is None:
            self.reset(measurement)
            return float(max(measurement, 0.0))
        # Scalar 1-D update (S is 1x1): K = P H^T / S, no matrix inverse.
        # Predict
        # x = A x, P = A P A^T + Q with A=[[1,1],[0,1]]
        r0 = float(self.x[0] + self.x[1])
        r1 = float(self.x[1])
        p00, p01, p10, p11 = float(self.P[0, 0]), float(self.P[0, 1]), float(self.P[1, 0]), float(self.P[1, 1])
        q00, q11 = float(self.q_pos), float(self.q_vel)
        np00 = p00 + p01 + p10 + p11 + q00
        np01 = p01 + p11
        np10 = p10 + p11
        np11 = p11 + q11
        residual = float(measurement) - r0
        S = np00 + float(self.measurement_var)
        k0 = np00 / S
        k1 = np10 / S
        self.x = np.array([r0 + k0 * residual, r1 + k1 * residual], dtype=np.float64)
        self.P = np.array([[np00 - k0 * np00, np01 - k0 * np01], [np10 - k1 * np00, np11 - k1 * np01]], dtype=np.float64)
        self.x[0] = np.clip(self.x[0], 0.0, None)
        self.x[1] = min(self.x[1], 0.0)
        return float(self.x[0])


def track_by_engine(predictions: np.ndarray, engine_ids: np.ndarray, tracker_cfg: tuple[float, float, float]) -> np.ndarray:
    out = np.empty(len(predictions), dtype=np.float64)
    tracker = None
    prev_engine = None
    for i, (pred, eid) in enumerate(zip(predictions, engine_ids)):
        if tracker is None or eid != prev_engine:
            tracker = CausalRULKalman(*tracker_cfg)
            tracker.reset(float(pred))
            # First output equals raw measurement (matches predict_edge.py online
            # path where update() on fresh state returns measurement). Previously
            # this called update() after reset, adding an extra predict step.
            out[i] = float(max(float(pred), 0.0))
            prev_engine = eid
        else:
            out[i] = tracker.update(float(pred))
    return out
