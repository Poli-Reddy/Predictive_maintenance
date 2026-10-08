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
        A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        Q = np.diag([self.q_pos, self.q_vel])
        R = np.array([[self.measurement_var]], dtype=np.float64)
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q
        residual = np.array([[float(measurement)]]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ residual).ravel()
        self.P = (np.eye(2) - K @ H) @ self.P
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
            prev_engine = eid
        out[i] = tracker.update(float(pred))
    return out
