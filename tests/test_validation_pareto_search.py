from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validation_pareto_search import common_sequence_cadence_mask, pareto_front


def test_pareto_front_keeps_tradeoffs_and_drops_dominated_candidate():
    rows = [
        {"MAE_hours": 1.0, "RMSE_hours": 1.0, "nasa_phm08_score": 1.0,
         "macro_engine_MAE_hours": 1.0, "macro_engine_RMSE_hours": 1.0,
         "macro_engine_nasa_phm08_score": 1.0, "within_1h_percent": 80.0,
         "within_2h_percent": 90.0, "R2": 0.9,
         "macro_engine_within_1h_percent": 80.0, "macro_engine_within_2h_percent": 90.0,
         "booster_size_bytes": 100},
        {"MAE_hours": 1.1, "RMSE_hours": 1.1, "nasa_phm08_score": 1.1,
         "macro_engine_MAE_hours": 1.1, "macro_engine_RMSE_hours": 1.1,
         "macro_engine_nasa_phm08_score": 1.1, "within_1h_percent": 79.0,
         "within_2h_percent": 89.0, "R2": 0.89,
         "macro_engine_within_1h_percent": 79.0, "macro_engine_within_2h_percent": 89.0,
         "booster_size_bytes": 110},
        {"MAE_hours": 0.9, "RMSE_hours": 1.2, "nasa_phm08_score": 0.9,
         "macro_engine_MAE_hours": 0.9, "macro_engine_RMSE_hours": 1.2,
         "macro_engine_nasa_phm08_score": 0.9, "within_1h_percent": 81.0,
         "within_2h_percent": 91.0, "R2": 0.91,
         "macro_engine_within_1h_percent": 81.0, "macro_engine_within_2h_percent": 91.0,
         "booster_size_bytes": 90},
    ]
    assert pareto_front(rows) == [True, False, True]


def test_common_sequence_cadence_resets_per_engine():
    validation = pd.DataFrame({"engine_id": [1] * 8 + [2] * 8})
    mask = common_sequence_cadence_mask(validation, sequence_length=3, stride=2)
    assert np.flatnonzero(mask).tolist() == [2, 4, 6, 10, 12, 14]
