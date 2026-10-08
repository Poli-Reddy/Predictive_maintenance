from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def create_production_splits(engine_manifest: pd.DataFrame, cfg: dict) -> dict[str, list[int]]:
    """Chronological production-cohort split. Entire engines belong to one partition."""
    frac = cfg["split"]
    n = len(engine_manifest)
    n_train = int(n * float(frac["train_fraction"]))
    n_val = int(n * float(frac["validation_fraction"]))
    if n_train < 1 or n_val < 1 or n_train + n_val >= n:
        raise ValueError("Invalid split fractions")
    engines = engine_manifest.sort_values(["production_batch", "engine_id"])["engine_id"].astype(int).tolist()
    train_ids = engines[:n_train]
    val_ids = engines[n_train : n_train + n_val]
    test_ids = engines[n_train + n_val :]
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise AssertionError("Engine leakage across splits")
    return {"train": train_ids, "validation": val_ids, "test": test_ids}


def save_splits(splits: dict[str, list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path: Path) -> dict[str, list[int]]:
    return json.loads(path.read_text())
