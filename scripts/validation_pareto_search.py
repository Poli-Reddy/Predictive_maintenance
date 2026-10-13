from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import build_features as build_feature_frame, model_feature_columns
from kalman import track_by_engine
from metrics import metric_dict
from preprocess import TrainFittedImputer, clip, fit_quantile_clipper
from splits import create_production_splits
from train_pipeline import deterministic_train_sample


MINIMIZE_METRICS = (
    "MAE_hours",
    "RMSE_hours",
    "nasa_phm08_score",
    "macro_engine_MAE_hours",
    "macro_engine_RMSE_hours",
    "macro_engine_nasa_phm08_score",
    "booster_size_bytes",
)
MAXIMIZE_METRICS = (
    "within_1h_percent",
    "within_2h_percent",
    "R2",
    "macro_engine_within_1h_percent",
    "macro_engine_within_2h_percent",
)


def pareto_front(rows: list[dict]) -> list[bool]:
    """Return non-dominated rows across pooled and macro-engine validation metrics."""
    is_pareto = []
    for i, candidate in enumerate(rows):
        dominated = False
        for j, challenger in enumerate(rows):
            if i == j:
                continue
            no_worse = all(challenger[m] <= candidate[m] for m in MINIMIZE_METRICS)
            no_worse &= all(challenger[m] >= candidate[m] for m in MAXIMIZE_METRICS)
            strictly_better = any(challenger[m] < candidate[m] for m in MINIMIZE_METRICS)
            strictly_better |= any(challenger[m] > candidate[m] for m in MAXIMIZE_METRICS)
            if no_worse and strictly_better:
                dominated = True
                break
        is_pareto.append(not dominated)
    return is_pareto


def annotate_pareto(rows: list[dict]) -> None:
    common_rows = [
        {metric: row[metric] for metric in (*MINIMIZE_METRICS, *MAXIMIZE_METRICS)}
        for row in rows
    ]
    for row, is_pareto in zip(rows, pareto_front(common_rows)):
        row["pareto_optimal"] = is_pareto


def summarize_metrics(y: np.ndarray, pred: np.ndarray, engine_ids: np.ndarray) -> dict[str, float]:
    pooled = metric_dict(y, pred)
    engine_metrics = [
        metric_dict(y[engine_ids == engine_id], pred[engine_ids == engine_id])
        for engine_id in np.unique(engine_ids)
    ]
    pooled.update({
        f"macro_engine_{key}": float(np.mean([m[key] for m in engine_metrics]))
        for key in (
            "MAE_hours",
            "RMSE_hours",
            "within_1h_percent",
            "within_2h_percent",
            "nasa_phm08_score",
        )
    })
    return pooled


def common_sequence_cadence_mask(validation: pd.DataFrame, sequence_length: int, stride: int) -> np.ndarray:
    """Match LSTM sequence endpoints: first endpoint after warm-up, then fixed stride per engine."""
    mask = np.zeros(len(validation), dtype=bool)
    for _, indices in validation.groupby("engine_id", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        selected = idx[(np.arange(len(idx)) >= sequence_length - 1) &
                       ((np.arange(len(idx)) - (sequence_length - 1)) % stride == 0)]
        mask[selected] = True
    return mask


def load_partition(root: Path, cfg: dict, engine_ids: list[int]) -> pd.DataFrame:
    wanted = set(engine_ids)
    chunks = [
        chunk[chunk["engine_id"].isin(wanted)]
        for chunk in pd.read_csv(root / cfg["dataset"]["output_path"], chunksize=100_000)
    ]
    return pd.concat(chunks, ignore_index=True).sort_values(
        ["engine_id", "timestamp_min"]
    ).reset_index(drop=True)


def build_features(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    result = build_feature_frame(
        raw,
        tuple(cfg["features"]["lags"]),
        tuple(cfg["features"]["windows"]),
        tuple(cfg["features"]["ewm_spans"]),
    )
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].astype(np.float32)
    return result

def run_search(cfg_path: str) -> list[dict]:
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    manifest = pd.read_csv(ROOT / "data/processed/engine_manifest.csv")
    splits = create_production_splits(manifest, cfg)
    if set(splits["train"]) & set(splits["validation"]):
        raise AssertionError("Train and validation engines overlap")

    train_raw = load_partition(ROOT, cfg, splits["train"])
    validation_raw = load_partition(ROOT, cfg, splits["validation"])
    del manifest

    row_cap = int(cfg["training_sampling"]["xgb_max_rows_per_engine"])
    train = deterministic_train_sample(build_features(train_raw, cfg), row_cap)
    validation = build_features(validation_raw, cfg)
    del train_raw, validation_raw

    feature_columns = model_feature_columns(train)
    qlo, qhi = fit_quantile_clipper(train[feature_columns])
    imputer = TrainFittedImputer.fit(clip(train[feature_columns], qlo, qhi))
    x_train = imputer.transform(clip(train[feature_columns], qlo, qhi))
    x_validation = imputer.transform(clip(validation[feature_columns], qlo, qhi))
    y_train = train["rul_hours"].to_numpy(dtype=np.float32)
    y_validation = validation["rul_hours"].to_numpy(dtype=np.float32)
    engine_ids = validation["engine_id"].to_numpy()
    cadence_mask = common_sequence_cadence_mask(
        validation, int(cfg["lstm"]["sequence_length"]), int(cfg["lstm"]["test_stride"])
    )
    if not cadence_mask.any():
        raise ValueError("No validation rows on the configured LSTM comparison cadence")

    xcfg = cfg["xgb"]
    candidates = xcfg.get("validation_search_candidates", [])
    if not candidates:
        raise ValueError("No xgb.validation_search_candidates configured")
    rows: list[dict] = []
    kalman_candidates = [tuple(c) for c in cfg["kalman"]["candidates"]]
    for number, candidate in enumerate(candidates, start=1):
        params = {
            "objective": candidate["objective"],
            "tree_method": xcfg["tree_method"],
            "subsample": float(xcfg["subsample"]),
            "colsample_bytree": float(xcfg["colsample_bytree"]),
            "reg_alpha": float(xcfg["reg_alpha"]),
            "reg_lambda": float(xcfg["reg_lambda"]),
            "max_depth": int(candidate["max_depth"]),
            "learning_rate": float(candidate["learning_rate"]),
            "n_estimators": int(candidate["n_estimators"]),
            "min_child_weight": int(candidate["min_child_weight"]),
            "max_bin": 256,
            "n_jobs": int(xcfg["n_jobs"]),
            "random_state": int(cfg["project"]["seed"]),
        }
        model = XGBRegressor(**params)
        model.fit(x_train, y_train, verbose=False)
        raw_prediction = np.clip(model.predict(x_validation), 0.0, None)
        candidates_for_scoring = [("raw", None, raw_prediction)]
        for kalman_params in kalman_candidates:
            tracked = track_by_engine(raw_prediction, engine_ids, kalman_params)
            candidates_for_scoring.append(("kalman", kalman_params, tracked))

        model_size = len(model.get_booster().save_raw(raw_format="ubj"))
        for variant, kalman_params, prediction in candidates_for_scoring:
            full_metrics = summarize_metrics(y_validation, prediction, engine_ids)
            common_metrics = summarize_metrics(
                y_validation[cadence_mask],
                prediction[cadence_mask],
                engine_ids[cadence_mask],
            )
            rows.append({
                "candidate_id": f"xgb_{number:02d}_{variant}_{len(rows)}",
                "objective": params["objective"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "n_estimators": params["n_estimators"],
                "min_child_weight": params["min_child_weight"],
                "postprocessor": variant,
                "kalman_params": json.dumps(kalman_params) if kalman_params else "",
                "booster_size_bytes": model_size,
                "common_validation_rows": int(cadence_mask.sum()),
                **{f"full_{key}": value for key, value in full_metrics.items()},
                **common_metrics,
            })
        print(
            f"[{number}/{len(candidates)}] {params['objective']} depth={params['max_depth']} "
            f"lr={params['learning_rate']} trees={params['n_estimators']} complete",
            flush=True,
        )
        del model, raw_prediction, candidates_for_scoring
        gc.collect()

    # Compare variants at LSTM-comparable validation timestamps and account for both pooled
    # accuracy, average engine-level behavior, and serialized booster size.
    annotate_pareto(rows)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(
        ["pareto_optimal", "within_1h_percent", "MAE_hours"],
        ascending=[False, False, True],
    )
    frame.to_csv(reports / "validation_pareto.csv", index=False)
    (reports / "validation_pareto.json").write_text(json.dumps({
        "selection_data": "train and validation engines only",
        "test_partition_used": False,
        "validation_cadence": {
            "sequence_length": int(cfg["lstm"]["sequence_length"]),
            "stride": int(cfg["lstm"]["test_stride"]),
            "rows": int(cadence_mask.sum()),
        },
        "pareto_metrics": {
            "minimize": list(MINIMIZE_METRICS),
            "maximize": list(MAXIMIZE_METRICS),
        },
        "candidates": rows,
    }, indent=2))
    print(frame.loc[frame["pareto_optimal"], [
        "candidate_id", "objective", "postprocessor", "MAE_hours", "RMSE_hours",
        "R2_percent", "within_1h_percent", "within_2h_percent", "nasa_phm08_score",
        "booster_size_bytes",
    ]].to_string(index=False))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a train/validation-only XGBoost Pareto search.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run_search(args.config)


if __name__ == "__main__":
    main()
