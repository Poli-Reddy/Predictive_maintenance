from __future__ import annotations

import argparse
import gc
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from features import build_features, model_feature_columns
from kalman import track_by_engine
from metrics import metric_dict
from preprocess import TrainFittedImputer, clip, fit_quantile_clipper, save_preprocessor
from splits import create_production_splits, save_splits
from train_lstm import fit_model as fit_lstm, predict as predict_lstm


def load_cfg(root: Path, path: str) -> dict:
    return yaml.safe_load((root / path).read_text())


def load_raw(root: Path, cfg: dict, engine_ids: list[int] | None = None) -> pd.DataFrame:
    raw = pd.read_csv(root / cfg["dataset"]["output_path"])
    if engine_ids is not None:
        raw = raw[raw.engine_id.isin(engine_ids)].copy()
    return raw.sort_values(["engine_id", "timestamp_min"]).reset_index(drop=True)


def make_features(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = build_features(
        raw,
        tuple(cfg["features"]["lags"]),
        tuple(cfg["features"]["windows"]),
        tuple(cfg["features"]["ewm_spans"]),
    )
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].astype(np.float32)
    return out


def deterministic_train_sample(df: pd.DataFrame, max_rows_per_engine: int) -> pd.DataFrame:
    """Reduce highly correlated adjacent training rows while preserving every engine and full-life coverage.
    Validation/test are never sampled. This is a compute/memory control, not an evaluation shortcut.
    """
    if max_rows_per_engine <= 0:
        return df
    pieces=[]
    for _, g in df.groupby("engine_id", sort=False):
        if len(g) <= max_rows_per_engine:
            pieces.append(g)
        else:
            idx=np.linspace(0,len(g)-1,max_rows_per_engine,dtype=np.int64)
            pieces.append(g.iloc[idx])
    return pd.concat(pieces,ignore_index=True)

def prepare_xgb_train(X: pd.DataFrame, y: np.ndarray):
    qlo,qhi=fit_quantile_clipper(X)
    clipped=clip(X,qlo,qhi)
    imp=TrainFittedImputer.fit(clipped)
    return imp.transform(clipped),qlo,qhi,imp


def fit_dev_model(Xtr, ytr, Xv, yv, cfg):
    xcfg=cfg["xgb"]
    candidates=[{"max_depth":int(d),"learning_rate":float(lr),"n_estimators":int(ne),"min_child_weight":int(mc)} for d,lr,ne,mc in product(xcfg["max_depth_candidates"],xcfg["learning_rate_candidates"],xcfg["n_estimators_candidates"],xcfg["min_child_weight_candidates"])]
    rows=[]; best=None
    for p in candidates:
        model=XGBRegressor(objective=xcfg["objective"],tree_method=xcfg["tree_method"],subsample=float(xcfg["subsample"]),colsample_bytree=float(xcfg["colsample_bytree"]),reg_alpha=float(xcfg["reg_alpha"]),reg_lambda=float(xcfg["reg_lambda"]),max_depth=p["max_depth"],learning_rate=p["learning_rate"],n_estimators=p["n_estimators"],min_child_weight=p["min_child_weight"],max_bin=256,n_jobs=int(xcfg["n_jobs"]),random_state=int(cfg["project"]["seed"]))
        model.fit(Xtr,ytr,eval_set=[(Xv,yv)],verbose=False)
        pred=np.clip(model.predict(Xv),0,None)
        m=metric_dict(yv,pred); rows.append({**p,"validation_RMSE_hours":m["RMSE_hours"],"validation_R2":m["R2"]})
        if best is None or m["RMSE_hours"]<best[0]: best=(m["RMSE_hours"],p,model,pred)
        print(f"XGB dev candidate {p}: RMSE={m['RMSE_hours']:.4f}, R2={m['R2_percent']:.2f}%")
        del model,pred
        gc.collect()
    return best, sorted(rows,key=lambda r:r["validation_RMSE_hours"])


def final_model(X,y,best,cfg):
    xcfg=cfg["xgb"]
    model=XGBRegressor(objective=xcfg["objective"],tree_method=xcfg["tree_method"],subsample=float(xcfg["subsample"]),colsample_bytree=float(xcfg["colsample_bytree"]),reg_alpha=float(xcfg["reg_alpha"]),reg_lambda=float(xcfg["reg_lambda"]),max_depth=int(best["max_depth"]),learning_rate=float(best["learning_rate"]),n_estimators=int(best["n_estimators"]),min_child_weight=int(best["min_child_weight"]),max_bin=256,n_jobs=int(xcfg["n_jobs"]),random_state=int(cfg["project"]["seed"]))
    model.fit(X,y,verbose=False)
    return model


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; cfg=load_cfg(root,args.config); seed=int(cfg["project"]["seed"]); np.random.seed(seed)

    # Split manifest is created from engine IDs before any model feature or preprocessing work.
    manifest=pd.read_csv(root/"data/processed/engine_manifest.csv")
    splits=create_production_splits(manifest,cfg); save_splits(splits,root/cfg["dataset"]["split_manifest"])
    print({k:len(v) for k,v in splits.items()})

    # ------------------------------ LSTM BASELINE ------------------------------
    train_raw=load_raw(root,cfg,splits["train"]); val_raw=load_raw(root,cfg,splits["validation"])
    lstm_model,lstm_scaler,lstm_device,lstm_seq=fit_lstm(train_raw,val_raw,cfg,root)
    lstm_val_meta,lstm_val_y,lstm_val_pred=predict_lstm(
        lstm_model,lstm_scaler,lstm_device,val_raw,lstm_seq,root,stride=5
    )
    val_lstm_metrics=metric_dict(lstm_val_y,lstm_val_pred)
    best_lstm_calibration=None
    for scale in np.arange(0.90,1.101,0.025):
        for offset in np.arange(-0.60,0.601,0.025):
            calibrated=np.clip(scale*lstm_val_pred+offset,0,None)
            metrics=metric_dict(lstm_val_y,calibrated)
            score=(metrics["within_1h_percent"],-metrics["MAE_hours"],-metrics["RMSE_hours"])
            if best_lstm_calibration is None or score>best_lstm_calibration[0]:
                best_lstm_calibration=(score,float(scale),float(offset),metrics)
    _,lstm_calibration_scale,lstm_calibration_offset,lstm_val_calibrated_metrics=best_lstm_calibration
    test_raw=load_raw(root,cfg,splits["test"])
    lstm_meta,lstm_y,lstm_pred=predict_lstm(lstm_model,lstm_scaler,lstm_device,test_raw,lstm_seq,root)
    lstm_pred=np.clip(lstm_calibration_scale*lstm_pred+lstm_calibration_offset,0,None)
    lp=lstm_meta.copy(); lp["y_true_rul_hours"]=lstm_y; lp["lstm_rul_hours"]=lstm_pred
    lp.to_csv(root/"reports/lstm_test_predictions.csv.gz",index=False,compression="gzip")
    del train_raw,test_raw,lstm_model,lstm_scaler,lstm_pred,lstm_y,lstm_meta; gc.collect()

    # ------------------------------ XGBOOST DEV ------------------------------
    train_raw=load_raw(root,cfg,splits["train"]); val_raw=load_raw(root,cfg,splits["validation"])
    xgb_rows = int(cfg["training_sampling"]["xgb_max_rows_per_engine"])
    train_feat=deterministic_train_sample(make_features(train_raw,cfg), xgb_rows)
    val_feat=make_features(val_raw,cfg)
    feature_cols=model_feature_columns(train_feat)
    Xtr,qlo,qhi,imp=prepare_xgb_train(train_feat[feature_cols],train_feat["rul_hours"].to_numpy(np.float32))
    Xv=imp.transform(clip(val_feat[feature_cols],qlo,qhi)); yv=val_feat["rul_hours"].to_numpy(np.float32)
    ytr=train_feat["rul_hours"].to_numpy(np.float32)
    best, candidates=fit_dev_model(Xtr,ytr,Xv,yv,cfg)
    best_rmse,best_params,dev_model,val_pred=best

    # Select Kalman parameters for within-one-hour accuracy on validation timestamps
    # where the LSTM baseline is evaluated; the test partition remains untouched.
    val_meta=val_feat[["engine_id","timestamp_min"]]
    dev_val_metrics=metric_dict(yv,val_pred)
    lstm_val_keys=pd.MultiIndex.from_frame(lstm_val_meta[["engine_id","timestamp_min"]])
    val_keys=pd.MultiIndex.from_frame(val_meta)
    common_val=val_keys.isin(lstm_val_keys)
    if int(common_val.sum())!=len(lstm_val_meta):
        raise AssertionError("LSTM and XGBoost validation timestamps do not align")
    best_kf=None; best_kf_score=None; best_kf_metrics=None
    for cand in cfg["kalman"]["candidates"]:
        tracked=track_by_engine(val_pred,val_meta.engine_id.to_numpy(),tuple(cand))
        metrics=metric_dict(yv[common_val],tracked[common_val])
        if cfg["kalman"]["selection_metric"]=="within_1h_percent":
            score=(metrics["within_1h_percent"],-metrics["MAE_hours"],-metrics["RMSE_hours"])
        elif cfg["kalman"]["selection_metric"]=="RMSE_hours":
            score=(-metrics["RMSE_hours"],-metrics["MAE_hours"],metrics["within_1h_percent"])
        else:
            raise ValueError(f"Unsupported Kalman selection metric: {cfg['kalman']['selection_metric']}")
        if best_kf_score is None or score>best_kf_score:
            best_kf_score=score; best_kf=tuple(cand); best_kf_metrics=metrics
    del tracked,Xtr,Xv,train_feat,val_feat,train_raw,val_raw,imp,qlo,qhi,dev_model,val_pred; gc.collect()

    # ------------------------------ FINAL FIT ------------------------------
    tv_ids=splits["train"]+splits["validation"]
    tv_raw=load_raw(root,cfg,tv_ids); test_raw=load_raw(root,cfg,splits["test"])
    tv_feat=deterministic_train_sample(make_features(tv_raw,cfg), xgb_rows)
    test_feat=make_features(test_raw,cfg)
    if model_feature_columns(tv_feat)!=feature_cols:
        raise AssertionError("Final feature schema changed after model selection")
    Xtv,qlo_tv,qhi_tv,imp_tv=prepare_xgb_train(tv_feat[feature_cols],tv_feat["rul_hours"].to_numpy(np.float32))
    Xt=imp_tv.transform(clip(test_feat[feature_cols],qlo_tv,qhi_tv)); yt=test_feat["rul_hours"].to_numpy(np.float32)
    final=final_model(Xtv,tv_feat["rul_hours"].to_numpy(np.float32),best_params,cfg)
    pred_xgb=np.clip(final.predict(Xt),0,None); pred_kf=track_by_engine(pred_xgb,test_feat.engine_id.to_numpy(),best_kf)
    xpred=test_feat[["engine_id","timestamp_min"]].copy(); xpred["y_true_rul_hours"]=yt; xpred["xgb_rul_hours"]=pred_xgb; xpred["xgb_kalman_rul_hours"]=pred_kf
    xpred.to_csv(root/"reports/xgb_test_predictions.csv.gz",index=False,compression="gzip")

    model_dir=root/"models"; report_dir=root/"reports"; proc=root/"data/processed"
    for d in (model_dir,report_dir,proc): d.mkdir(parents=True,exist_ok=True)
    final.save_model(model_dir/"xgb_rul.json"); save_preprocessor(model_dir,qlo_tv,qhi_tv,imp_tv,feature_cols); joblib.dump(best_kf,model_dir/"kalman_params.joblib")
    joblib.dump({"version":cfg["project"]["version"],"seed":seed,"feature_columns":feature_cols,"selected_xgb":best_params,"objective":cfg["xgb"]["objective"]},model_dir/"model_metadata.joblib")
    (model_dir/"prediction_calibration.json").write_text(json.dumps({
        "lstm_scale":lstm_calibration_scale,
        "lstm_offset_hours":lstm_calibration_offset,
        "lstm_selection_metric":"within_1h_percent",
        "selection_data":"validation engines only",
    },indent=2))

    comp=xpred.merge(lp[["engine_id","timestamp_min","lstm_rul_hours"]],on=["engine_id","timestamp_min"],how="inner",validate="one_to_one",suffixes=("_xgb","_lstm"))
    y=comp["y_true_rul_hours"].to_numpy()
    rows=[{"model":"XGBoost","common_test_rows":len(comp),**metric_dict(y,comp["xgb_rul_hours"].to_numpy())},{"model":"XGBoost + Kalman","common_test_rows":len(comp),**metric_dict(y,comp["xgb_kalman_rul_hours"].to_numpy())},{"model":"LSTM","common_test_rows":len(comp),**metric_dict(y,comp["lstm_rul_hours"].to_numpy())}]
    pd.DataFrame(rows).to_csv(report_dir/"comparison.csv",index=False); (report_dir/"comparison.json").write_text(json.dumps(rows,indent=2))
    (report_dir/"model_selection.json").write_text(json.dumps({"selected_xgb":best_params,"xgb_objective":cfg["xgb"]["objective"],"validation_xgb":dev_val_metrics,"selected_kalman":best_kf,"kalman_selection_metric":cfg["kalman"]["selection_metric"],"validation_xgb_kalman":best_kf_metrics,"validation_lstm":val_lstm_metrics,"lstm_calibration":{"scale":lstm_calibration_scale,"offset_hours":lstm_calibration_offset,"validation_calibrated_metrics":lstm_val_calibrated_metrics},"candidate_results":candidates},indent=2))
    (report_dir/"xgb_results.json").write_text(json.dumps({
        "validation_xgb":dev_val_metrics,
        "validation_xgb_kalman":best_kf_metrics,
        "test_xgb":metric_dict(yt,pred_xgb),
        "test_xgb_kalman":metric_dict(yt,pred_kf),
        "test_rows":len(test_feat),
        "common_test_rows":len(comp),
        "feature_count":len(feature_cols),
        "final_train_rows":len(tv_feat),
        "params":best_params,
        "objective":cfg["xgb"]["objective"],
        "kalman_params":best_kf,
        "kalman_selection_metric":cfg["kalman"]["selection_metric"],
        "lstm_calibration":{"scale":lstm_calibration_scale,"offset_hours":lstm_calibration_offset},
        "test_used_for_selection":False,
        "training_row_sampling":(
            "all training rows after full-resolution causal feature generation"
            if xgb_rows <= 0 else
            f"up to {xgb_rows} evenly-spaced rows per training engine after full-resolution causal feature generation"
        ),
    },indent=2))

    (proc/"split_audit.json").write_text(json.dumps({
        "train_engines":splits["train"],
        "validation_engines":splits["validation"],
        "test_engines":splits["test"],
        "strategy":cfg["split"]["strategy"],
        "test_frozen_until_final_evaluation":True,
        "test_used_for_selection":False,
        "production_inference_uses_current_and_past_telemetry_only":True,
    },indent=2))
    sampling_summary = (
        "XGBoost training sample: all training rows after full-resolution causal feature generation.\n"
        if xgb_rows <= 0 else
        f"XGBoost training sample: at most {xgb_rows} evenly spaced rows per training engine after full-resolution causal feature generation.\n"
    )
    (report_dir/"training_summary.md").write_text(
        f"# Training summary\n\nDataset rows: {sum(1 for _ in open(root/cfg['dataset']['output_path'],'rb')) if False else 'generated CSV.GZ'}\n\n"
        f"Production split: {len(splits['train'])} train / {len(splits['validation'])} validation / {len(splits['test'])} test engines.\n"
        f"Feature count: {len(feature_cols)}.\n"
        f"{sampling_summary}"
        "service_hours_since_overhaul is excluded from XGBoost and LSTM inputs as an age proxy.\n"
        f"LSTM training-window cap: {int(cfg['lstm']['max_train_sequences'])}; maximum epochs: {int(cfg['lstm']['epochs'])}; early-stopping patience: {int(cfg['lstm']['patience'])}.\n"
        "LSTM validation windows are used for early stopping/checkpoint selection, not gradient updates when refit_final is false; per-epoch window counts and selected epoch are in reports/lstm_training_history.json.\n"
        "Preprocessing is fit only on training data during development and on train+validation for final refit.\n"
        f"LSTM refit_final: {bool(cfg['lstm'].get('refit_final', False))}.\n"
        "The test partition is never used for model or tracker selection.\n"
        "All online features are causal and use only present/past telemetry.\n"
    )
    # Free large arrays before process end.
    del tv_raw,test_raw,tv_feat,test_feat,Xtv,Xt,final,pred_xgb,pred_kf,comp; gc.collect()
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
