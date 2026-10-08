from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from features import build_features, model_feature_columns
from kalman import CausalRULKalman
from preprocess import clip


def main():
    ap=argparse.ArgumentParser(description="Online causal RUL inference for one engine history")
    ap.add_argument("--history",required=True,help="CSV containing only telemetry observed so far for one engine")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    hist=pd.read_csv(args.history).sort_values("timestamp_min").reset_index(drop=True)
    if hist.engine_id.nunique()!=1: raise ValueError("--history must contain exactly one engine")
    feat=build_features(hist)
    cols=joblib.load(root/"models/feature_columns.joblib")
    qlo=joblib.load(root/"models/xgb_qlo.joblib"); qhi=joblib.load(root/"models/xgb_qhi.joblib"); imp=joblib.load(root/"models/xgb_imputer.joblib")
    model=xgb.XGBRegressor(); model.load_model(root/"models/xgb_rul.json")
    X=imp.transform(clip(feat[cols],qlo,qhi))
    raw=np.clip(model.predict(X),0,None)
    cfg=joblib.load(root/"models/kalman_params.joblib")
    tracker=CausalRULKalman(*cfg); tracked=[]
    for p in raw: tracked.append(tracker.update(float(p)))
    result={"engine_id":int(hist.engine_id.iloc[-1]),"timestamp_min":float(hist.timestamp_min.iloc[-1]),"xgb_rul_hours":float(raw[-1]),"xgb_kalman_rul_hours":float(tracked[-1]),"online_causal":True}
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
