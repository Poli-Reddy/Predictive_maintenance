from __future__ import annotations
import json,time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib,xgboost as xgb
from features import build_features
from preprocess import clip
from kalman import CausalRULKalman

root=Path(__file__).resolve().parents[1]
hist=pd.read_csv(root/"reports/edge_demo_history.csv")
start=time.perf_counter(); feat=build_features(hist); feature_s=time.perf_counter()-start
cols=joblib.load(root/"models/feature_columns.joblib"); qlo=joblib.load(root/"models/xgb_qlo.joblib"); qhi=joblib.load(root/"models/xgb_qhi.joblib"); imp=joblib.load(root/"models/xgb_imputer.joblib")
model=xgb.XGBRegressor(); model.load_model(root/"models/xgb_rul.json")
X=imp.transform(clip(feat[cols],qlo,qhi))
start=time.perf_counter(); raw=np.clip(model.predict(X),0,None); pred_s=time.perf_counter()-start
kf=joblib.load(root/"models/kalman_params.joblib"); tracker=CausalRULKalman(*kf)
start=time.perf_counter(); tracked=[tracker.update(float(p)) for p in raw]; kf_s=time.perf_counter()-start
files=[root/"models/xgb_rul.json",root/"models/xgb_imputer.joblib",root/"models/xgb_qlo.joblib",root/"models/xgb_qhi.joblib",root/"models/feature_columns.joblib",root/"models/kalman_params.joblib"]
size=sum(f.stat().st_size for f in files)
out={"history_rows":len(hist),"feature_generation_seconds":feature_s,"xgb_prediction_seconds":pred_s,"kalman_seconds":kf_s,"package_bytes":size,"package_mb":size/1024/1024,"last_rul_hours":float(tracked[-1])}
(root/"reports/edge_benchmark.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
