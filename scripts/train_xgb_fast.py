from pathlib import Path
import sys, json, gc
import numpy as np, pandas as pd, yaml
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from features import build_features, model_feature_columns
from preprocess import TrainFittedImputer, clip, fit_quantile_clipper
from splits import create_production_splits
from metrics import metric_dict
root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/'config/config.yaml').read_text())
raw=pd.read_csv(root/cfg['dataset']['output_path']); man=pd.read_csv(root/'data/processed/engine_manifest.csv')
s=create_production_splits(man,cfg)
train=raw[raw.engine_id.isin(s['train'])].copy(); val=raw[raw.engine_id.isin(s['validation'])].copy()
print('raw',len(train),len(val),flush=True)
ft=build_features(train,tuple(cfg['features']['lags']),tuple(cfg['features']['windows'])); fv=build_features(val,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']))
parts=[]
for _,g in ft.groupby('engine_id',sort=False):
    parts.append(g if len(g)<=200 else g.iloc[np.linspace(0,len(g)-1,200,dtype=int)])
ft=pd.concat(parts,ignore_index=True)
num=ft.select_dtypes(include=[np.number]).columns; ft[num]=ft[num].astype(np.float32); fv[num]=fv[num].astype(np.float32)
cols=model_feature_columns(ft); X=ft[cols]; y=ft.rul_hours.to_numpy(np.float32); qlo,qhi=fit_quantile_clipper(X); imp=TrainFittedImputer.fit(clip(X,qlo,qhi)); Xt=imp.transform(clip(X,qlo,qhi)); Xv=imp.transform(clip(fv[cols],qlo,qhi)); yv=fv.rul_hours.to_numpy(np.float32)
print('features',Xt.shape,flush=True)
for depth in [6]:
 m=XGBRegressor(objective='reg:squarederror',tree_method='hist',n_estimators=120,max_depth=depth,learning_rate=.05,min_child_weight=6,subsample=.9,colsample_bytree=.85,reg_alpha=.05,reg_lambda=3,max_bin=128,n_jobs=4,random_state=42)
 m.fit(Xt,y,eval_set=[(Xv,yv)],verbose=False); p=np.clip(m.predict(Xv),0,None); print(depth,metric_dict(yv,p),flush=True)
 if depth==6: m.save_model(root/'models/xgb_dev.json')
 del m,p; gc.collect()
