import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc, json
from features import build_features
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer, causal_despike
from metrics import metric_dict
from kalman import track_by_engine
from xgboost import XGBRegressor
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
raw=pd.read_csv(cfg['dataset']['output_path'])
print(f'raw engines {raw.engine_id.nunique()} rows {len(raw)}',flush=True)
train_ids=list(range(1,561))+list(range(801,1201))
val_ids=list(range(561,681))
print(f'train {len(train_ids)} val {len(val_ids)}',flush=True)
def load(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    df=causal_despike(df)
    feat=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']),tuple(cfg['features']['ewm_spans']))
    num=feat.select_dtypes(include=[np.number]).columns
    feat[num]=feat[num].astype(np.float32)
    return feat
print('building train...',flush=True)
tr=load(train_ids)
print('building val...',flush=True)
va=load(val_ids)
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in tr.columns if c not in banned and pd.api.types.is_numeric_dtype(tr[c])]
print(f'ncols {len(cols)}',flush=True)
def sample(df,m=300):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
trs=sample(tr,300)
ytr=trs['rul_hours'].to_numpy(np.float32); yv=va['rul_hours'].to_numpy(np.float32)
w=np.where(ytr>10,2.0,1.0).astype(np.float32)
Xtr_raw=trs[cols]; Xv_raw=va[cols]
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(Xv_raw,qlo,qhi))
print(f'train rows {len(Xtr)}',flush=True)
m=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=0.8,colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=10.0,max_depth=6,learning_rate=0.03,n_estimators=400,min_child_weight=12,max_bin=128,n_jobs=4,random_state=20261008)
m.fit(Xtr,ytr,sample_weight=w,verbose=False)
pred=np.clip(m.predict(Xv),0,None)
print('extended-train val raw',metric_dict(yv,pred),f'bytes {len(m.get_booster().save_raw("ubj"))}',flush=True)
eids=va['engine_id'].to_numpy()
best=None
for kf in [(0.002,1e-05,0.30),(0.01,1e-05,0.30),(0.01,1e-05,0.6),(0.005,1e-05,0.15)]:
    dd=metric_dict(yv,track_by_engine(pred,eids,tuple(kf)))
    s=(-dd['RMSE_hours'],-dd['MAE_hours'])
    if best is None or s>best[0]: best=(s,kf,dd)
print(f'bestkf {best[1]}',best[2],flush=True)
print('done',flush=True)
