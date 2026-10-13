import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc
from features import build_features
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer, causal_despike
from metrics import metric_dict
from xgboost import XGBRegressor
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
raw=pd.read_csv(cfg['dataset']['output_path'])
print('building...',flush=True)
def load(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    df=causal_despike(df)
    feat=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']),tuple(cfg['features']['ewm_spans']))
    num=feat.select_dtypes(include=[np.number]).columns
    feat[num]=feat[num].astype(np.float32)
    return feat
tr=load(splits['train']); va=load(splits['validation'])
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in tr.columns if c not in banned and pd.api.types.is_numeric_dtype(tr[c])]
print(f'ncols {len(cols)}',flush=True)
def sample(df,m=350):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
trs=sample(tr,350)
ytr=trs['rul_hours'].to_numpy(np.float32); yv=va['rul_hours'].to_numpy(np.float32)
# weights: early + hard slices (train labels allowed)
emanifest=manifest.set_index('engine_id')
tr_fail=trs['engine_id'].map(emanifest['failure_mode_sim'])
tr_env=trs['engine_id'].map(emanifest['environment'])
w_early=np.where(ytr>10,2.0,1.0).astype(np.float32)
w_hard=np.where(tr_fail=='transmission_wear',2.0,np.where(tr_env.isin(['high_altitude','cold']),1.5,1.0)).astype(np.float32)
w_both=(w_early*w_hard).astype(np.float32)
Xtr_raw=trs[cols]; Xv_raw=va[cols]
# no-clip variant: impute on raw (no quantile)
med=Xtr_raw.median(numeric_only=True).replace([np.inf,-np.inf],np.nan).fillna(0.0)
Xtr_nc=Xtr_raw.fillna(med).astype(np.float32).to_numpy(); Xv_nc=Xv_raw.fillna(med).astype(np.float32).to_numpy()
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(Xv_raw,qlo,qhi))
print(f'train {len(Xtr)}',flush=True)
configs=[
 ('clip+sq+noweight',Xtr,Xv,None,dict(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,reg_lambda=3.0)),
 ('clip+sq+earlyW',Xtr,Xv,w_early,dict(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,reg_lambda=3.0)),
 ('clip+sq+hardW',Xtr,Xv,w_hard,dict(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,reg_lambda=3.0)),
 ('clip+sq+bothW',Xtr,Xv,w_both,dict(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,reg_lambda=3.0)),
 ('noclip+sq+noweight',Xtr_nc,Xv_nc,None,dict(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,reg_lambda=3.0)),
 ('noclip+sq+bothW',Xtr_nc,Xv_nc,w_both,dict(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,reg_lambda=3.0)),
 ('clip+sq+bothW+reg10mcw12',Xtr,Xv,w_both,dict(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=12,reg_lambda=10.0)),
]
for name,Xa,Xb,w,kw in configs:
    m=XGBRegressor(tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,n_jobs=4,random_state=20261008,**kw)
    m.fit(Xa,ytr,sample_weight=w,verbose=False)
    print(f'{name} {metric_dict(yv,np.clip(m.predict(Xb),0,None))} bytes {len(m.get_booster().save_raw("ubj"))}',flush=True)
    del m; gc.collect()
print('done',flush=True)
