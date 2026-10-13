import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc, math
from features import build_features
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer
from metrics import metric_dict
from xgboost import XGBRegressor
root=Path('.')
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
raw=pd.read_csv(cfg['dataset']['output_path'])
def get(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    return build_features(df, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans']))
print('building...',flush=True)
train_feat=get(splits['train']); val_feat=get(splits['validation'])
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in train_feat.columns if c not in banned and pd.api.types.is_numeric_dtype(train_feat[c])]
print(f'ncols {len(cols)}',flush=True)
def sample(df,m=400):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
tr_s=sample(train_feat,400)
train_age=tr_s['service_hours_since_overhaul'].to_numpy(dtype=float)
val_age=val_feat['service_hours_since_overhaul'].to_numpy(dtype=float)
mean_life=manifest[manifest.engine_id.isin(splits['train'])]['lifetime_min'].mean()/60.0
ytr_rul=tr_s['rul_hours'].to_numpy(np.float32)
yv=val_feat['rul_hours'].to_numpy(np.float32)
ytr_res=ytr_rul-(mean_life-train_age)
Xtr_raw=tr_s[cols]; Xv_raw=val_feat[cols]
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(Xv_raw,qlo,qhi))
print(f'train rows {len(Xtr)} val rows {len(Xv)}',flush=True)
for name,ytr in [('direct',ytr_rul),('resid',ytr_res)]:
    m=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008)
    m.fit(Xtr,ytr,verbose=False)
    if name=='direct': pred=np.clip(m.predict(Xv),0,None)
    else: pred=np.clip((mean_life-val_age)+m.predict(Xv),0,None)
    d=metric_dict(yv,pred)
    print(f'{name} MAE {d["MAE_hours"]:.4f} RMSE {d["RMSE_hours"]:.4f} R2 {d["R2_percent"]:.2f} bytes {len(m.get_booster().save_raw("ubj"))}',flush=True)
    del m; gc.collect()
print('done',flush=True)
