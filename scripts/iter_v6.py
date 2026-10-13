import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc
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
def sample(df,m=300):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
tr_s=sample(train_feat,300)
yv=val_feat['rul_hours'].to_numpy(np.float32)
val_age=val_feat['service_hours_since_overhaul'].to_numpy(dtype=float)
Xv_raw=val_feat[cols]
# split train into early (RUL>8) and late (RUL<=12) with overlap 8-12
ytr=tr_s['rul_hours'].to_numpy(np.float32)
early_mask=ytr>8; late_mask=ytr<=12
print(f'early {early_mask.sum()} late {late_mask.sum()}',flush=True)
for name,mask in [('early',early_mask),('late',late_mask)]:
    print(f'training {name}...',flush=True)
    Xraw=tr_s[cols][mask]; y= ytr[mask]
    qlo,qhi=fit_quantile_clipper(Xraw); imp=TrainFittedImputer.fit(clip(Xraw,qlo,qhi))
    X=imp.transform(clip(Xraw,qlo,qhi))
    Xv=imp.transform(clip(Xv_raw,qlo,qhi))
    m=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.05,n_estimators=150,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008)
    m.fit(X,y,verbose=False)
    pred=np.clip(m.predict(Xv),0,None)
    print(f'{name} solo',metric_dict(yv,pred),f'bytes {len(m.get_booster().save_raw("ubj"))}',flush=True)
    # save for mixture
    import joblib
    joblib.dump((qlo,qhi,imp,m),f'/tmp/{name}_mix.joblib')
    del m; gc.collect()
# mixture by age gate: age<9 early, age>12 late, blend 9-12
import joblib
qlo_e,qhi_e,imp_e,m_e=joblib.load('/tmp/early_mix.joblib')
qlo_l,qhi_l,imp_l,m_l=joblib.load('/tmp/late_mix.joblib')
pe=np.clip(m_e.predict(imp_e.transform(clip(Xv_raw,qlo_e,qhi_e))),0,None)
pl=np.clip(m_l.predict(imp_l.transform(clip(Xv_raw,qlo_l,qhi_l))),0,None)
# gate weights
w_early=np.clip((12-val_age)/(12-9),0,1)  # 1 when age<=9, 0 when age>=12
mix=w_early*pe+(1-w_early)*pl
print('mixture age-gate',metric_dict(yv,mix),flush=True)
# oracle gate by true RUL (upper bound, uses future - not deployable, just to see headroom)
w_or=np.where(yv>10,1,0)  # not deployable
oracle=w_or*pe+(1-w_or)*pl
print('oracle RUL-gate (cheating upper bound)',metric_dict(yv,oracle),flush=True)
print('done',flush=True)
