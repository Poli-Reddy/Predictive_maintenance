import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc
from features import build_features
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer
from metrics import metric_dict
from xgboost import XGBRegressor
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
def sample(df,m=350):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
tr_s=sample(train_feat,350)
ytr=tr_s['rul_hours'].to_numpy(np.float32); yv=val_feat['rul_hours'].to_numpy(np.float32)
# Stage1: depth6/200 base
Xtr_raw=tr_s[cols]; Xv_raw=val_feat[cols]
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(Xv_raw,qlo,qhi))
m1=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008)
m1.fit(Xtr,ytr,verbose=False)
ptr=np.clip(m1.predict(Xtr),0,None); pv=np.clip(m1.predict(Xv),0,None)
print('stage1 train',metric_dict(ytr,ptr),'val',metric_dict(yv,pv),flush=True)
# Stage2: predict residual (y - stage1) using same X + stage1 pred as extra feature
res_tr=ytr-ptr
# build augmented matrices: X + pred column (need to refit imputer? just append)
Xtr2=np.concatenate([Xtr,ptr.reshape(-1,1).astype(np.float32)],axis=1)
Xv2=np.concatenate([Xv,pv.reshape(-1,1).astype(np.float32)],axis=1)
m2=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=4,learning_rate=0.05,n_estimators=100,min_child_weight=12,max_bin=128,n_jobs=4,random_state=20261008)
m2.fit(Xtr2,res_tr,verbose=False)
corr_tr=ptr+np.clip(m2.predict(Xtr2),-5,5)
corr_v=pv+np.clip(m2.predict(Xv2),-5,5)
corr_v=np.clip(corr_v,0,None); corr_tr=np.clip(corr_tr,0,None)
print('stage2 corrected train',metric_dict(ytr,corr_tr),'val',metric_dict(yv,corr_v),flush=True)
print(f'size1 {len(m1.get_booster().save_raw("ubj"))} size2 {len(m2.get_booster().save_raw("ubj"))} total {len(m1.get_booster().save_raw("ubj"))+len(m2.get_booster().save_raw("ubj"))}',flush=True)
# monotone constraint test: age decreasing
# find age col index
age_idx=cols.index('service_hours_since_overhaul')
mc=['0']*len(cols); mc[age_idx]='-1'
m3=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008,monotone_constraints='('+','.join(mc)+')')
m3.fit(Xtr,ytr,verbose=False)
print('monotone-age',metric_dict(yv,np.clip(m3.predict(Xv),0,None)),flush=True)
print('done',flush=True)
