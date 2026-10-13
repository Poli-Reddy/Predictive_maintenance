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
    return build_features(df, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans'])), df
print('building...',flush=True)
train_feat,train_raw=get(splits['train']); val_feat,val_raw=get(splits['validation'])
# per-load-bin baselines from first 150 rows, 3 bins, using train_raw/val_raw despiked? use raw as-is for speed (spikes rare)
for feat,dfraw in [(train_feat,train_raw),(val_feat,val_raw)]:
    load=feat['load_proxy'].to_numpy(dtype=np.float32)
    ts=dfraw['timestamp_min'].to_numpy()
    eids=feat['engine_id'].to_numpy()
    # precompute per-engine per-bin medians for 2 key sensors + residuals to limit cols (keep size down)
    for c in ['engine_coolant_temp_c','transmission_oil_temp_c','engine_oil_pressure_bar','transmission_slip_proxy','coolant_excess_vs_load','oil_pressure_deficit_vs_rpm']:
        x=feat[c].to_numpy(dtype=np.float32)
        # per-engine dict of 3 medians
        base_map={}
        for eid in np.unique(eids):
            m=eids==eid
            early=m&(ts<=150)
            b=[]
            for lo,hi in [(0,0.25),(0.25,0.6),(0.6,2.0)]:
                sel=early&(load>=lo)&(load<hi)
                v=x[sel]
                v=v[np.isfinite(v)]
                b.append(float(np.median(v)) if len(v)>=10 else float(np.nanmedian(x[early])) if np.isfinite(x[early]).any() else 0.0)
            base_map[eid]=b
        # map current load bin to baseline
        bins=np.digitize(load,[0.25,0.6])-0  # 0,1,2
        base=np.array([base_map[e][b] for e,b in zip(eids,bins)],dtype=np.float32)
        feat[f'{c}__binbase']=base
        feat[f'{c}__bindelta']= (x-base).astype(np.float32)
        feat[f'{c}__binratio']= (x/np.where(np.abs(base)>1e-3,base,1.0)).astype(np.float32)
    feat[feat.select_dtypes(include=[np.number]).columns]=feat[feat.select_dtypes(include=[np.number]).columns].astype(np.float32)
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in train_feat.columns if c not in banned and pd.api.types.is_numeric_dtype(train_feat[c])]
print(f'ncols {len(cols)}',flush=True)
def sample(df,m=350):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
tr_s=sample(train_feat,350)
yv=val_feat['rul_hours'].to_numpy(np.float32)
Xtr_raw=tr_s[cols]; ytr=tr_s['rul_hours'].to_numpy(np.float32)
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(val_feat[cols],qlo,qhi))
print(f'train {len(Xtr)}',flush=True)
for params in [dict(max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6),dict(max_depth=6,learning_rate=0.03,n_estimators=350,min_child_weight=6)]:
    m=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_bin=128,n_jobs=4,random_state=20261008,**params)
    m.fit(Xtr,ytr,verbose=False)
    pred=np.clip(m.predict(Xv),0,None)
    print(f"{params} {metric_dict(yv,pred)} bytes {len(m.get_booster().save_raw('ubj'))}",flush=True)
    del m; gc.collect()
print('done',flush=True)
