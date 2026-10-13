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
def load(feat_ids, ffill=False):
    df=raw[raw.engine_id.isin(feat_ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    df=causal_despike(df)
    if ffill:
        # causal forward-fill per engine (past-only), keep missing flag via features
        from features import CORE_SENSORS
        for c in CORE_SENSORS:
            df[c]=df.groupby('engine_id')[c].transform(lambda s: s.ffill())
    feat=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']),tuple(cfg['features']['ewm_spans']))
    num=feat.select_dtypes(include=[np.number]).columns
    feat[num]=feat[num].astype(np.float32)
    return feat
print('building...',flush=True)
for ffill in [False,True]:
    tr=load(splits['train'],ffill); va=load(splits['validation'],ffill)
    banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
    cols=[c for c in tr.columns if c not in banned and pd.api.types.is_numeric_dtype(tr[c])]
    def sample(df,m=300):
        pcs=[]
        for _,g in df.groupby('engine_id',sort=False):
            pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
        return pd.concat(pcs,ignore_index=True)
    trs=sample(tr,300)
    Xtr_raw=trs[cols]; ytr=trs['rul_hours'].to_numpy(np.float32)
    qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
    Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(va[cols],qlo,qhi)); yv=va['rul_hours'].to_numpy(np.float32)
    m=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008)
    m.fit(Xtr,ytr,verbose=False)
    print(f'ffill={ffill} ncols={len(cols)}',metric_dict(yv,np.clip(m.predict(Xv),0,None)),flush=True)
    del m,tr,va; gc.collect()
print('done',flush=True)
