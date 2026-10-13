import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc
from features import build_features, CORE_SENSORS
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer, causal_despike
from metrics import metric_dict
from xgboost import XGBRegressor
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
raw=pd.read_csv(cfg['dataset']['output_path'])
def robust_load(df_feat):
    # median of 3 normalized load proxies (rpm, clutch, solenoid) - robust to single-sensor spike
    rpm=df_feat['engine_rpm'].to_numpy(dtype=np.float32)/2400.0
    cl=(df_feat['clutch_pressure_bar'].to_numpy(dtype=np.float32)-3.0)/13.0
    so=(df_feat['solenoid_current_a'].to_numpy(dtype=np.float32)-0.18)/1.75
    stack=np.stack([rpm,cl,so],axis=1)
    # nanmedian ignoring NaNs
    with np.errstate(invalid='ignore'):
        med=np.nanmedian(np.where(np.isfinite(stack),stack,np.nan),axis=1)
    med=np.clip(np.where(np.isfinite(med),med,rpm),0,1).astype(np.float32)
    return med
print('building base...',flush=True)
def load(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    df=causal_despike(df)
    feat=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']),tuple(cfg['features']['ewm_spans']))
    # add robust-load residuals for 3 key sensors (causal, uses only current row sensors)
    rl=robust_load(feat)
    feat['rload']=rl.astype(np.float32)
    feat['coolant_excess_rload']=(feat['engine_coolant_temp_c'].to_numpy(dtype=np.float32)-(73.0+34.0*rl)).astype(np.float32)
    feat['trans_excess_rload']=(feat['transmission_oil_temp_c'].to_numpy(dtype=np.float32)-(64.0+31.0*rl)).astype(np.float32)
    feat['oildef_rload']=(feat['engine_oil_pressure_bar'].to_numpy(dtype=np.float32)-(2.1+3.2*np.sqrt(np.clip(rl,0,1)))).astype(np.float32)
    num=feat.select_dtypes(include=[np.number]).columns
    feat[num]=feat[num].astype(np.float32)
    return feat
tr=load(splits['train']); va=load(splits['validation'])
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in tr.columns if c not in banned and pd.api.types.is_numeric_dtype(tr[c])]
print(f'ncols {len(cols)}',flush=True)
def sample(df,m=300):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
trs=sample(tr,300)
Xtr_raw=trs[cols]; ytr=trs['rul_hours'].to_numpy(np.float32)
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(va[cols],qlo,qhi)); yv=va['rul_hours'].to_numpy(np.float32)
# huber sweep + reg sweep (small, fast)
for obj,extra in [('reg:absoluteerror',{}),('reg:squarederror',{}),('reg:pseudohubererror',{'huber_slope':2.0}),('reg:pseudohubererror',{'huber_slope':5.0})]:
    try:
        m=XGBRegressor(objective=obj,tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.05,n_estimators=150,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008,**extra)
        m.fit(Xtr,ytr,verbose=False)
        print(f'{obj} {extra}',metric_dict(yv,np.clip(m.predict(Xv),0,None)),flush=True)
        del m; gc.collect()
    except Exception as e:
        print(f'{obj} {extra} FAIL {e}',flush=True)
print('done',flush=True)
