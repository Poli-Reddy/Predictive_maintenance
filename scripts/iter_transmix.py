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
emanifest=manifest.set_index('engine_id')
tr_fail=tr['engine_id'].map(emanifest['failure_mode_sim'])
# split train: transmission vs rest
def sample(df,m=350):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
trs=sample(tr,350)
ytr=trs['rul_hours'].to_numpy(np.float32); yv=va['rul_hours'].to_numpy(np.float32)
ftr=trs['engine_id'].map(emanifest['failure_mode_sim'])
mt=ftr=='transmission_wear'
print(f'train trans {mt.sum()} rest {(~mt).sum()}',flush=True)
# train two experts (square, earlyW inside each? use earlyW for both)
for name,mask in [('trans',mt.to_numpy()),('rest',(~mt).to_numpy())]:
    Xr=trs[cols][mask]; y=ytr[mask]
    w=np.where(y>10,2.0,1.0).astype(np.float32)
    qlo,qhi=fit_quantile_clipper(Xr); imp=TrainFittedImputer.fit(clip(Xr,qlo,qhi))
    X=imp.transform(clip(Xr,qlo,qhi))
    Xv=imp.transform(clip(va[cols],qlo,qhi))
    m=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=0.8,colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=10.0,max_depth=6,learning_rate=0.05,n_estimators=150,min_child_weight=12,max_bin=128,n_jobs=4,random_state=20261008)
    m.fit(X,y,sample_weight=w,verbose=False)
    print(name,metric_dict(yv,np.clip(m.predict(Xv),0,None)),f'bytes {len(m.get_booster().save_raw("ubj"))}',flush=True)
    import joblib
    joblib.dump((qlo,qhi,imp,m),f'/tmp/{name}_s.joblib')
    del m; gc.collect()
import joblib
qlo_t,qhi_t,imp_t,m_t=joblib.load('/tmp/trans_s.joblib')
qlo_r,qhi_r,imp_r,m_r=joblib.load('/tmp/rest_s.joblib')
Xv_raw=va[cols]
pt=np.clip(m_t.predict(imp_t.transform(clip(Xv_raw,qlo_t,qhi_t))),0,None)
pr=np.clip(m_r.predict(imp_r.transform(clip(Xv_raw,qlo_r,qhi_r))),0,None)
# gate by slip EWM (unsupervised, causal, available at inference): high slip -> trans expert
slip=va['transmission_slip_proxy'].to_numpy(dtype=float)
# causal EWM of slip per engine (span 60, past-only via pandas ewm on sorted groups? approximate with rolling mean 60)
slip_s=pd.Series(slip).groupby(va['engine_id'].to_numpy()).transform(lambda s: s.rolling(60,min_periods=10).mean()).to_numpy()
slip_s=np.where(np.isfinite(slip_s),slip_s,0)
# gate: sigmoid around threshold p50? use percentile
thr=float(np.nanpercentile(slip_s,70))
w_t=np.clip((slip_s-thr)/0.02,0,1)  # sharp blend
mix=w_t*pt+(1-w_t)*pr
print(f'slipgate thr={thr:.4f}',metric_dict(yv,mix),flush=True)
# stratified: trans engines vs rest in val
fva=va['engine_id'].map(emanifest['failure_mode_sim']).to_numpy()
for f in ['transmission_wear','thermal_aging','lubrication_degradation','mixed_aging']:
    msk=fva==f
    if msk.sum()>0:
        print(f' val {f} nengines={len(np.unique(va["engine_id"].to_numpy()[msk]))} single-rest-expert MAE {metric_dict(yv[msk],pr[msk])["MAE_hours"]:.3f} mix {metric_dict(yv[msk],mix[msk])["MAE_hours"]:.3f}',flush=True)
print('done',flush=True)
