import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc
from features import build_features
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer, causal_despike
from metrics import metric_dict
from kalman import track_by_engine
from xgboost import XGBRegressor
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
raw=pd.read_csv(cfg['dataset']['output_path'])
print('building full train/val...',flush=True)
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
print(f'ncols {len(cols)} train {len(tr)} val {len(va)}',flush=True)
ytr=tr['rul_hours'].to_numpy(np.float32); yv=va['rul_hours'].to_numpy(np.float32)
w=np.where(ytr>10,2.0,1.0).astype(np.float32)
Xtr_raw=tr[cols]; Xv_raw=va[cols]
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(Xv_raw,qlo,qhi))
eids=va['engine_id'].to_numpy()
configs=[
 ('A_sq_lr003_n500',dict(objective='reg:squarederror',max_depth=6,learning_rate=0.03,n_estimators=500,min_child_weight=6,reg_lambda=3.0,max_bin=128,colsample_bytree=0.85,subsample=0.9)),
 ('B_sq_lr002_n700_b64',dict(objective='reg:squarederror',max_depth=6,learning_rate=0.02,n_estimators=700,min_child_weight=6,reg_lambda=5.0,max_bin=64,colsample_bytree=0.85,subsample=0.9)),
 ('C_sq_reg_lr003_n400',dict(objective='reg:squarederror',max_depth=6,learning_rate=0.03,n_estimators=400,min_child_weight=12,reg_lambda=10.0,max_bin=128,colsample_bytree=0.7,subsample=0.8)),
]
for name,kw in configs:
    m=XGBRegressor(tree_method='hist',reg_alpha=0.05,n_jobs=4,random_state=20261008,**kw)
    m.fit(Xtr,ytr,sample_weight=w,verbose=False)
    pred=np.clip(m.predict(Xv),0,None)
    d=metric_dict(yv,pred)
    print(f'{name} raw MAE {d["MAE_hours"]:.4f} RMSE {d["RMSE_hours"]:.4f} R2 {d["R2_percent"]:.2f} bytes {len(m.get_booster().save_raw("ubj"))}',flush=True)
    # RMSE-selected kalman (goal <2.0)
    best=None
    for kf in [(0.002,1e-05,0.30),(0.01,1e-05,0.30),(0.01,1e-05,0.60),(0.005,1e-05,0.15)]:
        trk=track_by_engine(pred,eids,tuple(kf))
        dd=metric_dict(yv,trk)
        s=(-dd['RMSE_hours'],-dd['MAE_hours'])
        if best is None or s>best[0]: best=(s,kf,dd)
    print(f'  bestkf {best[1]} RMSE {best[2]["RMSE_hours"]:.4f} R2 {best[2]["R2_percent"]:.2f}',flush=True)
    del m; gc.collect()
print('done',flush=True)
