import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc, math
from features import build_features, model_feature_columns, CORE_SENSORS
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer
from metrics import metric_dict
from xgboost import XGBRegressor
root=Path('.')
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
raw=pd.read_csv(cfg['dataset']['output_path'])
print('building base feats full...')
def get(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    return build_features(df, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans']))
train_feat=get(splits['train']); val_feat=get(splits['validation'])
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in train_feat.columns if c not in banned and pd.api.types.is_numeric_dtype(train_feat[c])]
print(f'ncols {len(cols)}')
# residual target: RUL - (mean_life - age)
train_age=train_feat['service_hours_since_overhaul'].to_numpy(dtype=float)
val_age=val_feat['service_hours_since_overhaul'].to_numpy(dtype=float)
mean_life=manifest[manifest.engine_id.isin(splits['train'])]['lifetime_min'].mean()/60.0
print(f'mean_life {mean_life:.3f}')
ytr_rul=train_feat['rul_hours'].to_numpy(np.float32)
yv=val_feat['rul_hours'].to_numpy(np.float32)
ytr_res=ytr_rul-(mean_life-train_age)
print(f'resid std {ytr_res.std():.3f} vs rul std {ytr_rul.std():.3f}')
Xtr_raw=train_feat[cols]; Xv_raw=val_feat[cols]
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi)); Xv=imp.transform(clip(Xv_raw,qlo,qhi))
# weight early higher? w=1+1.5*(y>10)
w=np.where(ytr_rul>10,2.0,1.0).astype(np.float32)
for use_w in [False,True]:
    for obj in ['reg:absoluteerror','reg:squarederror']:
        m=XGBRegressor(objective=obj,tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.03,n_estimators=400,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008)
        if use_w: m.fit(Xtr,ytr_res,sample_weight=w,verbose=False)
        else: m.fit(Xtr,ytr_res,verbose=False)
        res_pred=m.predict(Xv)
        rul_pred=np.clip((mean_life-val_age)+res_pred,0,None)
        d=metric_dict(yv,rul_pred)
        print(f'RESID obj={obj} weighted={use_w} MAE {d["MAE_hours"]:.4f} RMSE {d["RMSE_hours"]:.4f} R2 {d["R2_percent"]:.2f} bytes {len(m.get_booster().save_raw("ubj"))}')
        del m; gc.collect()
# direct baseline for comparison same params
m0=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.03,n_estimators=400,min_child_weight=6,max_bin=128,n_jobs=4,random_state=20261008)
m0.fit(Xtr,ytr_rul,verbose=False)
print('DIRECT',metric_dict(yv,np.clip(m0.predict(Xv),0,None)))
