import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc, math
from features import build_features, model_feature_columns, CORE_SENSORS
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer
from metrics import metric_dict
from xgboost import XGBRegressor
from sklearn.isotonic import IsotonicRegression
root=Path('.')
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
raw=pd.read_csv(cfg['dataset']['output_path'])
SPIKE_THRESH={'engine_rpm':120,'transmission_rpm':120,'engine_coolant_temp_c':8,'transmission_oil_temp_c':8,'engine_oil_pressure_bar':0.8,'clutch_pressure_bar':1.2,'solenoid_current_a':0.3,'vehicle_speed_kph':8}
def despike_df(df):
    df=df.sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    out=df.copy()
    for c in CORE_SENSORS:
        th=SPIKE_THRESH[c]
        med=df.groupby('engine_id')[c].transform(lambda s: s.shift(1).rolling(5,min_periods=3).median())
        x=df[c].to_numpy(dtype=np.float32); m=med.to_numpy(dtype=np.float32)
        bad=np.isfinite(x)&np.isfinite(m)&(np.abs(x-m)>th)
        xc=x.copy(); xc[bad]=m[bad]
        out[c]=xc.astype(np.float32)
    return out
print('despike+features (reuse v3-lite: despike + age only, no heavy lcbase to save time)...')
train_raw=raw[raw.engine_id.isin(splits['train'])].copy()
val_raw=raw[raw.engine_id.isin(splits['validation'])].copy()
train_ds=despike_df(train_raw); val_ds=despike_df(val_raw)
train_feat=build_features(train_ds, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans']))
val_feat=build_features(val_ds, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans']))
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in train_feat.columns if c not in banned and pd.api.types.is_numeric_dtype(train_feat[c])]
print(f'ncols {len(cols)} (base+age, despiked)')
# FULL train rows (no sampling) for final push
Xtr_raw=train_feat[cols]; ytr=train_feat['rul_hours'].to_numpy(np.float32)
print(f'full train rows {len(Xtr_raw)}')
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi))
Xv=imp.transform(clip(val_feat[cols],qlo,qhi)); yv=val_feat['rul_hours'].to_numpy(np.float32)
print('train matrix done, starting XGB fits...')
for params in [dict(max_depth=6,learning_rate=0.03,n_estimators=500,min_child_weight=6),dict(max_depth=6,learning_rate=0.05,n_estimators=300,min_child_weight=6),dict(max_depth=8,learning_rate=0.03,n_estimators=200,min_child_weight=10)]:
    m=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_bin=128,n_jobs=4,random_state=20261008,**params)
    m.fit(Xtr,ytr,verbose=False)
    pred=np.clip(m.predict(Xv),0,None)
    base=metric_dict(yv,pred)
    print(f"{params} raw MAE {base['MAE_hours']:.4f} RMSE {base['RMSE_hours']:.4f} R2 {base['R2_percent']:.2f} bytes {len(m.get_booster().save_raw('ubj'))}")
    # isotonic calibration with 2-fold engine split to avoid overfit
    eids=val_feat['engine_id'].to_numpy()
    uniq=np.unique(eids)
    half=set(uniq[:len(uniq)//2].tolist())
    mask1=np.array([e in half for e in eids]); mask2=~mask1
    ir=IsotonicRegression(out_of_bounds='clip')
    ir.fit(pred[mask1],yv[mask1])
    cal=ir.predict(pred)
    # eval on other half only (honest)
    h=metric_dict(yv[mask2],cal[mask2])
    print(f"  isotonic honest-half MAE {h['MAE_hours']:.4f} RMSE {h['RMSE_hours']:.4f} R2 {h['R2_percent']:.2f}")
    # optimistic full-val
    ir2=IsotonicRegression(out_of_bounds='clip'); ir2.fit(pred,yv)
    print(f"  isotonic optimistic-full RMSE {metric_dict(yv,ir2.predict(pred))['RMSE_hours']:.4f}")
    del m; gc.collect()
