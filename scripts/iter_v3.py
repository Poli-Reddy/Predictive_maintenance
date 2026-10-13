import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, gc, math
from features import build_features, model_feature_columns, CORE_SENSORS
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer
from metrics import metric_dict
from kalman import track_by_engine
from xgboost import XGBRegressor
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
        # causal median of previous 5 (shifted)
        med=df.groupby('engine_id')[c].transform(lambda s: s.shift(1).rolling(5,min_periods=3).median())
        x=df[c].to_numpy(dtype=np.float32)
        m=med.to_numpy(dtype=np.float32)
        bad=np.isfinite(x)&np.isfinite(m)&(np.abs(x-m)>th)
        # replace spike with median (causal), keep NaNs as-is
        xc=x.copy(); xc[bad]=m[bad]
        out[c]=xc.astype(np.float32)
    return out

def add_v3(df_raw_despiked, feat):
    out=feat.copy()
    # age
    # service_hours/timestamp already in feat
    # expanding cumulative load proxy + excess trends (causal)
    # load_proxy already in feat
    for c in ['load_proxy','coolant_excess_vs_load','trans_oil_excess_vs_load','oil_pressure_deficit_vs_rpm','transmission_slip_proxy','engine_coolant_temp_c','engine_oil_pressure_bar']:
        if c not in feat.columns: continue
        s=feat[c]
        # expanding mean per engine
        expm=s.groupby(feat['engine_id']).transform(lambda x: x.expanding(min_periods=30).mean())
        out[f'{c}__expmean']=expm.astype(np.float32).to_numpy()
        # slope of EWM? long slope 180 already in v2? add slope360 via lag360
    # load-conditional baseline: median of sensor when load_proxy in [0.3,0.55] in first 150 rows
    # compute per-engine baseline from despiked raw + load_proxy from feat
    tmp=pd.DataFrame({'eid':feat['engine_id'].to_numpy(),'ts':feat['timestamp_min'].to_numpy() if 'timestamp_min' in feat.columns else df_raw_despiked['timestamp_min'].to_numpy(),'load':feat['load_proxy'].to_numpy(dtype=np.float32)})
    for c in CORE_SENSORS:
        x=feat[c].to_numpy(dtype=np.float32)
        # per-engine early load-filtered median
        bas={}
        for eid,g in tmp.groupby('eid'):
            m=(g['ts']<=150)&(g['load']>=0.3)&(g['load']<=0.6)
            idx=g.index[m]
            if len(idx)>=10:
                bas[eid]=np.nanmedian(x[idx])
            else:
                # fallback early median
                eidx=g.index[g['ts']<=60]
                bas[eid]=np.nanmedian(x[eidx]) if len(eidx)>0 else 0.0
        base=np.array([bas[e] for e in tmp['eid'].to_numpy()],dtype=np.float32)
        out[f'{c}__lcbase']=base
        out[f'{c}__lc_delta']= (x-base).astype(np.float32)
    num=out.select_dtypes(include=[np.number]).columns
    out[num]=out[num].astype(np.float32)
    return out

print('despiking train/val...')
train_raw=raw[raw.engine_id.isin(splits['train'])].copy()
val_raw=raw[raw.engine_id.isin(splits['validation'])].copy()
train_ds=despike_df(train_raw)
val_ds=despike_df(val_raw)
print('building features...')
train_feat=add_v3(train_ds, build_features(train_ds, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans'])))
val_feat=add_v3(val_ds, build_features(val_ds, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans'])))
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
# allow service_hours + timestamp as age (deployable: operating hours known)
cols=[c for c in train_feat.columns if c not in banned and pd.api.types.is_numeric_dtype(train_feat[c])]
print(f'v3 ncols {len(cols)}')
def sample(df,m=400):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
tr_s=sample(train_feat,400)
for obj in ['reg:squarederror','reg:absoluteerror']:
    for params in [dict(max_depth=8,learning_rate=0.05,n_estimators=200,min_child_weight=10),dict(max_depth=6,learning_rate=0.05,n_estimators=200,min_child_weight=6)]:
        Xtr_raw=tr_s[cols]; ytr=tr_s['rul_hours'].to_numpy(np.float32)
        qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
        Xtr=imp.transform(clip(Xtr_raw,qlo,qhi))
        Xv=imp.transform(clip(val_feat[cols],qlo,qhi)); yv=val_feat['rul_hours'].to_numpy(np.float32)
        m=XGBRegressor(objective=obj,tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_bin=128,n_jobs=4,random_state=20261008,**params)
        m.fit(Xtr,ytr,verbose=False)
        pred=np.clip(m.predict(Xv),0,None)
        print(f'{obj} {params} raw:',metric_dict(yv,pred))
        # calibration scale/offset on val? use nested split: first half engines for fit, second for eval to avoid overfit? quick full-val optimistic
        best=None
        for sc in [0.92,0.96,1.0,1.04]:
            for off in [-0.5,-0.25,0,0.25]:
                cal=np.clip(pred*sc+off,0,None)
                dd=metric_dict(yv,cal)
                score=(-dd['RMSE_hours'],-dd['MAE_hours'])
                if best is None or score>best[0]: best=(score,sc,off,dd)
        print('  calibrated',best[1],best[2],{k:round(v,4) for k,v in best[3].items() if k in ['MAE_hours','RMSE_hours','R2']})
        eids=val_feat['engine_id'].to_numpy()
        for kf in [(0.002,1e-05,0.30),(0.005,1e-05,0.15),(0.01,1e-05,0.30),(0.002,0.0001,0.5)]:
            trk=track_by_engine(pred,eids,kf)
            print('  kf',kf, {k:round(v,4) for k,v in metric_dict(yv,trk).items() if k in ['MAE_hours','RMSE_hours','R2']})
        print('  bytes',len(m.get_booster().save_raw('ubj')))
        del m; gc.collect()
