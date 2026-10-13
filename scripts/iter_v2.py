"""Iterative XGB experiment - v2 features, no LSTM, val-only tuning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, json, gc
from features import build_features, model_feature_columns, CORE_SENSORS
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer
from metrics import metric_dict
from kalman import track_by_engine
from xgboost import XGBRegressor

root = Path('.')
cfg = yaml.safe_load(open('config/config.yaml'))
manifest = pd.read_csv('data/processed/engine_manifest.csv')
splits = create_production_splits(manifest, cfg)
raw = pd.read_csv(cfg['dataset']['output_path'])

def add_v2(feat_raw, feat):
    """Add causal per-engine baseline + age + long-window features. All causal."""
    # age (known at inference: operating hours)
    # service_hours and timestamp already in feat (from raw concat)
    out = feat.copy()
    # per-engine early baseline (first 60 rows mean per sensor), causal: constant after t>=60
    for c in CORE_SENSORS:
        # compute per-engine baseline via transform
        # feat has engine_id, timestamp_min sorted
        baselines = feat_raw.groupby('engine_id')[c].apply(lambda s: s.iloc[:60].mean(skipna=True))
        # map
        base = feat['engine_id'].map(baselines).astype(np.float32)
        out[f'{c}__base60'] = base
        out[f'{c}__delta_base60'] = (feat[c].to_numpy(dtype=np.float32) - base.to_numpy(dtype=np.float32)).astype(np.float32)
        # long window 180 mean/std/slope via simple rolling on already-sorted per-engine groups
        # use pandas rolling per engine (causal, min_periods)
        g = feat_raw.groupby('engine_id')[c]
        # mean180
        mean180 = g.transform(lambda s: s.rolling(180, min_periods=45).mean())
        std180 = g.transform(lambda s: s.rolling(180, min_periods=45).std())
        out[f'{c}__mean180'] = mean180.astype(np.float32).to_numpy()
        out[f'{c}__std180'] = std180.astype(np.float32).to_numpy()
        # slope180
        lag180 = g.shift(180)
        # need aligned: feat_raw order == feat order? build_features sorts by engine,timestamp, same as feat_raw sorted
        slope = (feat[c].to_numpy(dtype=np.float32) - lag180.to_numpy(dtype=np.float32)) / 180.0
        out[f'{c}__slope180'] = slope.astype(np.float32)
    # missing count last 60 per sensor
    for c in CORE_SENSORS:
        miss = feat_raw[c].isna().astype(np.float32)
        cnt = miss.groupby(feat_raw['engine_id']).transform(lambda s: s.rolling(60, min_periods=1).sum())
        out[f'{c}__misscnt60'] = cnt.astype(np.float32).to_numpy()
    # numeric cast
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].astype(np.float32)
    return out

def get_feat(ids):
    df = raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    base = build_features(df, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans']))
    v2 = add_v2(df, base)
    return v2

def allowed_cols(feat):
    banned = {'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','timestamp_min','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
    # allow service_hours + timestamp? timestamp banned originally, but age is legitimate for deployment (operating hours known)
    # keep production_batch banned (chronological generalisation), keep env/failure banned
    banned = banned - {'service_hours_since_overhaul','timestamp_min'}
    # timestamp_min now allowed as age; service_hours allowed
    cols = [c for c in feat.columns if c not in banned and pd.api.types.is_numeric_dtype(feat[c])]
    return cols

print('building train/val feats...')
train_feat = get_feat(splits['train'])
val_feat = get_feat(splits['validation'])
cols = allowed_cols(train_feat)
print(f'v2 ncols {len(cols)} vs base 128')
# sample for speed first: 200/engine
def sample(df,m=200):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        if len(g)<=m: pcs.append(g)
        else: pcs.append(g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)

tr_s = sample(train_feat, 300)
Xtr_raw = tr_s[cols]; ytr = tr_s['rul_hours'].to_numpy(np.float32)
qlo,qhi = fit_quantile_clipper(Xtr_raw); imp = TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr = imp.transform(clip(Xtr_raw,qlo,qhi))
Xv = imp.transform(clip(val_feat[cols],qlo,qhi)); yv = val_feat['rul_hours'].to_numpy(np.float32)
eids = val_feat['engine_id'].to_numpy()
for params in [
    dict(max_depth=6, learning_rate=0.05, n_estimators=160, min_child_weight=6),
    dict(max_depth=8, learning_rate=0.05, n_estimators=160, min_child_weight=10),
    dict(max_depth=8, learning_rate=0.05, n_estimators=250, min_child_weight=10),
]:
    m = XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_bin=128,n_jobs=4,random_state=20261008,**params)
    m.fit(Xtr,ytr,verbose=False)
    pred = np.clip(m.predict(Xv),0,None)
    print(params, metric_dict(yv,pred))
    for kf in [(0.002,1e-05,0.30),(0.01,1e-05,0.30),(0.005,0.0002,0.15)]:
        trk = track_by_engine(pred, eids, kf)
        mk = metric_dict(yv,trk)
        print('  kalman',kf,'MAE',round(mk['MAE_hours'],4),'RMSE',round(mk['RMSE_hours'],4),'R2',round(mk['R2_percent'],2),'w1h',round(mk['within_1h_percent'],2))
    print('  booster bytes', len(m.get_booster().save_raw('ubj')))
    del m; gc.collect()
