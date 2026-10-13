import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml
from features import build_features, model_feature_columns
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer
from metrics import metric_dict
from xgboost import XGBRegressor
root=Path('.')
cfg=yaml.safe_load(open('config/config.yaml'))
manifest=pd.read_csv('data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
raw=pd.read_csv(cfg['dataset']['output_path'])
def get_feat(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    return build_features(df, tuple(cfg['features']['lags']), tuple(cfg['features']['windows']), tuple(cfg['features']['ewm_spans']))
train_feat=get_feat(splits['train'])
val_feat=get_feat(splits['validation'])
cols=model_feature_columns(train_feat)
def sample(df,m=300):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
tr_s=sample(train_feat,300)
Xtr_raw=tr_s[cols]; ytr=tr_s['rul_hours'].to_numpy(np.float32)
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi))
Xv=imp.transform(clip(val_feat[cols],qlo,qhi)); yv=val_feat['rul_hours'].to_numpy(np.float32)
m=XGBRegressor(objective='reg:absoluteerror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.05,n_estimators=160,min_child_weight=6,max_bin=256,n_jobs=4,random_state=20261008)
m.fit(Xtr,ytr,verbose=False)
pred=np.clip(m.predict(Xv),0,None)
# bucket analysis
for lo,hi in [(0,2),(2,5),(5,10),(10,15),(15,30)]:
    mask=(yv>=lo)&(yv<hi)
    if mask.sum()==0: continue
    d=metric_dict(yv[mask],pred[mask])
    print(f'RUL bucket {lo}-{hi} n={mask.sum()} MAE={d["MAE_hours"]:.3f} RMSE={d["RMSE_hours"]:.3f} R2={d["R2_percent"]:.1f} w1h={d["within_1h_percent"]:.1f}')
# also error vs timestamp (early vs late in engine life)
# need progress? approximate via service_hours? use timestamp
val_raw=raw[raw.engine_id.isin(splits['validation'])].copy()
# merge pred with timestamp
import pandas as pd
vf=val_feat[['engine_id','timestamp_min']].copy(); vf['y']=yv; vf['p']=pred
# lifetime per engine
lif=manifest.set_index('engine_id')['lifetime_min'].to_dict()
vf['lifetime']=vf['engine_id'].map(lif)
vf['progress']=vf['timestamp_min']/(vf['lifetime']-1)
for lo,hi in [(0,0.3),(0.3,0.6),(0.6,0.8),(0.8,1.0)]:
    mask=(vf['progress']>=lo)&(vf['progress']<hi)
    d=metric_dict(vf.loc[mask,'y'].to_numpy(),vf.loc[mask,'p'].to_numpy())
    print(f'progress {lo}-{hi} n={mask.sum()} MAE={d["MAE_hours"]:.3f} RMSE={d["RMSE_hours"]:.3f} w1h={d["within_1h_percent"]:.1f}')
