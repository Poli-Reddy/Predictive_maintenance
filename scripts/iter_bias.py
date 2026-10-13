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
# bias by bucket
for lo,hi in [(0,2),(2,5),(5,10),(10,15),(15,30)]:
    mask=(yv>=lo)&(yv<hi)
    err=pred[mask]-yv[mask]
    print(f'RUL {lo}-{hi} bias(mean err)={err.mean():+.3f} median={np.median(err):+.3f} std={err.std():.3f} p10={np.percentile(err,10):+.2f} p90={np.percentile(err,90):+.2f}')
# lifetime analysis: per-engine mean pred vs true lifetime
vf=val_feat[['engine_id','timestamp_min']].copy(); vf['y']=yv; vf['p']=pred
lif=manifest.set_index('engine_id')['lifetime_min'].to_dict()
vf['life_true']=vf['engine_id'].map(lif)/60.0
# implied lifetime pred = pred + age (age=timestamp/60)
vf['age']=vf['timestamp_min']/60.0
vf['life_pred']=vf['p']+vf['age']
# per-engine lifetime RMSE (mean over rows vs true)
per_eng=vf.groupby('engine_id').agg(life_true=('life_true','first'),life_pred_mean=('life_pred','mean'),life_pred_median=('life_pred','median'),n=('life_true','size'))
from sklearn.metrics import mean_squared_error
import math
rmse_mean=math.sqrt(mean_squared_error(per_eng['life_true'],per_eng['life_pred_mean']))
rmse_med=math.sqrt(mean_squared_error(per_eng['life_true'],per_eng['life_pred_median']))
print(f'lifetime pred mean RMSE={rmse_mean:.3f} median RMSE={rmse_med:.3f} lifetime std={per_eng["life_true"].std():.3f}')
print(per_eng.head(10).to_string())
