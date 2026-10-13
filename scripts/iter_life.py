import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np, yaml, math
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
print('building...')
train_feat=get_feat(splits['train'])
val_feat=get_feat(splits['validation'])
cols=model_feature_columns(train_feat)
# map lifetime
lif=manifest.set_index('engine_id')['lifetime_min'].to_dict()
for df in [train_feat, val_feat]:
    df['lifetime_h']=df['engine_id'].map(lif)/60.0
    df['age_h']=df['timestamp_min']/60.0
def sample(df,m=300):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
tr_s=sample(train_feat,300)
# Experiment A: predict lifetime from same features
Xtr_raw=tr_s[cols]; ytr_life=tr_s['lifetime_h'].to_numpy(np.float32)
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi))
Xv=imp.transform(clip(val_feat[cols],qlo,qhi))
yv=val_feat['rul_hours'].to_numpy(np.float32)
age_v=val_feat['age_h'].to_numpy()
life_true_v=val_feat['lifetime_h'].to_numpy()
for params in [dict(max_depth=6,learning_rate=0.05,n_estimators=160,min_child_weight=6),dict(max_depth=8,learning_rate=0.05,n_estimators=160,min_child_weight=10)]:
    m=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=0.9,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_bin=128,n_jobs=4,random_state=20261008,**params)
    m.fit(Xtr,ytr_life,verbose=False)
    life_pred=np.clip(m.predict(Xv),0,None)
    # per-row RUL via lifetime-age
    rul_pred=np.clip(life_pred-age_v,0,None)
    print(f'lifetime model {params}')
    print('  lifetime per-row RMSE',math.sqrt(np.mean((life_pred-life_true_v)**2)),'MAE',np.mean(np.abs(life_pred-life_true_v)))
    print('  implied RUL',metric_dict(yv,rul_pred))
    # causal smoothed lifetime: expanding mean of life_pred per engine (past only)
    vf=pd.DataFrame({'eid':val_feat['engine_id'].to_numpy(),'life_pred':life_pred,'age':age_v,'y':yv})
    vf=vf.sort_values(['eid','age'])
    # expanding mean with min 30 rows burn-in, else raw
    def smooth(g):
        vals=g['life_pred'].to_numpy()
        c=np.cumsum(vals)/np.arange(1,len(vals)+1)
        # burn-in: first 30 use raw? use expanding anyway (causal)
        return pd.Series(c,index=g.index)
    vf['life_smooth']=vf.groupby('eid',sort=False).apply(smooth,include_groups=False).reset_index(level=0,drop=True).sort_index().to_numpy() if False else vf.groupby('eid')['life_pred'].transform(lambda s: s.expanding().mean())
    rul_smooth=np.clip(vf['life_smooth'].to_numpy()-age_v[np.argsort(np.argsort(vf.index))] if False else 0,0,None)
    # simpler: sort already, transform expanding mean preserves order
    vf2=vf.sort_values(['eid','age']).copy()
    vf2['life_smooth']=vf2.groupby('eid')['life_pred'].transform(lambda s: s.expanding(min_periods=1).mean())
    # need to map back to original order (vf2 is sorted same as Xv? Xv order is engine,timestamp sorted, same)
    rul_smooth=np.clip(vf2['life_smooth'].to_numpy()-vf2['age'].to_numpy(),0,None)
    print('  smoothed RUL',metric_dict(vf2['y'].to_numpy(),rul_smooth))
    print('  booster bytes',len(m.get_booster().save_raw('ubj')))
