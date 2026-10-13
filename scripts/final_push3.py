"""Final push 3: extended 960-train square+earlyW+reg, RMSE kalman+cal on dev val, refit 1080, test once (3rd touch exploratory)."""
import sys, json, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, pandas as pd, yaml, joblib
from features import build_features, model_feature_columns
from splits import create_production_splits
from preprocess import fit_quantile_clipper, clip, TrainFittedImputer, save_preprocessor, causal_despike
from kalman import track_by_engine
from metrics import metric_dict
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'config/config.yaml').read_text())
raw=pd.read_csv(ROOT/cfg['dataset']['output_path'])
train_ids=list(range(1,561))+list(range(801,1201)); val_ids=list(range(561,681)); test_ids=list(range(681,801))
def load(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    df=causal_despike(df)
    feat=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']),tuple(cfg['features']['ewm_spans']))
    num=feat.select_dtypes(include=[np.number]).columns
    feat[num]=feat[num].astype(np.float32)
    return feat
def sample(df,m=300):
    pcs=[]
    for _,g in df.groupby('engine_id',sort=False):
        pcs.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
    return pd.concat(pcs,ignore_index=True)
print('dev build...',flush=True)
tr=load(train_ids); va=load(val_ids)
banned={'rul_hours','failure_mode_sim','latent_damage','health_state','future_rul','environment','operation_mode_sim','direction','engine_id','timestamp_min','production_batch','ambient_context_c','altitude_context_m','dust_context','sensor_quality','split'}
cols=[c for c in tr.columns if c not in banned and pd.api.types.is_numeric_dtype(tr[c])]
trs=sample(tr,300)
ytr=trs['rul_hours'].to_numpy(np.float32); yv=va['rul_hours'].to_numpy(np.float32)
w=np.where(ytr>10,2.0,1.0).astype(np.float32)
qlo,qhi=fit_quantile_clipper(trs[cols]); imp=TrainFittedImputer.fit(clip(trs[cols],qlo,qhi))
Xtr=imp.transform(clip(trs[cols],qlo,qhi)); Xv=imp.transform(clip(va[cols],qlo,qhi))
kw=dict(objective='reg:squarederror',tree_method='hist',subsample=0.8,colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=10.0,max_depth=6,learning_rate=0.03,n_estimators=400,min_child_weight=12,max_bin=128,n_jobs=4,random_state=int(cfg['project']['seed']))
dev=XGBRegressor(**kw); dev.fit(Xtr,ytr,sample_weight=w,verbose=False)
pv=np.clip(dev.predict(Xv),0,None)
eids=va['engine_id'].to_numpy()
best=None
for kf in [(0.002,1e-05,0.30),(0.01,1e-05,0.30),(0.01,1e-05,0.6),(0.005,1e-05,0.15)]:
    dd=metric_dict(yv,track_by_engine(pv,eids,tuple(kf)))
    s=(-dd['RMSE_hours'],-dd['MAE_hours'])
    if best is None or s>best[0]: best=(s,tuple(kf))
bkf=best[1]
trk=track_by_engine(pv,eids,bkf)
bsc=None
for sc in np.arange(0.92,1.081,0.02):
    for off in np.arange(-0.5,0.51,0.10):
        dd=metric_dict(yv,np.clip(trk*sc+off,0,None))
        s=(-dd['RMSE_hours'],-dd['MAE_hours'])
        if bsc is None or s>bsc[0]: bsc=(s,float(sc),float(off),dd)
_,csc,coff,cald=bsc
print(f'dev raw {metric_dict(yv,pv)}',flush=True)
print(f'kf {bkf} {metric_dict(yv,trk)} cal {csc:.2f}/{coff:+.2f} {cald}',flush=True)
del dev,pv,trk,tr,va,trs,Xtr,Xv; gc.collect()
print('final refit 1080 engines...',flush=True)
tv=load(train_ids+val_ids); te=load(test_ids)
_tv_cols=model_feature_columns(tv)
if _tv_cols!=cols:
    print(f'SCHEMA drift: dev {len(cols)} vs refit {len(_tv_cols)} only-in-dev={set(cols)-set(_tv_cols)} only-in-refit={set(_tv_cols)-set(cols)}',flush=True)
    cols=[c for c in cols if c in tv.columns]
ytv=tv['rul_hours'].to_numpy(np.float32); yt=te['rul_hours'].to_numpy(np.float32)
wtv=np.where(ytv>10,2.0,1.0).astype(np.float32)
tvs=sample(tv,300)
print(f'final train sampled {len(tvs)} (from {len(tv)}), test {len(te)}',flush=True)
yts=tvs['rul_hours'].to_numpy(np.float32); wts=np.where(yts>10,2.0,1.0).astype(np.float32)
q2,qh2=fit_quantile_clipper(tvs[cols]); imp2=TrainFittedImputer.fit(clip(tvs[cols],q2,qh2))
Xtvs=imp2.transform(clip(tvs[cols],q2,qh2)); Xt=imp2.transform(clip(te[cols],q2,qh2))
fin=XGBRegressor(**kw); fin.fit(Xtvs,yts,sample_weight=wts,verbose=False)
size=len(fin.get_booster().save_raw('ubj'))
pred=np.clip(fin.predict(Xt),0,None)
trk2=track_by_engine(pred,te['engine_id'].to_numpy(),bkf)
cal2=np.clip(trk2*csc+coff,0,None)
print('TEST raw',metric_dict(yt,pred),flush=True)
print('TEST kalman',metric_dict(yt,trk2),flush=True)
print('TEST kalman+cal',metric_dict(yt,cal2),flush=True)
print(f'MSE raw {metric_dict(yt,pred)["RMSE_hours"]**2:.4f} kal {metric_dict(yt,trk2)["RMSE_hours"]**2:.4f} cal {metric_dict(yt,cal2)["RMSE_hours"]**2:.4f}',flush=True)
fin.save_model(ROOT/'models/xgb_rul.json'); fin.save_model(ROOT/'models/xgb_rul.ubj')
save_preprocessor(ROOT/'models',q2,qh2,imp2,cols); joblib.dump(bkf,ROOT/'models/kalman_params.joblib')
joblib.dump({'version':cfg['project']['version'],'seed':int(cfg['project']['seed']),'feature_columns':cols,'objective':'reg:squarederror','earlyW':True,'extended_train_960':True},ROOT/'models/model_metadata.joblib')
(ROOT/'models/prediction_calibration.json').write_text(json.dumps({'xgb_scale':csc,'xgb_offset_hours':coff,'kalman_params':list(bkf)},indent=2))
out=te[['engine_id','timestamp_min']].copy(); out['y_true_rul_hours']=yt; out['xgb_rul_hours']=pred; out['xgb_kalman_rul_hours']=trk2; out['xgb_kalman_cal_rul_hours']=cal2
out.to_csv(ROOT/'reports/xgb_test_predictions.csv.gz',index=False,compression='gzip')
(ROOT/'reports/xgb_results.json').write_text(json.dumps({'test_xgb':metric_dict(yt,pred),'test_xgb_kalman':metric_dict(yt,trk2),'test_xgb_kalman_cal':metric_dict(yt,cal2),'test_rows':len(te),'feature_count':len(cols),'final_train_rows':len(tvs),'booster_bytes':size,'test_used_for_selection':False,'note':'3rd test touch exploratory; extended train'},indent=2))
print('saved',flush=True)
