"""Final push: despike + age + depth6/500 + RMSE-selected Kalman + linear cal. Test touched ONCE."""
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
manifest=pd.read_csv(ROOT/'data/processed/engine_manifest.csv')
splits=create_production_splits(manifest,cfg)
print({k:len(v) for k,v in splits.items()},flush=True)
raw=pd.read_csv(ROOT/cfg['dataset']['output_path'])
def load_ids(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    df=causal_despike(df)
    feat=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']),tuple(cfg['features']['ewm_spans']))
    num=feat.select_dtypes(include=[np.number]).columns
    feat[num]=feat[num].astype(np.float32)
    return feat
print('building train...',flush=True)
train_feat=load_ids(splits['train'])
print('building val...',flush=True)
val_feat=load_ids(splits['validation'])
cols=model_feature_columns(train_feat)
print(f'feature_count {len(cols)} (expect 129 with service_hours)',flush=True)
Xtr_raw=train_feat[cols]; ytr=train_feat['rul_hours'].to_numpy(np.float32)
qlo,qhi=fit_quantile_clipper(Xtr_raw); imp=TrainFittedImputer.fit(clip(Xtr_raw,qlo,qhi))
Xtr=imp.transform(clip(Xtr_raw,qlo,qhi))
Xv=imp.transform(clip(val_feat[cols],qlo,qhi)); yv=val_feat['rul_hours'].to_numpy(np.float32)
print(f'dev train rows {len(Xtr)} val rows {len(Xv)}',flush=True)
params=dict(objective='reg:absoluteerror',tree_method='hist',subsample=0.90,colsample_bytree=0.85,reg_alpha=0.05,reg_lambda=3.0,max_depth=6,learning_rate=0.03,n_estimators=500,min_child_weight=6,max_bin=128,n_jobs=4,random_state=int(cfg['project']['seed']))
dev=XGBRegressor(**params); dev.fit(Xtr,ytr,verbose=False)
val_pred=np.clip(dev.predict(Xv),0,None)
print('dev val raw',metric_dict(yv,val_pred),flush=True)
eids=val_feat['engine_id'].to_numpy()
cands=[(0.001,1e-05,0.15),(0.002,1e-05,0.15),(0.002,1e-05,0.30),(0.005,1e-05,0.30),(0.01,1e-05,0.30),(0.002,1e-04,0.30),(0.005,2e-04,0.15),(0.01,1e-05,0.60)]
best=None
for c in cands:
    trk=track_by_engine(val_pred,eids,tuple(c))
    m=metric_dict(yv,trk)
    score=(-m['RMSE_hours'],-m['MAE_hours'],m['within_1h_percent'])
    if best is None or score>best[0]: best=(score,tuple(c),m,trk)
_,best_kf,best_km,val_trk=best
print(f'selected kalman {best_kf} {best_km}',flush=True)
# linear calibration on val tracked (val-only, same protocol as LSTM cal)
bsc=None
for sc in np.arange(0.90,1.081,0.02):
    for off in np.arange(-0.60,0.61,0.10):
        cal=np.clip(val_trk*sc+off,0,None)
        d=metric_dict(yv,cal)
        s=(-d['RMSE_hours'],-d['MAE_hours'])
        if bsc is None or s>bsc[0]: bsc=(s,float(sc),float(off),d)
_,csc,coff,cal_m=bsc
print(f'cal scale {csc:.3f} offset {coff:.3f} {cal_m}',flush=True)
del dev,val_pred,val_trk,Xtr,Xv,train_feat,val_feat; gc.collect()
print('building train+val for final...',flush=True)
tv_feat=load_ids(splits['train']+splits['validation'])
print('building test (ONE touch)...',flush=True)
test_feat=load_ids(splits['test'])
assert model_feature_columns(tv_feat)==cols,'schema changed'
Xtv_raw=tv_feat[cols]; ytv=tv_feat['rul_hours'].to_numpy(np.float32)
qlo2,qhi2=fit_quantile_clipper(Xtv_raw); imp2=TrainFittedImputer.fit(clip(Xtv_raw,qlo2,qhi2))
Xtv=imp2.transform(clip(Xtv_raw,qlo2,qhi2))
Xt=imp2.transform(clip(test_feat[cols],qlo2,qhi2)); yt=test_feat['rul_hours'].to_numpy(np.float32)
print(f'final train rows {len(Xtv)} test rows {len(Xt)}',flush=True)
final=XGBRegressor(**params); final.fit(Xtv,ytv,verbose=False)
size=len(final.get_booster().save_raw('ubj'))
print(f'booster bytes {size} ({size/1024/1024:.3f} MB)',flush=True)
pred=np.clip(final.predict(Xt),0,None)
trk=track_by_engine(pred,test_feat['engine_id'].to_numpy(),best_kf)
cal=np.clip(trk*csc+coff,0,None)
print('TEST raw',metric_dict(yt,pred),flush=True)
print('TEST kalman',metric_dict(yt,trk),flush=True)
print('TEST kalman+cal',metric_dict(yt,cal),flush=True)
# save artifacts (overwrite production)
mdir=ROOT/'models'; rdir=ROOT/'reports'; pdir=ROOT/'data/processed'
final.save_model(mdir/'xgb_rul.json'); save_preprocessor(mdir,qlo2,qhi2,imp2,cols); joblib.dump(best_kf,mdir/'kalman_params.joblib')
joblib.dump({'version':cfg['project']['version'],'seed':int(cfg['project']['seed']),'feature_columns':cols,'selected_xgb':{'max_depth':6,'learning_rate':0.03,'n_estimators':500,'min_child_weight':6},'objective':'reg:absoluteerror','despike':True,'age_feature':'service_hours_since_overhaul'},mdir/'model_metadata.joblib')
(mdir/'prediction_calibration.json').write_text(json.dumps({'xgb_scale':csc,'xgb_offset_hours':coff,'xgb_selection_metric':'RMSE_hours','kalman_params':list(best_kf),'selection_data':'validation engines only'},indent=2))
out=test_feat[['engine_id','timestamp_min']].copy(); out['y_true_rul_hours']=yt; out['xgb_rul_hours']=pred; out['xgb_kalman_rul_hours']=trk; out['xgb_kalman_cal_rul_hours']=cal
out.to_csv(rdir/'xgb_test_predictions.csv.gz',index=False,compression='gzip')
(rdir/'xgb_results.json').write_text(json.dumps({'validation_xgb_raw':metric_dict(yv if False else ytv[:0],ytv[:0]) if False else {},'test_xgb':metric_dict(yt,pred),'test_xgb_kalman':metric_dict(yt,trk),'test_xgb_kalman_cal':metric_dict(yt,cal),'test_rows':len(test_feat),'feature_count':len(cols),'final_train_rows':len(tv_feat),'params':{'max_depth':6,'learning_rate':0.03,'n_estimators':500,'min_child_weight':6},'objective':'reg:absoluteerror','kalman_params':list(best_kf),'booster_bytes':size,'test_used_for_selection':False},indent=2))
print('saved',flush=True)
