from pathlib import Path
import sys, json, gc, joblib
import numpy as np, pandas as pd, yaml
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from features import build_features, model_feature_columns
from preprocess import TrainFittedImputer, clip, fit_quantile_clipper, save_preprocessor
from splits import create_production_splits
from kalman import track_by_engine
from metrics import metric_dict

root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'config/config.yaml').read_text()); raw=pd.read_csv(root/cfg['dataset']['output_path']); man=pd.read_csv(root/'data/processed/engine_manifest.csv'); s=create_production_splits(man,cfg)

def feat(df):
    out=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']))
    num=out.select_dtypes(include=[np.number]).columns; out[num]=out[num].astype(np.float32); return out

def sample(gdf,max_per):
    parts=[]
    for _,g in gdf.groupby('engine_id',sort=False):
        parts.append(g if len(g)<=max_per else g.iloc[np.linspace(0,len(g)-1,max_per,dtype=int)])
    return pd.concat(parts,ignore_index=True)

tr=raw[raw.engine_id.isin(s['train'])].copy(); va=raw[raw.engine_id.isin(s['validation'])].copy(); te=raw[raw.engine_id.isin(s['test'])].copy()
print('raw',len(tr),len(va),len(te),flush=True)
ft=sample(feat(tr),int(cfg['training_sampling']['xgb_max_rows_per_engine'])); fv=feat(va)
cols=model_feature_columns(ft); qlo,qhi=fit_quantile_clipper(ft[cols]); imp=TrainFittedImputer.fit(clip(ft[cols],qlo,qhi)); Xtr=imp.transform(clip(ft[cols],qlo,qhi)); Xv=imp.transform(clip(fv[cols],qlo,qhi)); ytr=ft.rul_hours.to_numpy(np.float32); yv=fv.rul_hours.to_numpy(np.float32)
params={'max_depth':6,'learning_rate':0.05,'n_estimators':120,'min_child_weight':6}
# Dev model; validation is the only selection set.
dev=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=.9,colsample_bytree=.85,reg_alpha=.05,reg_lambda=3,max_bin=128,n_jobs=4,random_state=int(cfg['project']['seed']),**params)
dev.fit(Xtr,ytr,eval_set=[(Xv,yv)],verbose=False); pval=np.clip(dev.predict(Xv),0,None); print('validation',metric_dict(yv,pval),flush=True)
# Tune causal Kalman only on validation predictions.
best_kf=None; best_rmse=np.inf; best_kf_metrics=None
for cand in cfg['kalman']['candidates']:
    trk=track_by_engine(pval,fv.engine_id.to_numpy(),tuple(cand)); m=metric_dict(yv,trk)
    if m['RMSE_hours']<best_rmse: best_rmse=m['RMSE_hours']; best_kf=tuple(cand); best_kf_metrics=m
print('kalman validation',best_kf_metrics,flush=True)
# Final: refit preprocessing/model on train+validation only.
tv=sample(feat(pd.concat([tr,va],ignore_index=True)),int(cfg['training_sampling']['xgb_max_rows_per_engine'])); ftst=feat(te)
qlo,qhi=fit_quantile_clipper(tv[cols]); imp=TrainFittedImputer.fit(clip(tv[cols],qlo,qhi)); Xtv=imp.transform(clip(tv[cols],qlo,qhi)); Xt=imp.transform(clip(ftst[cols],qlo,qhi)); ytv=tv.rul_hours.to_numpy(np.float32); yt=ftst.rul_hours.to_numpy(np.float32)
final=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=.9,colsample_bytree=.85,reg_alpha=.05,reg_lambda=3,max_bin=128,n_jobs=4,random_state=int(cfg['project']['seed']),**params)
final.fit(Xtv,ytv,verbose=False); p=np.clip(final.predict(Xt),0,None); pk=track_by_engine(p,ftst.engine_id.to_numpy(),best_kf)
print('test xgb',metric_dict(yt,p),flush=True); print('test hybrid',metric_dict(yt,pk),flush=True)
md=root/'models'; rep=root/'reports'; md.mkdir(exist_ok=True); rep.mkdir(exist_ok=True)
final.save_model(md/'xgb_rul.json'); save_preprocessor(md,qlo,qhi,imp,cols); joblib.dump(best_kf,md/'kalman_params.joblib'); joblib.dump({'version':cfg['project']['version'],'selected_xgb':params,'feature_columns':cols},md/'model_metadata.joblib')
out=ftst[['engine_id','timestamp_min']].copy(); out['y_true_rul_hours']=yt; out['xgb_rul_hours']=p; out['xgb_kalman_rul_hours']=pk; out.to_csv(rep/'xgb_test_predictions.csv.gz',index=False,compression='gzip')
(rep/'xgb_results.json').write_text(json.dumps({'selected_xgb':params,'validation_xgb':metric_dict(yv,pval),'selected_kalman':best_kf,'validation_xgb_kalman':best_kf_metrics,'test_xgb':metric_dict(yt,p),'test_xgb_kalman':metric_dict(yt,pk),'train_rows_after_sampling':len(ft),'final_train_rows_after_sampling':len(tv),'test_rows':len(ftst),'feature_count':len(cols),'sampling_policy':'Deterministic evenly spaced sampling after full-resolution causal feature generation; validation/test not sampled.'},indent=2))
