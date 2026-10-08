from pathlib import Path
import sys,json,joblib,gc
import numpy as np,pandas as pd,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from features import build_features,model_feature_columns
from preprocess import TrainFittedImputer,clip,fit_quantile_clipper,save_preprocessor
from splits import create_production_splits
from kalman import track_by_engine
from metrics import metric_dict
from xgboost import XGBRegressor
root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'config/config.yaml').read_text()); raw=pd.read_csv(root/cfg['dataset']['output_path']); man=pd.read_csv(root/'data/processed/engine_manifest.csv'); s=create_production_splits(man,cfg)

def feat(df):
 o=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows'])); n=o.select_dtypes(include=[np.number]).columns; o[n]=o[n].astype(np.float32); return o

def sample(df,m=100):
 parts=[]
 for _,g in df.groupby('engine_id',sort=False): parts.append(g if len(g)<=m else g.iloc[np.linspace(0,len(g)-1,m,dtype=int)])
 return pd.concat(parts,ignore_index=True)
tr=raw[raw.engine_id.isin(s['train'])]; va=raw[raw.engine_id.isin(s['validation'])]; te=raw[raw.engine_id.isin(s['test'])]
params={'max_depth':6,'learning_rate':0.05,'n_estimators':80,'min_child_weight':6}
# DEV fit (train only) and validation-only Kalman tuning.
ft=sample(feat(tr),100); fv=feat(va); cols=model_feature_columns(ft)
qlo,qhi=fit_quantile_clipper(ft[cols]); imp=TrainFittedImputer.fit(clip(ft[cols],qlo,qhi)); Xtr=imp.transform(clip(ft[cols],qlo,qhi)); Xv=imp.transform(clip(fv[cols],qlo,qhi)); ytr=ft.rul_hours.to_numpy(np.float32); yv=fv.rul_hours.to_numpy(np.float32)
dev=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=.9,colsample_bytree=.85,reg_alpha=.05,reg_lambda=3,max_bin=128,n_jobs=4,random_state=int(cfg['project']['seed']),**params)
dev.fit(Xtr,ytr,eval_set=[(Xv,yv)],verbose=False); pval=np.clip(dev.predict(Xv),0,None); val_xgb=metric_dict(yv,pval)
best_kf=None; best_kf_metrics=None
for cand in cfg['kalman']['candidates']:
 pk=track_by_engine(pval,fv.engine_id.to_numpy(),tuple(cand)); mm=metric_dict(yv,pk)
 if best_kf is None or mm['RMSE_hours'] < best_kf_metrics['RMSE_hours']: best_kf=tuple(cand); best_kf_metrics=mm
(root/'reports/kalman_selection.json').write_text(json.dumps({'best_kf':best_kf,'validation_xgb':val_xgb,'validation_xgb_kalman':best_kf_metrics},indent=2))
print('validation',val_xgb, 'hybrid',best_kf_metrics,flush=True)
del dev,Xtr,Xv,pval,ft,fv,tr,va,imp,qlo,qhi; gc.collect()
# Final model on train+validation after all choices are frozen; test remains unseen until this point.
tv=sample(feat(pd.concat([raw[raw.engine_id.isin(s['train'])],raw[raw.engine_id.isin(s['validation'])]],ignore_index=True)),100); ftst=feat(te)
qlo,qhi=fit_quantile_clipper(tv[cols]); imp=TrainFittedImputer.fit(clip(tv[cols],qlo,qhi)); Xtv=imp.transform(clip(tv[cols],qlo,qhi)); Xt=imp.transform(clip(ftst[cols],qlo,qhi)); ytv=tv.rul_hours.to_numpy(np.float32); yt=ftst.rul_hours.to_numpy(np.float32)
final=XGBRegressor(objective='reg:squarederror',tree_method='hist',subsample=.9,colsample_bytree=.85,reg_alpha=.05,reg_lambda=3,max_bin=128,n_jobs=4,random_state=int(cfg['project']['seed']),**params)
final.fit(Xtv,ytv,verbose=False); p=np.clip(final.predict(Xt),0,None); pk=track_by_engine(p,ftst.engine_id.to_numpy(),best_kf)
res={'test_xgb':metric_dict(yt,p),'test_xgb_kalman':metric_dict(yt,pk),'test_rows':len(ftst),'feature_count':len(cols),'final_train_rows':len(tv),'params':params,'kalman_params':best_kf,'test_used_for_selection':False,'training_row_sampling':'100 evenly-spaced rows per engine AFTER full-resolution causal feature generation'}
md=root/'models'; rep=root/'reports'; md.mkdir(exist_ok=True); rep.mkdir(exist_ok=True)
final.save_model(md/'xgb_rul.json'); save_preprocessor(md,qlo,qhi,imp,cols); joblib.dump(best_kf,md/'kalman_params.joblib'); joblib.dump({'version':cfg['project']['version'],'selected_xgb':params,'feature_columns':cols,'trained_on':'train+validation engine cohorts only'},md/'model_metadata.joblib')
out=ftst[['engine_id','timestamp_min']].copy(); out['y_true_rul_hours']=yt; out['xgb_rul_hours']=p; out['xgb_kalman_rul_hours']=pk; out.to_csv(rep/'xgb_test_predictions.csv.gz',index=False,compression='gzip')
(rep/'xgb_results.json').write_text(json.dumps({'validation_xgb':val_xgb,'validation_xgb_kalman':best_kf_metrics,**res},indent=2))
(root/'data/processed/split_audit.json').write_text(json.dumps({'train_engines':s['train'],'validation_engines':s['validation'],'test_engines':s['test'],'strategy':'chronological_engine_cohort','test_used_for_selection':False,'production_inference_uses_current_and_past_telemetry_only':True},indent=2))
print(json.dumps(res,indent=2))
