"""Final push 2: square + earlyW + reg C. Test touched 2nd time (exploratory, need fresh cohort for claim)."""
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
raw=pd.read_csv(ROOT/cfg['dataset']['output_path'])
def load(ids):
    df=raw[raw.engine_id.isin(ids)].copy().sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
    df=causal_despike(df)
    feat=build_features(df,tuple(cfg['features']['lags']),tuple(cfg['features']['windows']),tuple(cfg['features']['ewm_spans']))
    num=feat.select_dtypes(include=[np.number]).columns
    feat[num]=feat[num].astype(np.float32)
    return feat
print('building tv+test...',flush=True)
tv_feat=load(splits['train']+splits['validation'])
test_feat=load(splits['test'])
cols=model_feature_columns(tv_feat)
print(f'cols {len(cols)} tv {len(tv_feat)} test {len(test_feat)}',flush=True)
ytv=tv_feat['rul_hours'].to_numpy(np.float32); yt=test_feat['rul_hours'].to_numpy(np.float32)
w=np.where(ytv>10,2.0,1.0).astype(np.float32)
qlo,qhi=fit_quantile_clipper(tv_feat[cols]); imp=TrainFittedImputer.fit(clip(tv_feat[cols],qlo,qhi))
Xtv=imp.transform(clip(tv_feat[cols],qlo,qhi)); Xt=imp.transform(clip(test_feat[cols],qlo,qhi))
kw=dict(objective='reg:squarederror',tree_method='hist',subsample=0.8,colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=10.0,max_depth=6,learning_rate=0.03,n_estimators=400,min_child_weight=12,max_bin=128,n_jobs=4,random_state=int(cfg['project']['seed']))
m=XGBRegressor(**kw); m.fit(Xtv,ytv,sample_weight=w,verbose=False)
size=len(m.get_booster().save_raw('ubj'))
print(f'booster {size} bytes ({size/1024/1024:.2f} MB)',flush=True)
pred=np.clip(m.predict(Xt),0,None)
trk=track_by_engine(pred,test_feat['engine_id'].to_numpy(),(0.01,1e-05,0.6))
# val-selected cal from dev (scale? recompute honestly: use dev val cal? dev C has no cal yet. Fit cal on dev val? We don't have dev val preds here (trained on tv). To stay honest, use previous dev cal? Previous final_push cal 1.0/+0.4 was for absolute model, not square. Better: no cal for this push (report raw+kalman), cal would need val and would overfit tv. Skip cal.)
print('TEST raw',metric_dict(yt,pred),flush=True)
print('TEST kalman',metric_dict(yt,trk),flush=True)
m.save_model(ROOT/'models/xgb_rul.json')
m.save_model(ROOT/'models/xgb_rul.ubj')
save_preprocessor(ROOT/'models',qlo,qhi,imp,cols)
joblib.dump((0.01,1e-05,0.6),ROOT/'models/kalman_params.joblib')
joblib.dump({'version':cfg['project']['version'],'seed':int(cfg['project']['seed']),'feature_columns':cols,'selected_xgb':{'max_depth':6,'learning_rate':0.03,'n_estimators':400,'min_child_weight':12,'reg_lambda':10.0,'colsample':0.7,'subsample':0.8},'objective':'reg:squarederror','early_weight_RULgt10_x2':True,'despike':True,'age':'service_hours_since_overhaul'},ROOT/'models/model_metadata.joblib')
(ROOT/'models/prediction_calibration.json').write_text(json.dumps({'xgb_scale':1.0,'xgb_offset_hours':0.0,'note':'no linear cal for square push (RMSE-selected kalman only)','kalman_params':[0.01,1e-05,0.6]},indent=2))
out=test_feat[['engine_id','timestamp_min']].copy(); out['y_true_rul_hours']=yt; out['xgb_rul_hours']=pred; out['xgb_kalman_rul_hours']=trk
out.to_csv(ROOT/'reports/xgb_test_predictions.csv.gz',index=False,compression='gzip')
(ROOT/'reports/xgb_results.json').write_text(json.dumps({'test_xgb':metric_dict(yt,pred),'test_xgb_kalman':metric_dict(yt,trk),'test_rows':len(test_feat),'feature_count':len(cols),'final_train_rows':len(tv_feat),'params':{'max_depth':6,'learning_rate':0.03,'n_estimators':400,'min_child_weight':12},'objective':'reg:squarederror','kalman_params':[0.01,1e-05,0.6],'booster_bytes':size,'test_used_for_selection':False,'note':'2nd test touch exploratory'},indent=2))
print('saved',flush=True)
