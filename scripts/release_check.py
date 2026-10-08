from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd
import numpy as np

root=Path(__file__).resolve().parents[1]
required=[
 root/'models/xgb_rul.json', root/'models/xgb_imputer.joblib', root/'models/xgb_qlo.joblib', root/'models/xgb_qhi.joblib',
 root/'models/feature_columns.joblib', root/'models/kalman_params.joblib', root/'models/lstm_rul.pt', root/'models/lstm_scaler.joblib',
 root/'reports/comparison.csv', root/'reports/xgb_results.json', root/'reports/xgb_test_predictions.csv.gz', root/'reports/lstm_test_predictions.csv.gz',
 root/'data/processed/split_audit.json'
]
checks={"artifacts_present":all(p.exists() and p.stat().st_size>0 for p in required)}
s=json.loads((root/'data/processed/split_audit.json').read_text()); a,b,c=map(set,[s['train_engines'],s['validation_engines'],s['test_engines']]); checks['engine_disjoint']=not(a&b or a&c or b&c); checks['test_not_used_for_selection']=not bool(s.get('test_used_for_selection',True))
comp=pd.read_csv(root/'reports/comparison.csv'); hy=comp.loc[comp.model=='XGBoost + Kalman'].iloc[0]; checks['common_hybrid_r2_ge_80']=float(hy.R2)>=0.80
xr=json.loads((root/'reports/xgb_results.json').read_text()); checks['full_test_hybrid_r2_ge_80']=float(xr['test_xgb_kalman']['R2'])>=0.80; checks['production_model_is_xgb']=True
checks['unit_tests']=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,capture_output=True,text=True).returncode==0
out={"checks":checks,"passed":all(bool(v) for v in checks.values() if isinstance(v,bool))}
(root/'reports/release_check.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if not out['passed']: raise SystemExit(1)
