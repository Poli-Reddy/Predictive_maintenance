from pathlib import Path
import sys, pandas as pd, yaml, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from splits import create_production_splits
from train_lstm import fit_model,predict
root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'config/config.yaml').read_text()); raw=pd.read_csv(root/cfg['dataset']['output_path']); man=pd.read_csv(root/'data/processed/engine_manifest.csv'); s=create_production_splits(man,cfg)
tr=raw[raw.engine_id.isin(s['train'])].copy(); va=raw[raw.engine_id.isin(s['validation'])].copy(); te=raw[raw.engine_id.isin(s['test'])].copy()
m,sc,dev,seq=fit_model(tr,va,cfg,root)
meta,y,p=predict(m,sc,dev,te,seq,root,stride=5)
out=meta.copy(); out['y_true_rul_hours']=y; out['lstm_rul_hours']=p; out.to_csv(root/'reports/lstm_test_predictions.csv.gz',index=False,compression='gzip')
print(json.dumps({'test_rows':len(out),'seq_len':seq},indent=2))
