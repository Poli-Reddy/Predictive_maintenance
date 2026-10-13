"""Append 400 extra train-only engines (801-1200). Original 800 rows untouched. Backup first."""
import sys, shutil
from pathlib import Path
sys.path.insert(0,'src')
import pandas as pd, yaml
from simulate_dataset import build_engine_manifest, simulate_engine, SENSOR_COLUMNS
ROOT=Path('.')
# backup
for f in ['data/raw/arjun_powerpack_synthetic_v3.csv.gz','data/processed/engine_manifest.csv','data/processed/dataset_metadata.json']:
    p=ROOT/f
    if p.exists() and not (ROOT/(f+'.bak800')).exists():
        shutil.copy(p, ROOT/(f+'.bak800'))
        print('backed up',f,flush=True)
cfg=yaml.safe_load(open(ROOT/'config/config.yaml'))
cfg1200={**cfg,'dataset':{**cfg['dataset'],'n_engines':1200}}
scen=build_engine_manifest(cfg1200)[800:]
print(f'generating {len(scen)} engines...',flush=True)
frames=[]; mrows=[]
for i,sc in enumerate(scen):
    frames.append(simulate_engine(sc))
    mrows.append({'engine_id':sc.engine_id,'production_batch':sc.production_batch,'environment':sc.environment,'failure_mode_sim':sc.failure_mode,'lifetime_min':sc.lifetime_min,'seed':sc.seed,'severity_latent':sc.severity})
    if (i+1)%50==0: print(f'  {i+1}/{len(scen)}',flush=True)
new=pd.concat(frames,ignore_index=True)
for c in [*SENSOR_COLUMNS,'ambient_context_c','altitude_context_m','dust_context','sensor_quality','rul_hours','timestamp_min']:
    new[c]=new[c].astype('float32')
new['engine_id']=new['engine_id'].astype('int32')
new['production_batch']=new['production_batch'].astype('int16')
old=pd.read_csv(ROOT/cfg['dataset']['output_path'])
full=pd.concat([old,new],ignore_index=True).sort_values(['engine_id','timestamp_min']).reset_index(drop=True)
full.to_csv(ROOT/cfg['dataset']['output_path'],index=False,compression='gzip',float_format='%.6g')
man=pd.read_csv(ROOT/'data/processed/engine_manifest.csv')
man=pd.concat([man,pd.DataFrame(mrows)],ignore_index=True).sort_values('engine_id').reset_index(drop=True)
man.to_csv(ROOT/'data/processed/engine_manifest.csv',index=False)
print(f'done rows {len(full)} engines {full.engine_id.nunique()}',flush=True)
