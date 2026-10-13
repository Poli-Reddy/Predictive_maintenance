import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd, numpy as np
manifest=pd.read_csv('data/processed/engine_manifest.csv')
import json
splits=json.load(open('data/processed/engine_splits.json'))
for name in ['train','validation','test']:
    m=manifest[manifest.engine_id.isin(splits[name])]
    print(f'{name} n={len(m)} lifetime mean={m.lifetime_min.mean()/60:.2f}h std={m.lifetime_min.std()/60:.2f} min={m.lifetime_min.min()/60:.1f} max={m.lifetime_min.max()/60:.1f}')
    print(' env',m.environment.value_counts().to_dict())
    print(' fail',m.failure_mode_sim.value_counts().to_dict())
# test vs val RUL distribution (mean/var) from raw
raw_chunks=pd.read_csv('data/raw/arjun_powerpack_synthetic_v3.csv.gz',usecols=['engine_id','rul_hours'],chunksize=300000)
import math
for name in ['validation','test']:
    ids=set(splits[name])
    s=0;s2=0;n=0
    for ch in pd.read_csv('data/raw/arjun_powerpack_synthetic_v3.csv.gz',usecols=['engine_id','rul_hours'],chunksize=300000):
        c=ch[ch.engine_id.isin(ids)]['rul_hours'].to_numpy(dtype=float)
        s+=c.sum(); s2+=(c**2).sum(); n+=len(c)
    mean=s/n; var=s2/n-mean**2
    print(f'{name} RUL mean {mean:.3f} var {var:.3f} std {math.sqrt(var):.3f} n {n} R290-needs-RMSE {math.sqrt(var*0.10):.3f}')
