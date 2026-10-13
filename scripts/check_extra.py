import yaml
from pathlib import Path
cfg_path=Path('config/config.yaml')
import yaml as y
cfg=y.safe_load(open(cfg_path))
print('current n_engines',cfg['dataset']['n_engines'])
# check first-800 stability: build manifests for 800 and 1200 with same seed and compare first 800 lifetimes
import sys
sys.path.insert(0,'src')
from simulate_dataset import build_engine_manifest
m800=cfg.copy(); m800['dataset']=dict(cfg['dataset']); m800['dataset']['n_engines']=800
m1200=cfg.copy(); m1200['dataset']=dict(cfg['dataset']); m1200['dataset']['n_engines']=1200
s800=build_engine_manifest(m800); s1200=build_engine_manifest(m1200)
same=all(a.lifetime_min==b.lifetime_min and a.environment==b.environment and a.failure_mode==b.failure_mode for a,b in zip(s800,s1200[:800]))
print('first800 identical:',same)
print('extra 801-810:',[(s.engine_id,s.lifetime_min,s.environment,s.failure_mode) for s in s1200[800:810]])
