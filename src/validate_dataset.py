from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from features import CORE_SENSORS
from splits import create_production_splits


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args(); root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/args.config).read_text()); df=pd.read_csv(root/cfg["dataset"]["output_path"]); manifest=pd.read_csv(root/"data/processed/engine_manifest.csv")
    splits=create_production_splits(manifest,cfg); train=set(splits['train']); val=set(splits['validation']); test=set(splits['test'])
    checks={
        "rows_positive":len(df)>0,
        "engines_expected":df.engine_id.nunique()==cfg['dataset']['n_engines'],
        "sensor_columns_present":all(c in df.columns for c in CORE_SENSORS),
        "sensor_finite_or_missing":bool(np.isfinite(df[CORE_SENSORS].fillna(0).to_numpy()).all()),
        "rul_nonnegative":bool((df.rul_hours>=0).all()),
        "rul_monotone_per_engine":bool(df.groupby('engine_id').rul_hours.apply(lambda s:s.is_monotonic_decreasing).all()),
        "time_monotone_per_engine":bool(df.groupby('engine_id').timestamp_min.apply(lambda s:s.is_monotonic_increasing).all()),
        "rul_zero_at_failure":bool(np.allclose(df.groupby('engine_id').rul_hours.last().to_numpy(),0.0,atol=1e-6)),
        "engine_split_disjoint":not(train&val or train&test or val&test),
        "no_duplicate_engine_timestamp":not df.duplicated(['engine_id','timestamp_min']).any(),
    }
    out={"rows":len(df),"engines":df.engine_id.nunique(),"missing_rate":{c:float(df[c].isna().mean()) for c in CORE_SENSORS},"environment_counts":df.environment.value_counts().to_dict(),"failure_mode_counts":df.failure_mode_sim.value_counts().to_dict(),"checks":checks,"failed_checks":[k for k,v in checks.items() if not v]}
    (root/"reports/dataset_validation.json").write_text(json.dumps(out,indent=2,default=int)); print(json.dumps(out,indent=2));
    if out['failed_checks']: raise SystemExit(1)

if __name__=='__main__': main()
