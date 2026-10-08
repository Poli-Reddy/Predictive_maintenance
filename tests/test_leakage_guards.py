from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import build_features, model_feature_columns, BANNED_FEATURES

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/arjun_powerpack_synthetic_v3.csv.gz'


def test_partitions_are_engine_disjoint():
    s=json.loads((ROOT/'data/processed/split_audit.json').read_text())
    a,b,c=map(set,[s['train_engines'],s['validation_engines'],s['test_engines']])
    assert not (a&b or a&c or b&c)


def test_banned_columns_are_not_features():
    df=pd.read_csv(RAW,nrows=2500)
    f=build_features(df[df.engine_id==int(df.engine_id.iloc[0])])
    assert not (set(model_feature_columns(f)) & BANNED_FEATURES)


def test_feature_builder_is_causal():
    df=pd.read_csv(RAW)
    eid=int(df.engine_id.iloc[0]); one=df[df.engine_id==eid].reset_index(drop=True)
    if len(one)<20: return
    altered=one.copy(); altered.loc[len(altered)-1,'engine_rpm']+=500.0; altered.loc[len(altered)-1,'engine_coolant_temp_c']+=20.0
    fa=build_features(one); fb=build_features(altered)
    cols=model_feature_columns(fa)
    assert np.allclose(fa.loc[:len(one)-2,cols].fillna(0).to_numpy(), fb.loc[:len(one)-2,cols].fillna(0).to_numpy(), atol=1e-6)
