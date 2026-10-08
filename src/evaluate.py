from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import metric_dict


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    x=pd.read_csv(root/"reports/xgb_test_predictions.csv.gz")
    l=pd.read_csv(root/"reports/lstm_test_predictions.csv.gz")
    common=x.merge(l[["engine_id","timestamp_min","lstm_rul_hours"]],on=["engine_id","timestamp_min"],how="inner",validate="one_to_one")
    y=common["y_true_rul_hours"].to_numpy()
    rows=[]
    for name,c in [("XGBoost","xgb_rul_hours"),("XGBoost + Kalman","xgb_kalman_rul_hours"),("LSTM","lstm_rul_hours")]:
        rows.append({"model":name,"common_test_rows":len(common),**metric_dict(y,common[c].to_numpy())})
    pd.DataFrame(rows).to_csv(root/"reports/comparison.csv",index=False)
    (root/"reports/comparison.json").write_text(json.dumps(rows,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
