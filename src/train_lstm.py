from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from metrics import metric_dict

SENSORS = [
    "engine_rpm",
    "transmission_rpm",
    "engine_coolant_temp_c",
    "transmission_oil_temp_c",
    "engine_oil_pressure_bar",
    "clutch_pressure_bar",
    "solenoid_current_a",
    "vehicle_speed_kph",
]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def prepare_sequences(df: pd.DataFrame, seq_len: int, stride: int, scaler=None, fit_scaler=False, max_sequences=None):
    # Keep the sequence input strictly on sensor telemetry. Time-since-overhaul is a strong age proxy and is
    # intentionally excluded from the model to avoid inflated synthetic benchmark performance.
    context = np.zeros((len(df), 1), dtype=np.float32)
    arrays = df[SENSORS].to_numpy(dtype=np.float32)
    masks = df[SENSORS].isna().to_numpy(dtype=np.float32)
    y = df["rul_hours"].to_numpy(dtype=np.float32)
    engines = df["engine_id"].to_numpy()

    if fit_scaler:
        med = np.nanmedian(arrays, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        fill = np.where(np.isnan(arrays), med, arrays)
        mean = fill.mean(axis=0)
        std = fill.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        scaler = {"median": med.astype(np.float32), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    if scaler is None:
        raise ValueError("Scaler required")
    fill = np.where(np.isnan(arrays), scaler["median"], arrays)
    scaled = (fill - scaler["mean"]) / scaler["std"]
    # Add causal first-difference channels. Difference is within each engine only.
    diff = np.zeros_like(scaled)
    diff[1:] = scaled[1:] - scaled[:-1]
    same = engines[1:] == engines[:-1]
    diff[1:][~same] = 0.0
    if fit_scaler or "context_mean" not in scaler:
        c_mean = float(np.nanmean(context)) if np.isfinite(context).any() else 0.0
        c_std = float(np.nanstd(context)) if np.isfinite(context).any() else 1.0
        c_std = max(c_std,1e-6)
        scaler["context_mean"] = np.float32(c_mean); scaler["context_std"] = np.float32(c_std)
    else:
        c_mean = float(scaler["context_mean"]); c_std = max(float(scaler["context_std"]), 1e-6)
    context_scaled = (np.nan_to_num(context,nan=c_mean)-c_mean)/c_std
    x_all = np.concatenate([scaled, np.clip(diff, -8, 8), masks, context_scaled], axis=1).astype(np.float32)

    starts = []
    for eid, idx in df.groupby("engine_id", sort=False).indices.items():
        idx = np.asarray(idx)
        for end_pos in range(seq_len - 1, len(idx), stride):
            starts.append(int(idx[end_pos]))
    if max_sequences and len(starts) > max_sequences:
        # Deterministic thinning of TRAIN sequences only. Validation/test are never thinned.
        chosen = np.linspace(0, len(starts) - 1, max_sequences, dtype=int)
        starts = [starts[i] for i in chosen]
    return x_all, y, np.asarray(starts, dtype=np.int64), scaler


class WindowDataset(Dataset):
    def __init__(self, x_all, y, ends, seq_len):
        self.x = x_all; self.y = y; self.ends = ends; self.seq_len = seq_len
    def __len__(self): return len(self.ends)
    def __getitem__(self, i):
        end = int(self.ends[i]); start = end - self.seq_len + 1
        return torch.from_numpy(self.x[start:end + 1]), torch.tensor(self.y[end], dtype=torch.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True, dropout=(dropout if layers > 1 else 0.0))
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1, :])).squeeze(-1)


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    total = 0.0; n = 0; preds = []; ys = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total += float(loss.item()) * len(yb); n += len(yb)
        preds.append(pred.detach().cpu().numpy()); ys.append(yb.detach().cpu().numpy())
    return total / max(n, 1), np.concatenate(ys), np.concatenate(preds)


def fit_model(train_df, val_df, cfg, root):
    set_seed(int(cfg["project"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lc = cfg["lstm"]
    seq_len = int(lc["sequence_length"]); stride = int(lc["stride"])
    max_train_sequences = int(lc["max_train_sequences"])
    x_tr, y_tr_raw, e_tr, scaler = prepare_sequences(
        train_df, seq_len, stride, fit_scaler=True, max_sequences=max_train_sequences
    )
    x_va, y_va_raw, e_va, _ = prepare_sequences(val_df, seq_len, stride, scaler=scaler, max_sequences=None)
    y_mean=float(np.mean(y_tr_raw)); y_std=float(np.std(y_tr_raw)); y_std=max(y_std,1e-6)
    y_tr=(y_tr_raw-y_mean)/y_std; y_va=(y_va_raw-y_mean)/y_std
    tr_loader = DataLoader(WindowDataset(x_tr, y_tr, e_tr, seq_len), batch_size=int(lc["batch_size"]), shuffle=True, num_workers=0)
    va_loader = DataLoader(WindowDataset(x_va, y_va, e_va, seq_len), batch_size=int(lc["batch_size"]), shuffle=False, num_workers=0)

    model = LSTMRegressor(x_tr.shape[1], int(lc["hidden_size"]), int(lc["num_layers"]), float(lc["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lc["lr"]), weight_decay=float(lc["weight_decay"]))
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    best_state = None; best_val = float("inf"); patience = int(lc["patience"]); bad = 0; best_epoch = 1
    history=[]
    for epoch in range(1, int(lc["epochs"]) + 1):
        tr_loss, _, _ = run_epoch(model, tr_loader, loss_fn, opt, device)
        va_loss, yva_s, pva_s = run_epoch(model, va_loader, loss_fn, None, device)
        yva=yva_s*y_std+y_mean; pva=pva_s*y_std+y_mean
        m = metric_dict(yva, np.clip(pva, 0, None));         history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "training_windows": len(e_tr),
            "validation_windows": len(e_va),
            "max_train_sequences": max_train_sequences,
            **m,
        })
        print(f"LSTM epoch={epoch} train={tr_loss:.4f} val={va_loss:.4f} R2={m['R2_percent']:.2f}%")
        if va_loss < best_val - 1e-4:
            best_val = va_loss; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is None: raise RuntimeError("No LSTM checkpoint")

    refit_final = bool(cfg["lstm"].get("refit_final", False))
    if refit_final:
        # Optional production refit: once stopping point is selected, refit from scratch on TRAIN+VALIDATION.
        tv = pd.concat([train_df, val_df], ignore_index=True)
        x_tv, y_tv_raw, e_tv, scaler_tv = prepare_sequences(
            tv, seq_len, stride, fit_scaler=True, max_sequences=max_train_sequences
        )
        y_mean_tv=float(np.mean(y_tv_raw)); y_std_tv=max(float(np.std(y_tv_raw)),1e-6)
        y_tv=(y_tv_raw-y_mean_tv)/y_std_tv
        tv_loader = DataLoader(WindowDataset(x_tv, y_tv, e_tv, seq_len), batch_size=int(lc["batch_size"]), shuffle=True, num_workers=0)
        final_model = LSTMRegressor(x_tv.shape[1], int(lc["hidden_size"]), int(lc["num_layers"]), float(lc["dropout"])).to(device)
        final_opt = torch.optim.AdamW(final_model.parameters(), lr=float(lc["lr"]), weight_decay=float(lc["weight_decay"]))
        for _ in range(best_epoch):
            run_epoch(final_model, tv_loader, loss_fn, final_opt, device)
        scaler_tv["target_mean"] = np.float32(y_mean_tv); scaler_tv["target_std"] = np.float32(y_std_tv)
    else:
        # Frozen development model: no post-selection refit, preserving a strict train/validation/test protocol.
        final_model = model
        final_model.load_state_dict(best_state)
        scaler_tv = scaler
        scaler_tv["target_mean"] = np.float32(y_mean); scaler_tv["target_std"] = np.float32(y_std)

    model_dir = root / "models"; model_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": final_model.state_dict(), "input_size": x_tr.shape[1], "hidden": lc["hidden_size"], "layers": lc["num_layers"], "dropout": lc["dropout"], "sequence_length": seq_len, "target_mean": float(scaler_tv["target_mean"]), "target_std": float(scaler_tv["target_std"])}, model_dir / "lstm_rul.pt")
    joblib.dump(scaler_tv, model_dir / "lstm_scaler.joblib")
    for record in history:
        record["selected_epoch"] = best_epoch
    (root / "reports/lstm_training_history.json").write_text(json.dumps(history, indent=2))
    return final_model, scaler_tv, device, seq_len


def predict(model, scaler, device, df, seq_len, root, stride=5):
    x, y, ends, _ = prepare_sequences(df, seq_len, stride, scaler=scaler, max_sequences=None)
    loader = DataLoader(WindowDataset(x, y, ends, seq_len), batch_size=1024, shuffle=False)
    model.eval(); preds=[]; targets=[]; positions=[]
    with torch.no_grad():
        for i,(xb,yb) in enumerate(loader):
            p=model(xb.to(device)).cpu().numpy(); preds.append(p); targets.append(yb.numpy())
    raw_scaled=np.concatenate(preds); pred=np.clip(raw_scaled*float(scaler["target_std"])+float(scaler["target_mean"]),0,None); yt=np.concatenate(targets)
    return df.iloc[ends][["engine_id","timestamp_min"]].reset_index(drop=True), yt, pred
