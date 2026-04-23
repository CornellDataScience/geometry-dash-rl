"""Behavioral cloning training loop.

Usage:
    python -m gdrl.train.imitation --data-dir artifacts/recordings/ --epochs 100 --out artifacts/bc_model.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gdrl.data.obs_dataset import HumanPlayDataset, train_val_split, find_shards, ShardIndex
from gdrl.model.obs_preprocess import (
    ObsPreprocessor,
    ObsNormalizer,
    compute_normalizer,
    PROCESSED_FRAME_DIM,
)
from gdrl.model.mlp_agent import GDPolicyMLP


def compute_pos_weight(shard_dir: str | Path) -> float:
    sessions = find_shards(shard_dir)
    index = ShardIndex(sessions)
    total_pos = 0
    total = 0
    for i in range(len(index)):
        _, action, _ = index.get(i)
        total_pos += action
        total += 1
    total_neg = total - total_pos
    if total_pos == 0:
        return 1.0
    return total_neg / total_pos


def evaluate(model, loader, criterion, device, event_tolerance: int = 15) -> dict:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    n = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logit, _ = model(x)
            logit = logit.squeeze(-1)
            loss = criterion(logit, y)
            total_loss += loss.item() * len(y)
            n += len(y)
            pred = (logit > 0.0).float()
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    # jump-event metrics with temporal tolerance
    from gdrl.eval.offline_metrics import extract_jump_events, match_events
    preds = np.concatenate(all_preds).astype(int)
    labels = np.concatenate(all_labels).astype(int)
    # use sequential frame indices as pseudo-episode (val loader is not shuffled)
    episode_ids = np.zeros(len(preds), dtype=int)
    human_events = extract_jump_events(labels, episode_ids)
    model_events = extract_jump_events(preds, episode_ids)
    event_metrics = match_events(human_events, model_events, tolerance=event_tolerance)

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "event_precision": event_metrics["event_precision"],
        "event_recall": event_metrics["event_recall"],
        "event_f1": event_metrics["event_f1"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train BC model on human gameplay.")
    ap.add_argument("--data-dir", required=True, help="Root dir containing recording sessions.")
    ap.add_argument("--out", default="artifacts/bc_model.pt", help="Output checkpoint path.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    ap.add_argument("--no-normalize", action="store_true", help="Skip observation normalization.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # compute normalizer
    if args.no_normalize:
        normalizer = ObsNormalizer.identity()
        print("normalization disabled", flush=True)
    else:
        print("computing normalization stats...", flush=True)
        normalizer = compute_normalizer(data_dir, stack_size=args.stack)
        norm_path = out_path.with_suffix(".norm.npz")
        normalizer.save(norm_path)
        print(f"saved normalizer to {norm_path}", flush=True)

    preprocessor = ObsPreprocessor(normalizer=normalizer)

    # compute class weight
    pos_weight_val = compute_pos_weight(data_dir)
    print(f"pos_weight={pos_weight_val:.1f} (1 jump per {pos_weight_val:.0f} frames)", flush=True)

    # build datasets
    train_ds, val_ds = train_val_split(
        data_dir,
        val_fraction=args.val_fraction,
        stack_size=args.stack,
        preprocessor=preprocessor,
    )
    print(f"train={len(train_ds)} val={len(val_ds)} frames", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # build model
    input_dim = PROCESSED_FRAME_DIM * args.stack
    model = GDPolicyMLP(input_dim=input_dim).to(device)
    print(f"model params: {model.param_count():,}", flush=True)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logit, _ = model(x)
            logit = logit.squeeze(-1)
            loss = criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_n += len(y)

        val_metrics = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(
            f"epoch {epoch:3d}/{args.epochs} "
            f"train_loss={train_loss / max(train_n, 1):.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.3f} "
            f"f1={val_metrics['f1']:.3f} "
            f"evt_f1={val_metrics['event_f1']:.3f} "
            f"evt_p={val_metrics['event_precision']:.3f} "
            f"evt_r={val_metrics['event_recall']:.3f} "
            f"({dt:.1f}s)",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "stack_size": args.stack,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
            }, out_path)
            print(f"  saved best model → {out_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch} (patience={args.patience})", flush=True)
                break

    print(f"done. best val_loss={best_val_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
