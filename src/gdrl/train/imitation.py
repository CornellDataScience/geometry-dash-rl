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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gdrl.data.obs_dataset import (
    HumanPlayDataset,
    train_val_split,
    train_val_split_by_level,
    find_shards,
    ShardIndex,
)
from gdrl.model.obs_preprocess import (
    ObsPreprocessor,
    ObsNormalizer,
    compute_normalizer,
    PROCESSED_FRAME_DIM,
)
from gdrl.model.mlp_agent import GDPolicyMLP


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss from logits.

    alpha weights the positive class; gamma down-weights easy examples.
    alpha=-1 disables alpha weighting (equivalent to standard focal loss).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def compute_pos_weight(shard_dir: str | Path) -> float:
    sessions = find_shards(shard_dir)
    index = ShardIndex(sessions)
    total_pos = 0
    total = 0
    for i in range(len(index)):
        total_pos += index.get_label(i)
        total += 1
    total_neg = total - total_pos
    if total_pos == 0:
        return 1.0
    return total_neg / total_pos


def evaluate(model, loader, criterion, device, event_tolerance: int = 15, temporally_ordered: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    n = 0
    all_preds = []
    all_labels = []
    all_logits = []
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
            all_logits.append(logit.cpu().numpy())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    # jump-event metrics with temporal tolerance
    # only meaningful when val frames are in temporal order (e.g. held-out level)
    from gdrl.eval.offline_metrics import extract_jump_events, match_events
    preds = np.concatenate(all_preds).astype(int)
    labels = np.concatenate(all_labels).astype(int)
    logits = np.concatenate(all_logits)
    if temporally_ordered:
        ds = loader.dataset
        episode_ids = ds.episode_ids[ds.indices].astype(int)
    else:
        # shuffled val: events aren't real, just placeholder
        episode_ids = np.zeros(len(preds), dtype=int)
    human_events = extract_jump_events(labels, episode_ids)
    model_events = extract_jump_events(preds, episode_ids)
    event_metrics = match_events(human_events, model_events, tolerance=event_tolerance)

    best_threshold = 0.0
    best_threshold_f1 = f1
    if len(logits) > 0 and labels.sum() > 0:
        lo, hi = float(np.percentile(logits, 0.5)), float(np.percentile(logits, 99.5))
        for threshold in np.linspace(lo, hi, 200):
            thresh_preds = (logits > threshold).astype(int)
            thresh_tp = int(((thresh_preds == 1) & (labels == 1)).sum())
            thresh_fp = int(((thresh_preds == 1) & (labels == 0)).sum())
            thresh_fn = int(((thresh_preds == 0) & (labels == 1)).sum())
            thresh_p = thresh_tp / (thresh_tp + thresh_fp) if (thresh_tp + thresh_fp) > 0 else 0.0
            thresh_r = thresh_tp / (thresh_tp + thresh_fn) if (thresh_tp + thresh_fn) > 0 else 0.0
            thresh_f1 = 2 * thresh_p * thresh_r / (thresh_p + thresh_r) if (thresh_p + thresh_r) > 0 else 0.0
            if thresh_f1 > best_threshold_f1:
                best_threshold_f1 = thresh_f1
                best_threshold = float(threshold)

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
        "best_threshold": best_threshold,
        "best_threshold_f1": best_threshold_f1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train BC model on human gameplay.")
    ap.add_argument("--data-dir", required=True, help="Root dir containing recording sessions.")
    ap.add_argument("--out", default="artifacts/bc_model.pt", help="Output checkpoint path.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="Random frame fraction for val (ignored if --val-level set).")
    ap.add_argument("--val-level", default=None,
                    help="Hold out this level dir for val (e.g. 'stereo_madness'). "
                         "Gives meaningful evt_f1 since val is in temporal order.")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    ap.add_argument("--no-normalize", action="store_true", help="Skip observation normalization.")
    ap.add_argument("--pretrained", default=None,
                    help="Path to a BC checkpoint to fine-tune from. Reuses its normalizer "
                         "(if present at <pretrained>.norm.npz).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Focal loss gamma (focusing parameter). 0 = standard BCE.")
    ap.add_argument("--focal-alpha", type=float, default=0.75,
                    help="Focal loss alpha (positive class weight). -1 to disable.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # compute / load normalizer
    if args.no_normalize:
        normalizer = ObsNormalizer.identity()
        print("normalization disabled", flush=True)
    elif args.pretrained:
        # reuse the pretrained model's normalizer to keep input distribution
        # identical to what its weights expect
        pre_norm = Path(args.pretrained).with_suffix(".norm.npz")
        if pre_norm.exists():
            normalizer = ObsNormalizer.load(pre_norm)
            print(f"loaded pretrained normalizer from {pre_norm}", flush=True)
        else:
            print(f"WARNING: no normalizer at {pre_norm}, computing fresh — may break pretrained weights", flush=True)
            normalizer = compute_normalizer(data_dir, stack_size=args.stack)
        norm_path = out_path.with_suffix(".norm.npz")
        normalizer.save(norm_path)
        print(f"saved normalizer to {norm_path}", flush=True)
    else:
        print("computing normalization stats...", flush=True)
        normalizer = compute_normalizer(data_dir, stack_size=args.stack)
        norm_path = out_path.with_suffix(".norm.npz")
        normalizer.save(norm_path)
        print(f"saved normalizer to {norm_path}", flush=True)

    preprocessor = ObsPreprocessor(normalizer=normalizer)

    # compute class imbalance ratio (informational; used for alpha guidance)
    pos_weight_val = compute_pos_weight(data_dir)
    print(f"class ratio={pos_weight_val:.1f} (1 grounded jump press per {pos_weight_val:.0f} frames)", flush=True)
    print("BC target: action=1 only when recorded input is down and obs[4] on_ground is true", flush=True)
    print(f"focal loss: gamma={args.focal_gamma}  alpha={args.focal_alpha}", flush=True)

    # build datasets
    if args.val_level:
        train_ds, val_ds = train_val_split_by_level(
            data_dir,
            val_level=args.val_level,
            stack_size=args.stack,
            preprocessor=preprocessor,
        )
        print(f"val_level={args.val_level} (held out)", flush=True)
        temporally_ordered = True
    else:
        train_ds, val_ds = train_val_split(
            data_dir,
            val_fraction=args.val_fraction,
            stack_size=args.stack,
            preprocessor=preprocessor,
        )
        temporally_ordered = False
    print(f"train={len(train_ds)} val={len(val_ds)} frames", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # build model
    input_dim = PROCESSED_FRAME_DIM * args.stack
    model = GDPolicyMLP(input_dim=input_dim).to(device)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded pretrained weights from {args.pretrained} "
              f"(epoch {ckpt.get('epoch', '?')})", flush=True)
    print(f"model params: {model.param_count():,}", flush=True)

    criterion = lambda logits, targets: sigmoid_focal_loss(
        logits, targets, gamma=args.focal_gamma, alpha=args.focal_alpha
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score = -1.0
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

        val_metrics = evaluate(model, val_loader, criterion, device, temporally_ordered=temporally_ordered)
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
            f"th={val_metrics['best_threshold']:+.3f} "
            f"({dt:.1f}s)",
            flush=True,
        )

        score_key = "event_f1"
        score = float(val_metrics[score_key])
        loss = float(val_metrics["loss"])
        improved = score > best_score or (score == best_score and loss < best_val_loss)

        if improved:
            best_score = score
            best_val_loss = loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "stack_size": args.stack,
                "target_semantics": "grounded_jump_press",
                "epoch": epoch,
                "selection_metric": score_key,
                "selection_score": best_score,
                "action_threshold": float(val_metrics["best_threshold"]),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
            }, out_path)
            print(f"  saved best model by {score_key}={best_score:.4f} → {out_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch} (patience={args.patience})", flush=True)
                break

    print(f"done. best {score_key}={best_score:.4f} val_loss={best_val_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
