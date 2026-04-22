"""Behavioral cloning warmup for the teacher policy.

Trains TeacherPolicy on recorded human gameplay shards and saves a checkpoint
to be used as the starting point for DAgger iteration 0.

Usage:
    python -m gdrl.teacher.bc_train --data artifacts/recordings/ --out artifacts/bc_warmup.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gdrl.data.obs_dataset import HumanPlayDataset, train_val_split
from gdrl.teacher.model import TeacherPolicy


def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, val_ds = train_val_split(
        args.data,
        val_fraction=args.val_fraction,
        seed=args.seed,
        stack_size=4,
    )
    print(f"train={len(train_ds)} frames  val={len(val_ds)} frames")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TeacherPolicy(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for obs, action in train_dl:
            obs = obs.to(device)
            action = action.long().to(device)
            logits = model(obs)
            loss = loss_fn(logits, action)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss) * len(obs)
            train_correct += int((logits.argmax(dim=-1) == action).sum())
            train_total += len(obs)

        # --- val ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for obs, action in val_dl:
                obs = obs.to(device)
                action = action.long().to(device)
                logits = model(obs)
                loss = loss_fn(logits, action)
                val_loss += float(loss) * len(obs)
                val_correct += int((logits.argmax(dim=-1) == action).sum())
                val_total += len(obs)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(
            f"epoch={epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
            print(f"  saved best checkpoint -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="BC warmup training for teacher policy.")
    ap.add_argument("--data", required=True, help="Path to shard root (artifacts/recordings/)")
    ap.add_argument("--out", default="artifacts/bc_warmup.pt", help="Output checkpoint path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
