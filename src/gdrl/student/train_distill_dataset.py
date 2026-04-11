from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from gdrl.student.dataset import DistillNPZDataset
from gdrl.student.model import StudentAgent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the vision student from NPZ shards.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stack-size", type=int, default=4)
    ap.add_argument("--out", type=str, default="artifacts/student.pt")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = DistillNPZDataset(args.data, stack_size=args.stack_size)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset too small for stack_size={args.stack_size}")

    counts: dict[int, int] = {}
    for mode in ds.sample_modes.tolist():
        counts[mode] = counts.get(mode, 0) + 1
    weights = [1.0 / counts[int(mode)] for mode in ds.sample_modes]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentAgent(num_modes=max(1, ds.num_modes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    kl = nn.KLDivLoss(reduction="batchmean")
    ce = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total = 0.0
        for frames, teacher_probs, mode in dl:
            frames = frames.to(device)
            teacher_probs = teacher_probs.to(device)
            mode = mode.to(device)

            a_logits, m_logits = model(frames)
            loss = kl(torch.log_softmax(a_logits, dim=-1), teacher_probs) + 0.1 * ce(m_logits, mode)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())

        print(f"epoch={epoch + 1} loss={total / len(dl):.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_modes": max(1, ds.num_modes),
            "stack_size": args.stack_size,
        },
        out_path,
    )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
