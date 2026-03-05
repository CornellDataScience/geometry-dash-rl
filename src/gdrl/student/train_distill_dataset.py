from __future__ import annotations
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from gdrl.student.model import StudentAgent
from gdrl.student.dataset import DistillNPZDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', type=str, default='artifacts/student_mock.pt')
    args = ap.parse_args()

    ds = DistillNPZDataset(args.data)
    if len(ds) == 0:
        raise RuntimeError('Dataset too small for stack_size=4')

    # weighted sampling by mode
    modes = [int(ds.modes[i + ds.stack_size - 1]) for i in range(len(ds))]
    counts = {}
    for m in modes:
        counts[m] = counts.get(m, 0) + 1
    weights = [1.0 / counts[m] for m in modes]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StudentAgent().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    kl = nn.KLDivLoss(reduction='batchmean')
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

        print(f'epoch={epoch+1} loss={total/len(dl):.4f}')

    torch.save(model.state_dict(), args.out)
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
