from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class DistillNPZDataset(Dataset):
    def __init__(self, npz_path: str, stack_size: int = 4):
        data = np.load(npz_path)
        self.frames = data['frames']
        self.teacher_probs = data['teacher_probs']
        self.modes = data['modes']
        self.stack_size = stack_size

    def __len__(self):
        return max(0, len(self.frames) - self.stack_size)

    def __getitem__(self, idx):
        stacked = self.frames[idx:idx+self.stack_size].astype(np.float32) / 255.0
        probs = self.teacher_probs[idx + self.stack_size - 1].astype(np.float32)
        mode = int(self.modes[idx + self.stack_size - 1])
        return (
            torch.from_numpy(stacked),
            torch.from_numpy(probs),
            torch.tensor(mode, dtype=torch.long),
        )
