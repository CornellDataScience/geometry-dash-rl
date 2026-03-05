import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        r = x
        x = torch.relu(self.c1(torch.relu(x)))
        x = self.c2(x)
        return x + r


class StudentAgent(nn.Module):
    def __init__(self, num_modes: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.MaxPool2d(3, 2, 1), ResBlock(16), ResBlock(16),
            nn.Conv2d(16, 32, 3, padding=1), nn.MaxPool2d(3, 2, 1), ResBlock(32), ResBlock(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.MaxPool2d(3, 2, 1), ResBlock(32), ResBlock(32),
        )
        self.fc = nn.Linear(32 * 11 * 11, 256)
        self.action_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
        self.mode_head = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, num_modes))

    def forward(self, frames):
        x = self.backbone(frames)
        x = torch.relu(self.fc(torch.flatten(x, 1)))
        return self.action_head(x), self.mode_head(x)
