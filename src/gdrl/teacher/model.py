from __future__ import annotations
import torch
import torch.nn as nn

OBS_DIM = 608
STACK_SIZE = 4
INPUT_DIM = STACK_SIZE * OBS_DIM  # 2432
NUM_ACTIONS = 2


class TeacherPolicy(nn.Module):
    """MLP policy for the privileged teacher.

    Input:  (batch, 4 * 608) float32 — 4 stacked obs frames, oldest → newest
    Output: (batch, 2) logits — [no-jump, jump]
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted action indices (argmax over logits)."""
        with torch.no_grad():
            return self.forward(x).argmax(dim=-1)
