"""GDPolicyMLP: 512×512×512 MLP with action + value heads.

Input: stacked processed observations (default 1108 = 277 × 4 frames).
Output: (action_logit, value) tuple.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gdrl.model.obs_preprocess import PROCESSED_FRAME_DIM


class GDPolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = PROCESSED_FRAME_DIM * 4,  # 277 * 4 = 1108
        hidden: int = 512,
        n_layers: int = 3,
        neck: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            ])
            in_dim = hidden

        layers.extend([
            nn.Linear(hidden, neck),
            nn.ReLU(),
        ])
        self.backbone = nn.Sequential(*layers)

        self.action_head = nn.Linear(neck, 1)
        self.value_head = nn.Linear(neck, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logit, value), each shape (batch, 1)."""
        features = self.backbone(x)
        return self.action_head(features), self.value_head(features)

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> int:
        """Greedy action for inference. x shape: (1, input_dim) or (input_dim,)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logit, _ = self.forward(x)
        return int(logit.squeeze().item() > 0.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
