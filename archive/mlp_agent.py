"""GDPolicyMLP: 512×512×512 MLP with learned object type embeddings.

Input: stacked processed observations (default 628 = 157 × 4 frames).
The model internally expands type_id floats into 8-dim embeddings,
producing an effective input of 1468 = (7 + 30×12) × 4.
Output: (action_logit, value) tuple.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gdrl.model.obs_preprocess import (
    PROCESSED_FRAME_DIM,
    PROCESSED_PLAYER_DIM,
    PROCESSED_OBJ_DIM,
    N_SELECTED_OBJECTS,
    N_OBJ_TYPES,
    EMBED_DIM,
)

# Effective dim per object after embedding: relX + relY + embed(8) + scaleX + scaleY = 12
_EFFECTIVE_OBJ_DIM = EMBED_DIM + 4  # 12
# Effective dim per frame after embedding expansion
_EFFECTIVE_FRAME_DIM = PROCESSED_PLAYER_DIM + N_SELECTED_OBJECTS * _EFFECTIVE_OBJ_DIM  # 367


class GDPolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = PROCESSED_FRAME_DIM * 4,  # 157 * 4 = 628 (preprocessor output)
        hidden: int = 512,
        n_layers: int = 3,
        neck: int = 128,
        stack_size: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.stack_size = stack_size

        self.type_embedding = nn.Embedding(N_OBJ_TYPES, EMBED_DIM)

        effective_input = _EFFECTIVE_FRAME_DIM * stack_size  # 1468

        layers = []
        in_dim = effective_input
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

    def _expand_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Replace type_id floats with learned embeddings.

        Input: (batch, stack_size * 157)
        Output: (batch, stack_size * 367)
        """
        batch = x.shape[0]
        K = self.stack_size
        # reshape to (batch, stack, 157)
        frames = x.view(batch, K, PROCESSED_FRAME_DIM)

        expanded_frames = []
        for k in range(K):
            frame = frames[:, k, :]  # (batch, 157)
            player = frame[:, :PROCESSED_PLAYER_DIM]  # (batch, 7)

            obj_parts = []
            for j in range(N_SELECTED_OBJECTS):
                base = PROCESSED_PLAYER_DIM + j * PROCESSED_OBJ_DIM
                relX = frame[:, base:base + 1]       # (batch, 1)
                relY = frame[:, base + 1:base + 2]   # (batch, 1)
                type_id = frame[:, base + 2].long()   # (batch,)
                type_id = type_id.clamp(0, N_OBJ_TYPES - 1)
                scaleX = frame[:, base + 3:base + 4]  # (batch, 1)
                scaleY = frame[:, base + 4:base + 5]  # (batch, 1)

                emb = self.type_embedding(type_id)    # (batch, 8)
                obj_parts.append(torch.cat([relX, relY, emb, scaleX, scaleY], dim=1))  # (batch, 12)

            expanded = torch.cat([player] + obj_parts, dim=1)  # (batch, 367)
            expanded_frames.append(expanded)

        return torch.cat(expanded_frames, dim=1)  # (batch, 1468)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logit, value), each shape (batch, 1)."""
        expanded = self._expand_embeddings(x)
        features = self.backbone(expanded)
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
