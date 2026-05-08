"""Mode-gated Rainbow DQN for CV input.

- Conv backbone ported from cv-version DeeperDQNModelv2.
- 4 dueling heads, one per gamemode (cube=0, ship=1, ball=2, ufo=3).
- Hard router: gather expert output by `mode_id` index (provided by env from mod state).
- NoisyNet replaces ε-greedy.
- Optional aux head: predicts a few privileged-state features off the conv backbone
  (free supervision from the mod's IPC). MSE loss, weighted small.

Input: (B, T, C, H, W) frames uint8 or float, mode_id (B,) long.
Output: (q_values (B, A), aux (B, AUX_DIM) or None)
"""
from __future__ import annotations

import torch
from torch import nn

from gdrl.model.noisy_linear import NoisyLinear, LazyNoisyLinear

NUM_MODES = 4
NUM_ACTIONS = 2
AUX_DIM = 6  # y, vy, dx, on_ground, nearestX, nearestY


class _Backbone(nn.Module):
    """4-conv stack from DeeperDQNModelv2. Input (B, T*C, H, W)."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DuelingHead(nn.Module):
    """V/A streams, NoisyLinear throughout. Q = V + A - mean(A)."""
    def __init__(self, hidden: int = 512, num_actions: int = NUM_ACTIONS, std: float = 0.5):
        super().__init__()
        self.v_fc = LazyNoisyLinear(hidden, std=std)
        self.v_out = NoisyLinear(hidden, 1, std=std)
        self.a_fc = LazyNoisyLinear(hidden, std=std)
        self.a_out = NoisyLinear(hidden, num_actions, std=std)
        self.act = nn.SiLU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        v = self.v_out(self.act(self.v_fc(feat)))            # (B, 1)
        a = self.a_out(self.act(self.a_fc(feat)))            # (B, A)
        return v + a - a.mean(dim=-1, keepdim=True)

    def reset_noise(self) -> None:
        for m in (self.v_fc, self.v_out, self.a_fc, self.a_out):
            m.reset_noise()


class ModeGatedRainbow(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        stack: int = 4,
        num_modes: int = NUM_MODES,
        num_actions: int = NUM_ACTIONS,
        hidden: int = 512,
        noisy_std: float = 0.5,
        aux: bool = True,
    ):
        super().__init__()
        self.stack = stack
        self.in_channels = in_channels
        self.num_modes = num_modes
        self.num_actions = num_actions
        self.aux_enabled = aux

        self.backbone = _Backbone(in_channels * stack)
        self.experts = nn.ModuleList([
            _DuelingHead(hidden=hidden, num_actions=num_actions, std=noisy_std)
            for _ in range(num_modes)
        ])
        if aux:
            self.aux_head = nn.Sequential(
                nn.LazyLinear(128), nn.SiLU(), nn.Linear(128, AUX_DIM)
            )
        else:
            self.aux_head = None

        # warm-init: run a dummy forward so LazyLinear / LazyNoisyLinear params materialize
        # before optimizer.add_param_group. Run in eval mode so noise isn't applied.
        with torch.no_grad():
            self.eval()
            dummy = torch.zeros(1, stack, in_channels, 128, 128)
            dummy_mode = torch.zeros(1, dtype=torch.long)
            _ = self.forward(dummy, dummy_mode, return_aux=True)
            self.train()

    def _features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) uint8 or float → backbone features (B, F)."""
        if frames.dtype == torch.uint8:
            x = frames.float().div_(255.0)
        else:
            x = frames
        B, T, C, H, W = x.shape
        x = x.view(B, T * C, H, W)
        return self.backbone(x)

    def forward(
        self,
        frames: torch.Tensor,
        mode_id: torch.Tensor,
        return_aux: bool = False,
    ):
        feat = self._features(frames)                                # (B, F)
        # stack expert outputs and gather by mode_id (hard routing)
        expert_outs = torch.stack([e(feat) for e in self.experts], dim=1)  # (B, M, A)
        idx = mode_id.view(-1, 1, 1).expand(-1, 1, expert_outs.size(-1))
        q = expert_outs.gather(1, idx).squeeze(1)                    # (B, A)
        if return_aux and self.aux_head is not None:
            return q, self.aux_head(feat)
        return q

    def reset_noise(self) -> None:
        for e in self.experts:
            e.reset_noise()
