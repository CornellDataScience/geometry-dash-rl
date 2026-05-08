"""Factorized NoisyNet linear layer (Fortunato et al., 2017).

Ported from the cv-version. Two flavors:
- NoisyLinear:    fixed in/out dims, materialized at construction.
- LazyNoisyLinear: in_dim inferred from first forward pass.

Use `model.eval()` to disable noise (deterministic greedy policy for eval rollouts).
Call `reset_noise()` between episodes / training updates to resample epsilon.
"""
from __future__ import annotations
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn import UninitializedBuffer
from torch.nn.parameter import UninitializedParameter


def _factorized_noise(size: int, device) -> torch.Tensor:
    x = torch.randn(size, device=device)
    return x.sign().mul_(x.abs().sqrt_())


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std = std

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std / math.sqrt(self.in_features))
        self.bias.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        eps_in = _factorized_noise(self.in_features, self.weight.device)
        eps_out = _factorized_noise(self.out_features, self.weight.device)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight + self.weight_sigma * self.weight_epsilon
            b = self.bias + self.bias_sigma * self.bias_epsilon
            return F.linear(x, w, b)
        return F.linear(x, self.weight, self.bias)


class LazyNoisyLinear(LazyModuleMixin, nn.Module):
    def __init__(self, out_features: int, std: float = 0.5):
        super().__init__()
        self.in_features: int | None = None
        self.out_features = out_features
        self.std = std

        self.weight = UninitializedParameter()
        self.weight_sigma = UninitializedParameter()
        self.register_buffer("weight_epsilon", UninitializedBuffer())
        self.bias = UninitializedParameter()
        self.bias_sigma = UninitializedParameter()
        self.register_buffer("bias_epsilon", UninitializedBuffer())

    def initialize_parameters(self, x: torch.Tensor) -> None:
        self.in_features = x.size(-1)
        if isinstance(self.weight, UninitializedParameter):
            self.weight.materialize((self.out_features, self.in_features))
            self.weight_sigma.materialize((self.out_features, self.in_features))
            self.bias.materialize(self.out_features)
            self.bias_sigma.materialize(self.out_features)
        if isinstance(self.weight_epsilon, UninitializedBuffer):
            self.weight_epsilon.materialize((self.out_features, self.in_features))
            self.bias_epsilon.materialize(self.out_features)
        self._reset_after_init()

    def _reset_after_init(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std / math.sqrt(self.in_features))
        self.bias.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std / math.sqrt(self.out_features))
        eps_in = _factorized_noise(self.in_features, self.weight.device)
        eps_out = _factorized_noise(self.out_features, self.weight.device)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def reset_noise(self) -> None:
        if self.has_uninitialized_params() or self.in_features is None:
            return
        eps_in = _factorized_noise(self.in_features, self.weight.device)
        eps_out = _factorized_noise(self.out_features, self.weight.device)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight + self.weight_sigma * self.weight_epsilon
            b = self.bias + self.bias_sigma * self.bias_epsilon
            return F.linear(x, w, b)
        return F.linear(x, self.weight, self.bias)
