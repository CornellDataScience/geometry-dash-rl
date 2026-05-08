"""Tests for GDPolicyMLP model."""
from __future__ import annotations

import torch
import pytest

from gdrl.model.mlp_agent import GDPolicyMLP
from gdrl.model.obs_preprocess import PROCESSED_FRAME_DIM


def test_default_input_dim():
    model = GDPolicyMLP()
    assert model.input_dim == PROCESSED_FRAME_DIM * 4  # 157 * 4 = 628


def test_forward_output_shapes():
    model = GDPolicyMLP()
    x = torch.randn(8, model.input_dim)  # batch=8
    action_logit, value = model(x)
    assert action_logit.shape == (8, 1)
    assert value.shape == (8, 1)


def test_forward_single_sample():
    model = GDPolicyMLP()
    x = torch.randn(1, model.input_dim)
    action_logit, value = model(x)
    assert action_logit.shape == (1, 1)
    assert value.shape == (1, 1)


def test_act_returns_0_or_1():
    model = GDPolicyMLP()
    x = torch.randn(1, model.input_dim)
    action = model.act(x)
    assert action in (0, 1)


def test_act_unbatched():
    model = GDPolicyMLP()
    x = torch.randn(model.input_dim)  # no batch dim
    action = model.act(x)
    assert action in (0, 1)


def test_param_count_reasonable():
    model = GDPolicyMLP()
    count = model.param_count()
    # backbone: Linear(1468,512) + 2x Linear(512,512) + Linear(512,128) + heads + embedding(42,8)
    # ~1.1M-1.5M
    assert 800_000 < count < 1_500_000, f"param count {count} outside expected range"


def test_custom_architecture():
    model = GDPolicyMLP(input_dim=PROCESSED_FRAME_DIM * 2, hidden=64, n_layers=2, neck=32, stack_size=2)
    x = torch.randn(4, PROCESSED_FRAME_DIM * 2)
    action_logit, value = model(x)
    assert action_logit.shape == (4, 1)
    assert value.shape == (4, 1)


def test_gradient_flows():
    model = GDPolicyMLP()
    x = torch.randn(4, model.input_dim)
    action_logit, value = model(x)
    loss = action_logit.sum() + value.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"


def test_embedding_gradient():
    """Verify gradients flow through the type embedding."""
    model = GDPolicyMLP()
    x = torch.randn(4, model.input_dim)
    # put a known type_id in the first object of first frame
    from gdrl.model.obs_preprocess import PROCESSED_PLAYER_DIM, PROCESSED_OBJ_DIM
    x[:, PROCESSED_PLAYER_DIM + 2] = 9.0  # yellowOrb type_id
    action_logit, value = model(x)
    loss = action_logit.sum() + value.sum()
    loss.backward()
    assert model.type_embedding.weight.grad is not None
    assert not torch.all(model.type_embedding.weight.grad == 0)
