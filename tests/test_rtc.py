"""Tests for Training-Time RTC (per-token timestep, prefix conditioning, masked loss)."""

import torch
import pytest
from gemma4vla.model.config import Gemma4ActionExpertConfig
from gemma4vla.model.action_expert import (
    sinusoidal_embedding, AdaRMSNorm, create_action_expert,
)
from gemma4vla.model.flow_matching import FlowMatchingLoss, FlowMatchingSampler


@pytest.fixture
def config():
    return Gemma4ActionExpertConfig(
        variant="flow_shared",
        action_dim=32,
        action_horizon=15,
        expert_depth=4,  # small for testing
        expert_width=64,
        expert_heads=4,
        expert_kv_heads=2,
        expert_head_dim=16,
        expert_mlp_dim=128,
        time_embed_dim=64,
    )


def test_sinusoidal_embedding_global():
    """Standard (B,) timestep → (B, dim)."""
    t = torch.rand(4)
    emb = sinusoidal_embedding(t, 64)
    assert emb.shape == (4, 64)


def test_sinusoidal_embedding_per_token():
    """Per-token (B, T) timestep → (B, T, dim) for RTC."""
    t = torch.rand(4, 15)
    emb = sinusoidal_embedding(t, 64)
    assert emb.shape == (4, 15, 64)


def test_adarmsnorm_global_cond():
    """AdaRMSNorm with global conditioning (B, D)."""
    norm = AdaRMSNorm(64, 64)
    x = torch.randn(2, 15, 64)
    cond = torch.randn(2, 64)
    out, gate = norm(x, cond)
    assert out.shape == (2, 15, 64)
    assert gate.shape == (2, 1, 64)  # broadcast over T


def test_adarmsnorm_per_token_cond():
    """AdaRMSNorm with per-token conditioning (B, T, D) for RTC."""
    norm = AdaRMSNorm(64, 64)
    x = torch.randn(2, 15, 64)
    cond = torch.randn(2, 15, 64)  # per-token
    out, gate = norm(x, cond)
    assert out.shape == (2, 15, 64)
    assert gate.shape == (2, 15, 64)  # per-token


def test_expert_global_timestep(config):
    """Expert forward with standard global timestep (B,)."""
    expert = create_action_expert(config)
    B, H, D = 2, 15, 32
    noised = torch.randn(B, H, D)
    t = torch.rand(B)
    context = torch.randn(B, 20, 64)
    state = torch.randn(B, D)
    out = expert(noised, t, context, state)
    assert out.shape == (B, H, D)


def test_expert_per_token_timestep(config):
    """Expert forward with per-token timestep (B, H) for RTC."""
    expert = create_action_expert(config)
    B, H, D = 2, 15, 32
    noised = torch.randn(B, H, D)
    t = torch.rand(B, H)  # per-token
    context = torch.randn(B, 20, 64)
    state = torch.randn(B, D)
    out = expert(noised, t, context, state)
    assert out.shape == (B, H, D)


def test_flow_loss_no_rtc(config):
    """Standard flow matching loss (no prefix)."""
    expert = create_action_expert(config)
    loss_fn = FlowMatchingLoss()
    B, H, D = 2, 15, 32
    actions = torch.randn(B, H, D)
    state = torch.randn(B, D)
    context = torch.randn(B, 20, 64)
    loss = loss_fn(expert, actions, state, context, prefix_len=0)
    assert loss.shape == ()
    assert loss.item() > 0


def test_flow_loss_with_rtc(config):
    """Flow matching loss with RTC prefix conditioning."""
    expert = create_action_expert(config)
    loss_fn = FlowMatchingLoss()
    B, H, D = 2, 15, 32
    actions = torch.randn(B, H, D)
    state = torch.randn(B, D)
    context = torch.randn(B, 20, 64)
    loss = loss_fn(expert, actions, state, context, prefix_len=4)
    assert loss.shape == ()
    assert loss.item() > 0


def test_flow_loss_rtc_gradient_flows(config):
    """Verify gradients flow through postfix tokens with RTC."""
    expert = create_action_expert(config)
    loss_fn = FlowMatchingLoss()
    B, H, D = 2, 15, 32
    actions = torch.randn(B, H, D)
    state = torch.randn(B, D)
    context = torch.randn(B, 20, 64)

    loss = loss_fn(expert, actions, state, context, prefix_len=4)
    loss.backward()

    # Check gradients exist
    has_grad = False
    for p in expert.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients found in expert after RTC backward"


def test_sampler_no_rtc(config):
    """Standard sampling (no prefix)."""
    expert = create_action_expert(config)
    sampler = FlowMatchingSampler(num_steps=3)
    B, H, D = 2, 15, 32
    state = torch.randn(B, D)
    context = torch.randn(B, 20, 64)
    actions = sampler.sample(expert, (B, H, D), state, context, device="cpu")
    assert actions.shape == (B, H, D)


def test_sampler_with_rtc_prefix(config):
    """Sampling with RTC action prefix."""
    expert = create_action_expert(config)
    sampler = FlowMatchingSampler(num_steps=3)
    B, H, D = 2, 15, 32
    d = 4  # prefix length
    state = torch.randn(B, D)
    context = torch.randn(B, 20, 64)
    prefix = torch.randn(B, d, D)

    actions = sampler.sample(
        expert, (B, H, D), state, context, device="cpu", action_prefix=prefix,
    )
    assert actions.shape == (B, H, D)
    # Prefix should be preserved
    assert torch.allclose(actions[:, :d], prefix, atol=1e-5)
