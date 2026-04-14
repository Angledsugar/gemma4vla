"""Tests for Knowledge Insulation training strategy."""

import torch
import pytest

from gemma4vla.model.config import Gemma4ActionExpertConfig
from gemma4vla.model.policy import Gemma4VLAPolicy
from gemma4vla.training.knowledge_insulation import (
    ActionTokenizer,
    KnowledgeInsulationTrainer,
)


B, H, D = 2, 15, 32


def test_action_tokenizer_encode():
    """Tokenizer should produce valid bin indices."""
    tok = ActionTokenizer(num_bins=256)
    actions = torch.rand(B, H, D) * 2 - 1  # [-1, 1]
    tokens = tok.encode(actions)

    assert tokens.dtype == torch.long
    assert tokens.min() >= 0
    assert tokens.max() < 256
    assert tokens.shape == (B, H, D)


def test_fast_action_head_shape():
    """FAST action head should output (B, S, vocab_size) logits."""
    from gemma4vla.model.fast_tokenizer import FASTActionHead
    head = FASTActionHead(hidden_size=2560, vocab_size=1000)
    features = torch.randn(B, 100, 2560)
    logits = head(features)
    assert logits.shape == (B, 100, 1000)


def test_knowledge_insulation_train_step():
    """One KI train step should produce backbone + expert losses."""
    config = Gemma4ActionExpertConfig(variant="flow_mlp", action_dim=D, action_horizon=H, expert_depth=4)
    policy = Gemma4VLAPolicy(config)

    ki = KnowledgeInsulationTrainer(policy, use_fast_backbone=False, device="cpu")

    images = [torch.randn(B, 3, 448, 448)]
    lang_tokens = torch.randint(0, 1000, (B, 16))
    state = torch.randn(B, D)
    actions = torch.randn(B, H, D)

    losses = ki.train_step(images, lang_tokens, state, actions)

    assert "backbone_loss" in losses
    assert "expert_loss" in losses
    assert "total_loss" in losses
    assert losses["backbone_loss"] > 0
    assert losses["expert_loss"] > 0


def test_gradient_isolation():
    """Expert loss should NOT produce gradients in backbone."""
    config = Gemma4ActionExpertConfig(variant="flow_mlp", action_dim=D, action_horizon=H, expert_depth=4)
    policy = Gemma4VLAPolicy(config)

    # Forward through backbone
    images = [torch.randn(B, 3, 448, 448)]
    lang_tokens = torch.randint(0, 1000, (B, 16))
    context = policy.encode_observation(images, lang_tokens)

    # Expert loss with detached context
    context_detached = context.detach()
    expert_loss = policy.flow_loss(
        predict_velocity_fn=policy.action_expert,
        target_actions=torch.randn(B, H, D),
        state=torch.randn(B, D),
        context=context_detached,
    )
    expert_loss.backward()

    # Backbone params should have NO gradients
    for name, p in policy.backbone.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() == 0, f"Backbone param {name} has gradient (should be isolated)"

    # Expert params SHOULD have gradients
    has_expert_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.action_expert.parameters()
    )
    assert has_expert_grad, "Expert should have gradients"
