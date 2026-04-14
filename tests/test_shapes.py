"""Shape sanity checks for all Action Expert variants.

Run: cd ~/project/gemma4vla && uv run python -m pytest tests/test_shapes.py -v
"""

import pytest
import torch

from gemma4vla.model.config import Gemma4ActionExpertConfig
from gemma4vla.model.action_expert import create_action_expert, sinusoidal_embedding
from gemma4vla.model.policy import Gemma4VLAPolicy


B = 2    # batch
H = 15   # action horizon
D = 32   # action dim
S = 280  # backbone sequence length (256 img + 24 lang tokens)
C = 2560 # backbone hidden size


@pytest.fixture(params=["flow_transformer", "flow_shared", "flow_mlp"])
def config(request):
    # Use depth=4 for fast testing (full depth=42 tested separately)
    return Gemma4ActionExpertConfig(variant=request.param, action_dim=D, action_horizon=H, expert_depth=4)


@pytest.fixture
def dummy_inputs():
    return {
        "noised_actions": torch.randn(B, H, D),
        "timestep": torch.rand(B),
        "context": torch.randn(B, S, C),
        "state": torch.randn(B, D),
    }


def test_action_expert_output_shape(config, dummy_inputs):
    """Expert output should be (B, H, action_dim)."""
    expert = create_action_expert(config)
    out = expert(**dummy_inputs)
    assert out.shape == (B, H, D), f"Expected ({B}, {H}, {D}), got {out.shape}"


def test_sinusoidal_embedding_shape():
    """Time embedding should be (B, dim)."""
    t = torch.rand(B)
    emb = sinusoidal_embedding(t, 1280)
    assert emb.shape == (B, 1280)


def test_policy_forward_loss():
    """Full policy should compute scalar loss."""
    config = Gemma4ActionExpertConfig(variant="flow_mlp", action_dim=D, action_horizon=H, expert_depth=4)
    policy = Gemma4VLAPolicy(config)

    images = [torch.randn(B, 3, 448, 448)]
    lang_tokens = torch.randint(0, 1000, (B, 16))
    state = torch.randn(B, D)
    actions = torch.randn(B, H, D)

    loss = policy.compute_loss(images, lang_tokens, state, actions)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"


def test_policy_inference_shape():
    """Inference should produce (B, H, action_dim) actions."""
    config = Gemma4ActionExpertConfig(variant="flow_mlp", action_dim=D, action_horizon=H, expert_depth=4)
    policy = Gemma4VLAPolicy(config)
    policy.eval()

    images = [torch.randn(B, 3, 448, 448)]
    lang_tokens = torch.randint(0, 1000, (B, 16))
    state = torch.randn(B, D)

    with torch.no_grad():
        actions = policy.predict_actions(images, lang_tokens, state)
    assert actions.shape == (B, H, D), f"Expected ({B}, {H}, {D}), got {actions.shape}"


def test_all_variants_produce_same_shape():
    """All variants should produce identical output shapes."""
    shapes = {}
    for variant in ["flow_transformer", "flow_shared", "flow_mlp"]:
        config = Gemma4ActionExpertConfig(variant=variant, action_dim=D, action_horizon=H, expert_depth=4)
        expert = create_action_expert(config)
        out = expert(
            torch.randn(B, H, D),
            torch.rand(B),
            torch.randn(B, S, C),
        )
        shapes[variant] = out.shape

    assert len(set(shapes.values())) == 1, f"Shape mismatch: {shapes}"


def test_param_counts():
    """Verify expert param counts at full depth=42 (π0.6 aligned)."""
    for variant in ["flow_transformer", "flow_shared", "flow_mlp"]:
        config = Gemma4ActionExpertConfig(variant=variant, action_dim=D, action_horizon=H)  # depth=42
        expert = create_action_expert(config)
        n_params = sum(p.numel() for p in expert.parameters())
        print(f"{variant}: {n_params/1e6:.1f}M params")
        assert n_params > 0
        if variant == "flow_mlp":
            assert n_params < 50_000_000  # < 50M


def test_gradient_flow(config, dummy_inputs):
    """Verify gradients flow through the expert."""
    expert = create_action_expert(config)
    # Ensure float32 for gradient computation
    expert = expert.float()
    inputs = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in dummy_inputs.items()}
    out = expert(**inputs)
    loss = out.pow(2).mean()
    loss.backward()

    trainable = [p for p in expert.parameters() if p.requires_grad]
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    assert has_grad, f"No gradients found in {config.variant} ({len(trainable)} trainable params)"
