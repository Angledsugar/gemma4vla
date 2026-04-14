"""Configuration for Gemma 4-native Action Expert.

Aligned with π0.6 architecture (Nov 2025 model card):
  - π0.6: SigLIP 400M + Gemma 3 4B backbone + 860M action expert
  - Ours: SigLIP (Gemma4) + Gemma 4 E4B backbone + ~860M action expert

Key π0.6 design choices replicated here:
  - Action expert has SAME number of layers as backbone
  - Expert width is ~half of backbone width (2:1 ratio)
  - Both flow matching AND FAST token generation
  - Knowledge Insulation: backbone predicts FAST, expert predicts flow
  - Image resolution: 448×448
  - Up to 4 camera inputs
  - 5 denoise steps (not 10)

Dimension mapping:
  π0.6:  Gemma 3 4B (width=3072?, depth=N) + expert (860M, depth=N)
  Ours:  Gemma 4 E4B (width=2560, depth=42) + expert (~860M, depth=42, width=1280)
"""

import dataclasses
from typing import Literal


ExpertVariant = Literal[
    "flow_transformer",   # Cross-attention transformer (recommended)
    "flow_shared",        # Shared attention like π0.5/π0.6
    "flow_mlp",           # Simple MLP head (baseline)
    "fast",               # FAST autoregressive tokens (π0-FAST style)
]


@dataclasses.dataclass(frozen=True)
class Gemma4BackboneConfig:
    """Gemma 4 E4B backbone dimensions."""
    hidden_size: int = 2560
    num_layers: int = 42
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 10240
    vocab_size: int = 262144


@dataclasses.dataclass
class Gemma4ActionExpertConfig:
    """Configuration for the Gemma 4-native Action Expert.

    Aligned with π0.6: expert depth = backbone depth, ~860M params.
    """

    # ---- Backbone reference ----
    backbone: Gemma4BackboneConfig = dataclasses.field(default_factory=Gemma4BackboneConfig)

    # ---- Expert variant selection ----
    # flow_shared = π0.6 style (shared attention with backbone, recommended)
    variant: ExpertVariant = "flow_shared"

    # ---- Expert architecture (π0.6-aligned) ----
    # Width sized to match π0.6's ~860M at backbone depth.
    # Gemma 4 E4B: 42 layers. At width=896, mlp=3584 → ~917M (closest to 860M).
    expert_width: int = 896
    # Depth = backbone depth (π0.6: "same number of layers as backbone")
    expert_depth: int = 42
    expert_heads: int = 8
    expert_kv_heads: int = 2
    expert_head_dim: int = 256
    expert_mlp_dim: int = 3584       # 4x expert_width

    # ---- Action space ----
    action_dim: int = 32             # 32 internal, 8 for DROID output
    action_horizon: int = 15         # 15 for DROID, 50 for general

    # ---- Flow matching (π0.6: 5 steps, not 10) ----
    flow_matching_steps: int = 5
    time_embed_dim: int = 896       # Match expert_width

    # ---- Timestep conditioning ----
    use_adarms: bool = True

    # ---- Image config (π0.6: 448×448, up to 4 cameras) ----
    image_size: int = 448
    max_cameras: int = 4

    # ---- MLP head config (baseline) ----
    mlp_hidden_mult: int = 4
    mlp_num_layers: int = 3

    # ---- FAST config ----
    fast_max_seq_len: int = 512
    fast_tokenizer_model: str = "physical-intelligence/fast"

    # ---- Attention pattern (π0.6 style) ----
    # image tokens: bidirectional
    # text tokens: causal
    # action tokens (expert): bidirectional
    image_attention: str = "bidirectional"
    text_attention: str = "causal"
    action_attention: str = "bidirectional"

    # ---- Training ----
    dtype: str = "bfloat16"

    # ---- Token strategy ----
    token_strategy: Literal["action_tokens", "last_hidden", "pooled"] = "action_tokens"

    @property
    def inner_dim(self) -> int:
        return self.expert_heads * self.expert_head_dim

    @property
    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)

    @property
    def estimated_expert_params(self) -> int:
        """Rough parameter count for the action expert."""
        # Per transformer layer: ~12 * width^2 (attn + ffn)
        per_layer = 12 * self.expert_width ** 2
        # Total: depth * per_layer + projections
        return self.expert_depth * per_layer + self.expert_width * self.action_dim * 2
