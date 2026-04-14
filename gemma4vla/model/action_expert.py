"""Gemma 4-native Action Expert variants.

Three architectures, each adapted for Gemma 4's 2560-dim hidden space:

1. FlowTransformerExpert (recommended):
   - Separate transformer with cross-attention to backbone KV
   - Most flexible, clean separation of concerns
   - Width=1280, depth=18, cross-attn to 2560-dim backbone

2. FlowSharedExpert:
   - Shared attention with backbone tokens (π0.5-style)
   - Action tokens (1280-dim) attend alongside backbone tokens (2560-dim)
   - Requires matched head_dim for attention compatibility

3. FlowMLPExpert:
   - Simple MLP on pooled backbone features
   - Fast but no temporal action structure
   - Baseline comparison

All variants use flow matching: given noised actions + timestep, predict velocity field.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import Gemma4ActionExpertConfig


# ============================================================================
# Shared components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm conditioned on timestep embedding.

    Follows π0.5's adaRMS: normed * (1 + scale) + shift, with gating.
    Conditioning generates (scale, shift, gate) via a dense layer.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # Modulation: cond → (scale, shift, gate)
        self.modulation = nn.Linear(cond_dim, 3 * dim, bias=True)
        nn.init.normal_(self.modulation.weight, std=0.02)
        # Initialize gate bias to 1 so gate starts open
        with torch.no_grad():
            self.modulation.bias.zero_()
            self.modulation.bias[2 * dim:] = 1.0  # gate portion starts at 1

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        Args:
            x: (B, T, D)
            cond: (B, cond_dim) or (B, T, cond_dim) for per-token conditioning (RTC)
        Returns:
            (normed_x, gate) where gate is (B, 1, D) or (B, T, D) for residual gating
        """
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        normed = (x.float() * norm).to(x.dtype) * self.weight

        # Modulation — support per-token conditioning for Training-Time RTC
        if cond.dim() == 3:
            # Per-token: cond is (B, T, cond_dim) → mod is (B, T, 3*D)
            mod = self.modulation(cond)
            scale, shift, gate = mod.chunk(3, dim=-1)  # each (B, T, D)
        else:
            # Global: cond is (B, cond_dim) → broadcast over T
            mod = self.modulation(cond)  # (B, 3*D)
            scale, shift, gate = mod.chunk(3, dim=-1)  # each (B, D)
            scale = scale.unsqueeze(1)  # (B, 1, D)
            shift = shift.unsqueeze(1)
            gate = gate.unsqueeze(1)

        return normed * (1 + scale) + shift, gate


class GatedFFN(nn.Module):
    """Gated feed-forward network (SwiGLU-style, matching Gemma)."""

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def sinusoidal_embedding(t: torch.Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
    """Sinusoidal positional embedding for flow matching timestep.

    Matches π0.5's create_sinusoidal_pos_embedding.
    Supports both global timestep (B,) and per-token timestep (B, T) for RTC.
    """
    assert dim % 2 == 0
    half = dim // 2
    fraction = torch.linspace(0, 1, half, device=t.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period) * 2 * math.pi

    if t.dim() == 1:
        # Global: (B,) → (B, dim)
        sin_input = scaling[None, :] * t.float()[:, None]
        emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
        return emb  # (B, dim)
    else:
        # Per-token: (B, T) → (B, T, dim)
        sin_input = scaling[None, None, :] * t.float().unsqueeze(-1)  # (B, T, half)
        emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
        return emb  # (B, T, dim)


# ============================================================================
# Variant 1: FlowTransformerExpert (cross-attention to backbone)
# ============================================================================

class CrossAttentionLayer(nn.Module):
    """Cross-attention: action tokens query backbone KV."""

    def __init__(self, expert_dim: int, context_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(expert_dim, inner, bias=False)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, expert_dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = rearrange(self.q_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k_proj(context), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.v_proj(context), "b s (h d) -> b h s d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_proj(out)


class SelfAttentionLayer(nn.Module):
    """Causal self-attention for action tokens."""

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.q_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_proj(out)


class TransformerExpertBlock(nn.Module):
    """One block: adaRMS → self-attn → adaRMS → cross-attn → adaRMS → FFN."""

    def __init__(self, config: Gemma4ActionExpertConfig):
        super().__init__()
        D = config.expert_width
        C = config.backbone.hidden_size

        if config.use_adarms:
            self.norm1 = AdaRMSNorm(D, D)
            self.norm2 = AdaRMSNorm(D, D)
            self.norm3 = AdaRMSNorm(D, D)
        else:
            self.norm1 = RMSNorm(D)
            self.norm2 = RMSNorm(D)
            self.norm3 = RMSNorm(D)

        self.self_attn = SelfAttentionLayer(D, config.expert_heads, config.expert_head_dim)
        self.cross_attn = CrossAttentionLayer(D, C, config.expert_heads, config.expert_head_dim)
        self.ffn = GatedFFN(D, config.expert_mlp_dim)
        self.use_adarms = config.use_adarms

    def _norm(self, norm_layer, x, cond=None):
        if self.use_adarms:
            normed, gate = norm_layer(x, cond)
            return normed, gate
        return norm_layer(x), None

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        normed, gate = self._norm(self.norm1, x, cond)
        sa_out = self.self_attn(normed)
        x = x + sa_out if gate is None else x + gate * sa_out

        # Cross-attention to backbone
        normed, gate = self._norm(self.norm2, x, cond)
        ca_out = self.cross_attn(normed, context)
        x = x + ca_out if gate is None else x + gate * ca_out

        # FFN
        normed, gate = self._norm(self.norm3, x, cond)
        ff_out = self.ffn(normed)
        x = x + ff_out if gate is None else x + gate * ff_out

        return x


class FlowTransformerExpert(nn.Module):
    """Variant 1: Cross-attention transformer action expert.

    - Separate transformer with its own self-attention
    - Cross-attends to frozen backbone hidden states
    - Clean separation: backbone forward → cache features → expert forward

    Trade-offs:
      + Easy to swap backbone without retraining expert
      + No dimension matching issues in attention (separate Q,K,V)
      + Backbone stays fully frozen
      - Information flow is indirect (only through cross-attention)
      - Requires learning cross-attention alignment from scratch
    """

    def __init__(self, config: Gemma4ActionExpertConfig):
        super().__init__()
        D = config.expert_width

        # Action input/output projections (matches π0.5)
        self.action_in_proj = nn.Linear(config.action_dim, D)
        self.action_out_proj = nn.Linear(D, config.action_dim)
        nn.init.normal_(self.action_out_proj.weight, std=0.01)
        nn.init.zeros_(self.action_out_proj.bias)

        # Timestep conditioning MLP (matches π0.5's time_mlp)
        self.time_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D),
            nn.SiLU(),
        )

        # Positional embedding for action tokens
        self.pos_embed = nn.Parameter(torch.randn(1, config.action_horizon, D) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerExpertBlock(config) for _ in range(config.expert_depth)
        ])
        self.final_norm = RMSNorm(D)

        self.config = config

    def forward(
        self,
        noised_actions: torch.Tensor,  # (B, H, action_dim)
        timestep: torch.Tensor,         # (B,)
        context: torch.Tensor,          # (B, S, backbone_dim=2560)
        state: torch.Tensor = None,     # (B, action_dim) optional
    ) -> torch.Tensor:
        """Predict velocity field for flow matching.

        Returns: (B, H, action_dim)
        """
        D = self.config.expert_width
        # Cast all inputs to model dtype for consistency
        dtype = self.action_in_proj.weight.dtype
        noised_actions = noised_actions.to(dtype)
        context = context.to(dtype)
        if state is not None:
            state = state.to(dtype)

        # Action embedding + positional
        action_tokens = self.action_in_proj(noised_actions) + self.pos_embed.to(dtype)

        # Timestep conditioning
        time_emb = sinusoidal_embedding(timestep, D).to(dtype)
        time_cond = self.time_mlp(time_emb)  # (B, D) — used for adaRMS

        # Transformer blocks
        for block in self.blocks:
            action_tokens = block(action_tokens, context, cond=time_cond)

        # Output
        action_tokens = self.final_norm(action_tokens)
        return self.action_out_proj(action_tokens)


# ============================================================================
# Variant 2: FlowSharedExpert (π0.6-style shared attention)
# ============================================================================

class ExpertLayer(nn.Module):
    """Single expert transformer layer (trainable).

    Matches backbone layer's attention pattern:
      - Sliding layers: head_dim=256
      - Global layers:  head_dim=512
    This ensures shared attention is possible at every layer.
    """

    def __init__(self, config: Gemma4ActionExpertConfig, head_dim: int = 256):
        super().__init__()
        E_dim = config.expert_width
        q_dim = config.expert_heads * head_dim
        kv_dim = config.expert_kv_heads * head_dim

        # Expert attention projections (GQA, matching backbone's head_dim)
        self.q_proj = nn.Linear(E_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(E_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(E_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, E_dim, bias=False)

        # Expert norms with adaRMS conditioning
        if config.use_adarms:
            self.input_norm = AdaRMSNorm(E_dim, E_dim)
            self.post_attn_norm = AdaRMSNorm(E_dim, E_dim)
        else:
            self.input_norm = RMSNorm(E_dim)
            self.post_attn_norm = RMSNorm(E_dim)

        # Expert FFN
        self.ffn = GatedFFN(E_dim, config.expert_mlp_dim)

        self.num_heads = config.expert_heads
        self.num_kv_heads = config.expert_kv_heads
        self.head_dim = head_dim  # per-layer: 256 (sliding) or 512 (global)
        self.scale = head_dim ** -0.5
        self.use_adarms = config.use_adarms


class FlowSharedExpert(nn.Module):
    """Variant 2: Shared-attention action expert (π0.6-style).

    Architecture (per layer):
      1. Backbone layer normalizes backbone tokens (using backbone's own norm)
      2. Expert layer normalizes action tokens (using expert's adaRMSNorm)
      3. Both compute Q, K, V with their own projections
      4. Q, K, V are concatenated → shared multi-head attention
      5. Output is split → each model's O projection
      6. Separate residual connections
      7. Separate FFN (backbone FFN = backbone's, expert FFN = expert's)

    Key difference from previous implementation:
      Backbone attention/FFN weights are REFERENCED from the actual backbone,
      not duplicated. Only expert layers (~860M) are trainable.

    When backbone_layers=None (dummy mode), uses standalone expert layers
    without shared attention (just self-attention on action tokens).
    """

    def __init__(self, config: Gemma4ActionExpertConfig):
        super().__init__()
        D = config.expert_width

        self.action_in_proj = nn.Linear(config.action_dim, D)
        self.action_out_proj = nn.Linear(D, config.action_dim)
        nn.init.normal_(self.action_out_proj.weight, std=0.01)
        nn.init.zeros_(self.action_out_proj.bias)

        self.time_mlp = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D), nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, config.action_horizon, D) * 0.02)

        # Expert layers (trainable, ~917M total)
        # Match Gemma 4's sliding/global pattern:
        #   Every 6th layer (starting at 5) + last layer = global (head_dim=512)
        #   All others = sliding (head_dim=256)
        self.global_layer_indices = set()
        for i in range(config.expert_depth):
            if (i + 1) % 6 == 0 or i == config.expert_depth - 1:
                self.global_layer_indices.add(i)

        self.expert_layers = nn.ModuleList([
            ExpertLayer(config, head_dim=512 if i in self.global_layer_indices else 256)
            for i in range(config.expert_depth)
        ])
        self.final_norm = RMSNorm(D)
        self.config = config

        # Backbone layers are set externally via set_backbone_layers()
        self._backbone_layers = None

    def set_backbone_layers(self, backbone_layers):
        """Set reference to backbone decoder layers (frozen, not owned).

        Called after loading the real Gemma 4 backbone.
        Stored as plain Python list to avoid registering backbone params
        as part of this module (they stay frozen in the backbone).
        """
        # CRITICAL: store as plain list, NOT nn.ModuleList
        # Otherwise PyTorch registers backbone params under expert
        self._backbone_layers = list(backbone_layers)

    def _shared_attention_step(
        self,
        backbone_tokens: torch.Tensor,   # (B, S, 2560)
        expert_tokens: torch.Tensor,     # (B, H, 1280)
        backbone_layer,                   # backbone decoder layer (frozen)
        expert_layer: ExpertLayer,        # expert layer (trainable)
        cond: torch.Tensor = None,       # (B, 1280) adaRMS conditioning
    ):
        """One layer of shared attention between backbone and expert tokens.

        Follows π0.6 / OpenPI compute_layer_complete exactly.
        """
        # Get backbone layer's actual head_dim (varies: 256 for sliding, 512 for global)
        backbone_head_dim = backbone_layer.self_attn.head_dim
        backbone_num_heads = backbone_layer.self_attn.q_proj.weight.shape[0] // backbone_head_dim
        backbone_num_kv = backbone_layer.self_attn.k_proj.weight.shape[0] // backbone_head_dim

        # Expert uses its own fixed head_dim
        h = expert_layer.num_heads
        d = expert_layer.head_dim
        scale = expert_layer.scale

        # === Pre-attention normalization ===
        # Backbone: uses its own RMSNorm (frozen)
        # Gemma 4's layernorm may return just tensor or (tensor, gate)
        try:
            b_normed = backbone_layer.input_layernorm(backbone_tokens)
        except TypeError:
            # Some Gemma versions need cond=None explicitly
            b_normed = backbone_layer.input_layernorm(backbone_tokens, cond=None)
        if isinstance(b_normed, tuple):
            b_normed, b_gate = b_normed
        else:
            b_gate = None

        # Expert: uses adaRMSNorm with timestep conditioning
        if expert_layer.use_adarms:
            e_normed, e_gate = expert_layer.input_norm(expert_tokens, cond)
        else:
            e_normed = expert_layer.input_norm(expert_tokens)
            e_gate = None

        # === Q, K, V projections (separate per model) ===
        # Backbone: use backbone's own head_dim (256 for sliding, 512 for global)
        b_q_raw = backbone_layer.self_attn.q_proj(b_normed)
        b_k_raw = backbone_layer.self_attn.k_proj(b_normed)
        b_v_raw = backbone_layer.self_attn.v_proj(b_normed)

        b_q = rearrange(b_q_raw, "b s (h d) -> b h s d", h=backbone_num_heads, d=backbone_head_dim)
        b_k = rearrange(b_k_raw, "b s (h d) -> b h s d", h=backbone_num_kv, d=backbone_head_dim)
        b_v = rearrange(b_v_raw, "b s (h d) -> b h s d", h=backbone_num_kv, d=backbone_head_dim)
        if backbone_num_kv < backbone_num_heads:
            rep = backbone_num_heads // backbone_num_kv
            b_k = b_k.repeat_interleave(rep, dim=1)
            b_v = b_v.repeat_interleave(rep, dim=1)

        # Expert: use expert's head_dim (always 256)
        e_q = rearrange(expert_layer.q_proj(e_normed), "b n (h d) -> b h n d", h=h, d=d)
        e_k_raw = expert_layer.k_proj(e_normed)
        e_v_raw = expert_layer.v_proj(e_normed)
        n_kv = e_k_raw.shape[-1] // d
        e_k = rearrange(e_k_raw, "b n (h d) -> b h n d", h=n_kv, d=d)
        e_v = rearrange(e_v_raw, "b n (h d) -> b h n d", h=n_kv, d=d)
        if n_kv < h:
            e_k = e_k.repeat_interleave(h // n_kv, dim=1)
            e_v = e_v.repeat_interleave(h // n_kv, dim=1)

        # Expert layer now matches backbone's head_dim (256 or 512 per layer)
        # === Concatenate for shared attention ===
        q = torch.cat([b_q, e_q], dim=2)
        k = torch.cat([b_k, e_k], dim=2)
        v = torch.cat([b_v, e_v], dim=2)

        # === Shared multi-head attention ===
        attn = (q @ k.transpose(-2, -1)) * scale
        # Mask: backbone tokens don't attend to action tokens (π0.6)
        S = backbone_tokens.shape[1]
        H_act = expert_tokens.shape[1]
        total = S + H_act
        mask = torch.zeros(total, total, device=q.device, dtype=torch.bool)
        mask[:, :] = True
        mask[:S, S:] = False  # backbone can't see action tokens
        attn = attn.masked_fill(~mask[None, None], float("-inf"))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")

        # === Split and output projection ===
        b_attn_out = backbone_layer.self_attn.o_proj(out[:, :S])
        e_attn_out = expert_layer.o_proj(out[:, S:])

        # === First residual ===
        backbone_tokens = backbone_tokens + b_attn_out
        expert_tokens = expert_tokens + (e_gate * e_attn_out if e_gate is not None else e_attn_out)

        # === Post-attention norm + FFN ===
        try:
            b_normed2 = backbone_layer.post_attention_layernorm(backbone_tokens)
        except TypeError:
            b_normed2 = backbone_layer.post_attention_layernorm(backbone_tokens, cond=None)
        if isinstance(b_normed2, tuple):
            b_normed2, b_gate2 = b_normed2
        else:
            b_gate2 = None

        if expert_layer.use_adarms:
            e_normed2, e_gate2 = expert_layer.post_attn_norm(expert_tokens, cond)
        else:
            e_normed2 = expert_layer.post_attn_norm(expert_tokens)
            e_gate2 = None

        # Separate FFNs
        b_ff = backbone_layer.mlp(b_normed2)
        e_ff = expert_layer.ffn(e_normed2)

        # Second residual
        backbone_tokens = backbone_tokens + b_ff
        expert_tokens = expert_tokens + (e_gate2 * e_ff if e_gate2 is not None else e_ff)

        return backbone_tokens, expert_tokens

    def _self_attention_only_layer(self, backbone_tokens, expert_tokens, expert_layer, cond, b_gate, e_gate):
        """Fallback for head_dim mismatch layers (global attention).

        Backbone processes through its own layer normally.
        Expert does self-attention only (no shared attention this layer).
        """
        # Expert self-attention
        expert_tokens = self._self_attention_only(expert_tokens, expert_layer, cond)

        # Backbone still needs to go through its own layer
        # (backbone_tokens are already normed above, but we return as-is
        #  since backbone forward is handled by the real backbone)
        return backbone_tokens, expert_tokens

    def _self_attention_only(self, expert_tokens, expert_layer, cond):
        """Fallback: expert self-attention only (no backbone sharing)."""
        h = expert_layer.num_heads
        d = expert_layer.head_dim
        scale = expert_layer.scale

        if expert_layer.use_adarms:
            normed, gate = expert_layer.input_norm(expert_tokens, cond)
        else:
            normed = expert_layer.input_norm(expert_tokens)
            gate = None

        q = rearrange(expert_layer.q_proj(normed), "b n (h d) -> b h n d", h=h)
        # GQA: K,V may have fewer heads
        k_raw = expert_layer.k_proj(normed)
        v_raw = expert_layer.v_proj(normed)
        n_kv = k_raw.shape[-1] // d
        k = rearrange(k_raw, "b n (h d) -> b h n d", h=n_kv)
        v = rearrange(v_raw, "b n (h d) -> b h n d", h=n_kv)
        if n_kv < h:
            repeats = h // n_kv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        attn_out = expert_layer.o_proj(out)

        expert_tokens = expert_tokens + (gate * attn_out if gate is not None else attn_out)

        if expert_layer.use_adarms:
            normed2, gate2 = expert_layer.post_attn_norm(expert_tokens, cond)
        else:
            normed2 = expert_layer.post_attn_norm(expert_tokens)
            gate2 = None

        ff_out = expert_layer.ffn(normed2)
        expert_tokens = expert_tokens + (gate2 * ff_out if gate2 is not None else ff_out)
        return expert_tokens

    def forward(self, noised_actions, timestep, context, state=None):
        """Forward pass with optional per-token timestep (Training-Time RTC).

        Args:
            noised_actions: (B, H, D) — noised action chunk
            timestep: (B,) global or (B, H) per-token flow matching timestep
            context: (B, S, 2560) — backbone embeddings
            state: unused (reserved for proprioception)
        """
        D = self.config.expert_width
        dtype = self.action_in_proj.weight.dtype
        noised_actions = noised_actions.to(dtype)
        context = context.to(dtype)

        action_tokens = self.action_in_proj(noised_actions) + self.pos_embed.to(dtype)
        time_emb = sinusoidal_embedding(timestep, D).to(dtype)
        # time_emb: (B, D) for global, (B, H, D) for per-token
        time_cond = self.time_mlp(time_emb)
        # time_cond: (B, D) or (B, H, D) — AdaRMSNorm handles both

        if self._backbone_layers is not None:
            # π0.6 mode: shared attention with backbone layers
            # Use gradient checkpointing to save ~40% VRAM on 42 layers
            for backbone_layer, expert_layer in zip(self._backbone_layers, self.expert_layers):
                if self.training and torch.is_grad_enabled():
                    context, action_tokens = torch.utils.checkpoint.checkpoint(
                        self._shared_attention_step,
                        context, action_tokens, backbone_layer, expert_layer, time_cond,
                        use_reentrant=False,
                    )
                else:
                    context, action_tokens = self._shared_attention_step(
                        context, action_tokens, backbone_layer, expert_layer, cond=time_cond
                    )
        else:
            # Dummy mode: expert self-attention only (for testing without backbone)
            for expert_layer in self.expert_layers:
                action_tokens = self._self_attention_only(action_tokens, expert_layer, cond=time_cond)

        action_tokens = self.final_norm(action_tokens)
        return self.action_out_proj(action_tokens)


# ============================================================================
# Variant 3: FlowMLPExpert (simple baseline)
# ============================================================================

class FlowMLPExpert(nn.Module):
    """Variant 3: Simple MLP action head.

    Pools backbone features, concatenates with noised actions and timestep,
    then predicts velocity through an MLP.

    Trade-offs:
      + Very simple, fast inference
      + Easy to train
      - No temporal structure (each action step is independent)
      - Information bottleneck (pooling loses spatial detail)
      - Baseline only, not expected to match transformer variants
    """

    def __init__(self, config: Gemma4ActionExpertConfig):
        super().__init__()
        C = config.backbone.hidden_size
        D = config.action_dim
        H = config.action_horizon
        hidden = config.expert_width

        # Pool backbone → fixed-size feature
        self.context_proj = nn.Linear(C, hidden)

        # Time embedding
        self.time_proj = nn.Linear(hidden, hidden)

        # Action MLP
        layers = []
        in_dim = hidden + D  # context + action
        for i in range(config.mlp_num_layers):
            out_dim = hidden if i < config.mlp_num_layers - 1 else D
            layers.extend([nn.Linear(in_dim, out_dim), nn.SiLU()] if i < config.mlp_num_layers - 1
                          else [nn.Linear(in_dim, out_dim)])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.config = config

    def forward(self, noised_actions, timestep, context, state=None):
        B, H, D = noised_actions.shape

        # Pool context
        ctx = context.mean(dim=1)  # (B, 2560)
        ctx = self.context_proj(ctx)  # (B, hidden)

        # Time
        time_emb = sinusoidal_embedding(timestep, self.config.expert_width)
        ctx = ctx + self.time_proj(time_emb)  # (B, hidden)

        # Per-step prediction
        ctx_expanded = ctx.unsqueeze(1).expand(B, H, -1)  # (B, H, hidden)
        x = torch.cat([ctx_expanded, noised_actions], dim=-1)  # (B, H, hidden+D)
        return self.mlp(x)  # (B, H, D)


# ============================================================================
# Factory
# ============================================================================

def create_action_expert(config: Gemma4ActionExpertConfig) -> nn.Module:
    """Create action expert from config."""
    variants = {
        "flow_transformer": FlowTransformerExpert,
        "flow_shared": FlowSharedExpert,
        "flow_mlp": FlowMLPExpert,
    }
    if config.variant not in variants:
        raise ValueError(f"Unknown variant: {config.variant}. Choose from {list(variants.keys())}")
    return variants[config.variant](config)
