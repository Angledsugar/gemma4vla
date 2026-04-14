"""Gemma4-VLA: Full model combining backbone + action expert.

Architecture:
    Gemma 4 (frozen) ──→ hidden states ──→ Cross-Attention ──→ Action Expert ──→ actions
                                              ↑
                                    flow matching denoiser
"""

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from .action_expert import ActionExpert
from .backbone import GEMMA4_DIMS, Gemma4Backbone
from .flow_matching import FlowMatchingLoss, FlowMatchingSampler


@dataclasses.dataclass
class Gemma4VLAConfig:
    """Configuration for the full VLA model."""

    # Backbone
    backbone_model: str = "google/gemma-4-E4B-it"
    backbone_dtype: str = "bfloat16"
    freeze_backbone: bool = True

    # Action Expert
    expert_width: int = 1024
    expert_depth: int = 18
    expert_heads: int = 8
    expert_head_dim: int = 64
    expert_mlp_dim: int = 4096

    # Action space
    action_dim: int = 8       # 7 arm joints + 1 gripper
    action_horizon: int = 15  # action chunk length (π0.5 default)

    # Flow matching
    flow_matching_steps: int = 10  # ODE integration steps at inference

    # LoRA (optional fine-tuning of action expert)
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 16.0

    @property
    def backbone_hidden_size(self) -> int:
        return GEMMA4_DIMS.get(self.backbone_model, 2560)

    @property
    def dtype(self):
        return getattr(torch, self.backbone_dtype)


class Gemma4VLA(nn.Module):
    """Gemma 4 backbone + π0.5-style action expert VLA.

    The backbone encodes images + language instruction into hidden states.
    The action expert cross-attends to these features and generates
    action chunks via flow matching.

    Trainable components:
        - Action expert (300M) — full or LoRA
        - Bridge projection is inside cross-attention (context_dim → expert_dim)

    Frozen components:
        - Gemma 4 backbone (vision encoder + language decoder)
    """

    def __init__(self, config: Gemma4VLAConfig):
        super().__init__()
        self.config = config

        # Action expert (trainable)
        self.action_expert = ActionExpert(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            width=config.expert_width,
            depth=config.expert_depth,
            num_heads=config.expert_heads,
            head_dim=config.expert_head_dim,
            mlp_dim=config.expert_mlp_dim,
            context_dim=config.backbone_hidden_size,
        )

        # Flow matching
        self.flow_loss = FlowMatchingLoss()
        self.sampler = FlowMatchingSampler(num_steps=config.flow_matching_steps)

        # Backbone (loaded separately to control device placement)
        self._backbone: Optional[Gemma4Backbone] = None

    def load_backbone(self, device: str = "cuda"):
        """Load the frozen Gemma 4 backbone. Call after model init."""
        self._backbone = Gemma4Backbone(
            model_name=self.config.backbone_model,
            device=device,
            dtype=self.config.dtype,
        )
        return self

    @property
    def backbone(self) -> Gemma4Backbone:
        if self._backbone is None:
            raise RuntimeError("Backbone not loaded. Call model.load_backbone() first.")
        return self._backbone

    def encode_observation(
        self,
        images: list[Image.Image],
        instruction: str,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Encode images + instruction into backbone features.

        Args:
            images: Camera images (e.g. [ext_cam, wrist_cam]).
            instruction: Language instruction.

        Returns:
            Backbone hidden states (1, seq_len, hidden_size).
        """
        return self.backbone.encode(images, instruction, device=device)

    def compute_loss(
        self,
        target_actions: torch.Tensor,  # (B, H, action_dim)
        state: torch.Tensor,           # (B, action_dim)
        context: torch.Tensor,         # (B, S, backbone_hidden_size)
    ) -> torch.Tensor:
        """Compute flow matching training loss."""
        return self.flow_loss(
            predict_velocity_fn=self.action_expert,
            target_actions=target_actions,
            state=state,
            context=context,
        )

    @torch.no_grad()
    def predict_actions(
        self,
        images: list[Image.Image],
        instruction: str,
        state: torch.Tensor,           # (action_dim,) — current proprioception
        device: str = "cuda",
    ) -> torch.Tensor:
        """Full inference: images + instruction + state → action chunk.

        Returns:
            Action chunk (action_horizon, action_dim).
        """
        # Encode observation
        context = self.encode_observation(images, instruction, device=device)

        # Prepare state
        state = state.unsqueeze(0).to(device=device, dtype=self.config.dtype)

        # Sample actions via flow matching ODE
        shape = (1, self.config.action_horizon, self.config.action_dim)
        actions = self.sampler.sample(
            predict_velocity_fn=self.action_expert,
            shape=shape,
            state=state,
            context=context,
            device=device,
        )

        return actions.squeeze(0)  # (H, action_dim)

    def get_trainable_parameters(self):
        """Return only trainable parameters (action expert)."""
        return self.action_expert.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.get_trainable_parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        total = sum(p.numel() for p in self.action_expert.parameters())
        if self._backbone is not None:
            total += sum(p.numel() for p in self._backbone.model.parameters())
        return total
