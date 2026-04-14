"""Projectors: vision features → backbone token space, state → expert token space.

In π0.5, SigLIP outputs 2048-dim features matching PaliGemma's hidden size.
For Gemma 4, the vision projector maps to 2560-dim.

State projection maps proprioception (action_dim) into expert token space.
"""

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """Projects vision encoder output to backbone hidden size.

    TODO: Replace with actual Gemma 4 SigLIP vision tower.
    Currently simulates the projection for shape testing.
    """

    def __init__(self, vision_dim: int = 1152, backbone_dim: int = 2560):
        super().__init__()
        # 2-layer MLP projector (following LLaVA 1.5 / PaliGemma design)
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, backbone_dim, bias=True),
            nn.GELU(),
            nn.Linear(backbone_dim, backbone_dim, bias=True),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, num_patches, vision_dim) from SigLIP/ViT
        Returns:
            (B, num_patches, backbone_dim) projected into backbone space
        """
        return self.proj(vision_features)


class DummyVisionEncoder(nn.Module):
    """Placeholder vision encoder for testing without real SigLIP.

    TODO: Replace with Gemma 4's actual vision tower.
    """

    def __init__(self, image_size: int = 448, patch_size: int = 14, vision_dim: int = 1152):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2  # 256 for 224/14
        self.vision_dim = vision_dim
        # Simple conv to get patch embeddings
        self.patch_embed = nn.Conv2d(3, vision_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(vision_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            (B, num_patches, vision_dim)
        """
        x = self.patch_embed(images)  # (B, vision_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, vision_dim)
        return self.norm(x)


class StateProjector(nn.Module):
    """Projects proprioceptive state into expert token space.

    In π0.5:
      - pi0: state_proj(action_dim → expert_width) as a separate token
      - pi0.5: state is discretized into language tokens (no separate projection)

    Here we support both modes.
    """

    def __init__(self, state_dim: int, expert_width: int, mode: str = "continuous"):
        super().__init__()
        self.mode = mode
        if mode == "continuous":
            self.proj = nn.Linear(state_dim, expert_width)
        # "discrete" mode: state is tokenized into language tokens (handled upstream)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
        Returns:
            (B, 1, expert_width) — single state token
        """
        if self.mode == "continuous":
            return self.proj(state).unsqueeze(1)
        raise ValueError(f"StateProjector mode '{self.mode}' requires upstream tokenization")
