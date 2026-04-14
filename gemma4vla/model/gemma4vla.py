"""Gemma4VLA: π0.5 with Gemma 4 vision encoder.

Replaces PaliGemma's SigLIP vision encoder with Gemma 4's vision encoder
while keeping the entire π0.5 decoder + action expert + flow matching intact.

Architecture:
    Gemma 4 Vision Encoder (frozen)
        ↓ image features
    Linear(gemma4_img_dim → 2048)  ← only trainable component
        ↓ projected features
    PaliGemma Language Model (from π0.5, frozen)  ← generates KV cache
        ↓ KV cache
    Gemma 300M Action Expert (from π0.5, frozen)  ← generates actions
        ↓ flow matching
    Action chunk (15, 8)
"""

import logging
import sys
import os

import safetensors.torch
import torch
import torch.nn as nn
from PIL import Image

# Add openpi to path for PI0Pytorch
OPENPI_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "openpi")
if os.path.isdir(os.path.join(OPENPI_ROOT, "src")):
    sys.path.insert(0, os.path.join(OPENPI_ROOT, "src"))

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.pi0_config import Pi0Config

logger = logging.getLogger(__name__)


class Gemma4VisionBridge(nn.Module):
    """Bridge between Gemma 4 vision encoder and PaliGemma's embedding space.

    Encodes images with Gemma 4's vision model and projects to
    PaliGemma's expected image embedding format (B, num_tokens, 2048).
    """

    def __init__(self, gemma4_model_name: str = "google/gemma-4-E4B-it", device="cuda"):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading Gemma 4 vision encoder: {gemma4_model_name}")
        self.processor = AutoProcessor.from_pretrained(gemma4_model_name)

        # Load full model but we only use the vision tower
        full_model = AutoModelForImageTextToText.from_pretrained(
            gemma4_model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )

        # Extract vision tower and multi-modal projector
        self.vision_tower = full_model.vision_tower
        self.multi_modal_projector = full_model.multi_modal_projector

        # Freeze
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.multi_modal_projector.parameters():
            p.requires_grad = False

        # Get output dimension from the projector
        # Gemma 4 projects vision features to its text hidden_size
        gemma4_proj_dim = full_model.config.text_config.hidden_size  # 2560 for E4B
        paligemma_dim = 2048  # PaliGemma's expected embedding dimension

        # Projection: Gemma 4 vision output → PaliGemma embedding space
        self.bridge_proj = nn.Linear(gemma4_proj_dim, paligemma_dim, bias=False)
        nn.init.eye_(self.bridge_proj.weight[:paligemma_dim, :paligemma_dim])

        # Clean up the rest of the full model to save memory
        del full_model.language_model
        del full_model
        torch.cuda.empty_cache()

        logger.info(f"Gemma 4 vision bridge: {gemma4_proj_dim} → {paligemma_dim}")

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images using Gemma 4's vision tower + projector.

        Args:
            pixel_values: (B, C, H, W) preprocessed images.

        Returns:
            Image embeddings (B, num_tokens, 2048) matching PaliGemma format.
        """
        # Vision tower
        vision_outputs = self.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state

        # Multi-modal projector (maps to Gemma 4's text hidden_size)
        projected = self.multi_modal_projector(image_features)  # (B, N, 2560)

        # Bridge to PaliGemma's embedding space
        bridged = self.bridge_proj(projected)  # (B, N, 2048)
        return bridged


class Gemma4PI05(nn.Module):
    """π0.5 with Gemma 4 vision encoder.

    Wraps PI0Pytorch and replaces its SigLIP vision encoder
    with Gemma 4's vision encoder + projection bridge.
    """

    def __init__(
        self,
        gemma4_model: str = "google/gemma-4-E4B-it",
        pi05_checkpoint: str = None,
        device: str = "cuda",
    ):
        super().__init__()

        # Load π0.5 PyTorch model
        logger.info("Loading π0.5 PyTorch model...")
        pi05_config = Pi0Config(
            action_dim=32,
            action_horizon=15,
            max_token_len=200,
            dtype="bfloat16",
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            pytorch_compile_mode=None,  # disable compile for flexibility
        )
        self.pi05 = PI0Pytorch(pi05_config)

        if pi05_checkpoint:
            logger.info(f"Loading π0.5 weights from {pi05_checkpoint}")
            safetensors.torch.load_model(self.pi05, os.path.join(pi05_checkpoint, "model.safetensors"))

        self.pi05.to(device)
        self.pi05.eval()

        # Freeze all π0.5 parameters
        for p in self.pi05.parameters():
            p.requires_grad = False

        # Load Gemma 4 vision bridge
        self.vision_bridge = Gemma4VisionBridge(gemma4_model, device=device)
        self.vision_bridge.to(device)

        # Replace π0.5's image embedding function
        self._original_embed_image = self.pi05.paligemma_with_expert.embed_image
        self.pi05.paligemma_with_expert.embed_image = self._gemma4_embed_image

        logger.info("Gemma4VLA ready. Trainable: bridge_proj only.")

    def _gemma4_embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Override PI0Pytorch's embed_image with Gemma 4 vision encoder."""
        return self.vision_bridge.encode_images(pixel_values)

    @torch.no_grad()
    def sample_actions(self, device, observation, num_steps=10):
        """Generate actions using Gemma 4 vision + π0.5 action expert."""
        return self.pi05.sample_actions(device, observation, num_steps=num_steps)

    def get_trainable_parameters(self):
        """Only the bridge projection is trainable."""
        return self.vision_bridge.bridge_proj.parameters()
