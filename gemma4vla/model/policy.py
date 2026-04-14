"""Full Gemma 4 VLA policy: backbone + projector + action expert + flow matching.

This is the top-level model that orchestrates:
1. Vision encoding (images → patches → backbone tokens)
2. Language encoding (text → token embeddings)
3. Backbone forward (all tokens → hidden states)
4. Action expert (hidden states + noised actions + timestep → velocity)
5. Flow matching (training loss / inference sampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_expert import create_action_expert, sinusoidal_embedding
from .config import Gemma4ActionExpertConfig
from .flow_matching import FlowMatchingLoss, FlowMatchingSampler
from .projector import DummyVisionEncoder, VisionProjector


class DummyBackbone(nn.Module):
    """Placeholder Gemma 4 backbone for shape testing.

    TODO: Replace with actual Gemma 4 decoder.
    Simulates: image tokens + language tokens → hidden states (B, S, 2560)
    """

    def __init__(self, hidden_size: int = 2560, num_layers: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        # Minimal transformer for testing
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeds: (B, S, hidden_size) — concatenated vision + language tokens
        Returns:
            (B, S, hidden_size) — backbone hidden states
        """
        return self.encoder(token_embeds)


class Gemma4VLAPolicy(nn.Module):
    """Complete VLA policy for Gemma 4 + Action Expert.

    Two modes:
        dummy=True:  DummyBackbone for shape testing (no GPU needed)
        dummy=False: Real Gemma 4 E4B backbone (requires GPU + ~8GB VRAM)
    """

    def __init__(self, config: Gemma4ActionExpertConfig, dummy: bool = True, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.use_dummy = dummy
        B_dim = config.backbone.hidden_size

        if dummy:
            # Dummy components for testing
            self.vision_encoder = DummyVisionEncoder(image_size=config.image_size, vision_dim=1152)
            self.vision_projector = VisionProjector(vision_dim=1152, backbone_dim=B_dim)
            self.lang_embed = nn.Embedding(config.backbone.vocab_size, B_dim)
            self.backbone = DummyBackbone(hidden_size=B_dim, num_layers=4)
        else:
            # Real Gemma 4 backbone
            from .backbone_gemma4 import Gemma4RealBackbone
            self._real_backbone = Gemma4RealBackbone(
                model_name="google/gemma-4-E4B-it",
                device=device,
                dtype=config.torch_dtype,
                freeze=True,
            )

        # Action Expert (main contribution — always real)
        self.action_expert = create_action_expert(config)

        # Connect backbone layers to shared expert (π0.6 style)
        if not dummy and config.variant == "flow_shared":
            self._connect_backbone_layers(device)

        # Flow matching
        self.flow_loss = FlowMatchingLoss()
        self.sampler = FlowMatchingSampler(num_steps=config.flow_matching_steps)

    def _connect_backbone_layers(self, device: str = "cuda"):
        """Wire backbone decoder layers into the shared-attention expert.

        The expert's shared attention uses backbone layer's Q,K,V,O projections
        and norms directly (frozen, not copied). Only expert layers are trainable.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get Gemma 4's decoder layers
        backbone_model = self._real_backbone.model
        # Gemma4ForConditionalGeneration → model → language_model → layers
        decoder_layers = backbone_model.model.language_model.layers

        n_backbone = len(decoder_layers)
        n_expert = len(self.action_expert.expert_layers)

        if n_backbone != n_expert:
            logger.warning(
                f"Layer count mismatch: backbone={n_backbone}, expert={n_expert}. "
                f"Using first {min(n_backbone, n_expert)} layers."
            )

        # Pass backbone layers to expert (reference, not copy)
        from .action_expert import FlowSharedExpert
        if isinstance(self.action_expert, FlowSharedExpert):
            self.action_expert.set_backbone_layers(decoder_layers)
            logger.info(
                f"Connected {min(n_backbone, n_expert)} backbone layers to shared expert. "
                f"Backbone layers are FROZEN (not in expert's parameters)."
            )

            # Move expert to same device/dtype as backbone
            dtype = self.config.torch_dtype
            self.action_expert.to(device=device, dtype=dtype)
            logger.info(f"Expert on {device}, dtype={dtype}")

    def encode_observation(
        self,
        images: list[torch.Tensor],       # list of (B, 3, H, W)
        lang_tokens: torch.Tensor = None,  # (B, L) int
    ) -> torch.Tensor:
        """Encode images + language into backbone hidden states.

        Returns: (B, S, backbone_dim=2560)
        """
        if not self.use_dummy:
            # Shared attention mode: return embeddings (expert will process layers)
            # Cross-attention mode: return full decoder hidden states
            embeddings_only = (self.config.variant == "flow_shared"
                               and hasattr(self.action_expert, '_backbone_layers')
                               and self.action_expert._backbone_layers is not None)

            if images and hasattr(images[0], 'mode'):  # PIL Image
                return self._real_backbone.encode_observation(
                    images_pil=images,
                    text="",
                    device="cuda",
                    return_embeddings_only=embeddings_only,
                )
            else:
                return self._real_backbone.encode_observation(
                    preprocessed=images[0] if isinstance(images[0], dict) else None,
                    device="cuda",
                    return_embeddings_only=embeddings_only,
                )

        # Dummy path
        all_tokens = []
        for img in images:
            patches = self.vision_encoder(img)
            projected = self.vision_projector(patches)
            all_tokens.append(projected)
        if lang_tokens is not None:
            lang_emb = self.lang_embed(lang_tokens)
            all_tokens.append(lang_emb)
        tokens = torch.cat(all_tokens, dim=1)
        return self.backbone(tokens)

    def compute_loss(
        self,
        images: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        state: torch.Tensor,         # (B, action_dim)
        target_actions: torch.Tensor, # (B, H, action_dim)
    ) -> torch.Tensor:
        """Training forward: compute flow matching loss."""
        context = self.encode_observation(images, lang_tokens)
        return self.flow_loss(
            predict_velocity_fn=self.action_expert,
            target_actions=target_actions,
            state=state,
            context=context,
        )

    @torch.no_grad()
    def predict_actions(
        self,
        images: list[torch.Tensor],
        lang_tokens: torch.Tensor = None,
        state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference: generate action chunk via flow matching ODE.

        Returns: (B, H, action_dim)
        """
        context = self.encode_observation(images, lang_tokens)
        B = context.shape[0]
        shape = (B, self.config.action_horizon, self.config.action_dim)

        if state is None:
            state = torch.zeros(B, self.config.action_dim, device=context.device, dtype=context.dtype)

        return self.sampler.sample(
            predict_velocity_fn=self.action_expert,
            shape=shape,
            state=state,
            context=context,
            device=context.device,
        )

    def get_action_expert_params(self):
        """Parameters for training only the action expert."""
        return self.action_expert.parameters()

    def get_all_trainable_params(self):
        """All trainable parameters (expert + projector if dummy)."""
        params = list(self.action_expert.parameters())
        if self.use_dummy:
            params += list(self.vision_projector.parameters())
        return params

    def param_summary(self) -> dict:
        """Count parameters by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        if self.use_dummy:
            return {
                "vision_encoder": count(self.vision_encoder),
                "vision_projector": count(self.vision_projector),
                "backbone": count(self.backbone),
                "action_expert": count(self.action_expert),
                "total": count(self),
            }
        else:
            return {
                "gemma4_backbone": count(self._real_backbone.model),
                "action_expert": count(self.action_expert),
                "total": count(self._real_backbone.model) + count(self.action_expert),
            }
