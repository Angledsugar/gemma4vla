"""Real Gemma 4 backbone implementation.

Loads google/gemma-4-E4B-it and extracts:
  1. Vision tower: images → 280 visual tokens (768-dim → projected to 2560)
  2. Language model: text tokens → hidden states (2560-dim)
  3. Full decoder: all tokens → contextualized hidden states (2560-dim)

The backbone is frozen during training (Knowledge Insulation uses
a separate discrete loss to update it).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Gemma4RealBackbone(nn.Module):
    """Real Gemma 4 E4B backbone with vision + language encoding.

    Replaces DummyBackbone and DummyVisionEncoder + VisionProjector.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading Gemma 4 backbone: {model_name}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=dtype, device_map=device,
        )
        self.hidden_size = self.model.config.text_config.hidden_size  # 2560
        self.vision_tokens_per_image = self.model.config.vision_soft_tokens_per_image  # 280
        self.dtype = dtype

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            logger.info("Backbone frozen")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Backbone loaded: {total_params/1e9:.2f}B params, hidden={self.hidden_size}")

    @torch.no_grad()
    def encode_observation(
        self,
        images_pil: list = None,
        text: str = None,
        preprocessed: dict = None,
        device: str = "cuda",
        return_embeddings_only: bool = False,
    ) -> torch.Tensor:
        """Encode images + text.

        Two modes:
          return_embeddings_only=False (default): Full decoder forward → last hidden states.
            Use when action expert uses cross-attention (needs contextualized features).

          return_embeddings_only=True: Only compute input embeddings (vision + text).
            Use when action expert uses shared attention (processes layers jointly).

        Returns:
            (B, S_total, 2560) hidden states or embeddings
        """
        if preprocessed is None:
            preprocessed = self.preprocess(images_pil or [], text or "")

        inputs = {k: v.to(device) for k, v in preprocessed.items() if isinstance(v, torch.Tensor)}

        if return_embeddings_only:
            # Only compute input embeddings (no decoder forward)
            # This is used for shared-attention mode where the expert
            # processes backbone layers jointly
            return self._compute_input_embeddings(inputs)

        # Full forward → last hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1].to(dtype=self.dtype)

    @torch.no_grad()
    def _compute_input_embeddings(self, inputs: dict) -> torch.Tensor:
        """Compute input embeddings without running decoder layers.

        Returns the token embeddings that would normally be fed into
        the first decoder layer. For shared attention, the expert
        will process these through backbone layers jointly.
        """
        model = self.model.model

        # Get input_ids and handle image tokens
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        image_position_ids = inputs.get("image_position_ids")
        mm_token_type_ids = inputs.get("mm_token_type_ids")

        # Text embeddings
        text_embeds = model.language_model.embed_tokens(input_ids)

        # If we have images, merge visual tokens
        if pixel_values is not None:
            # Encode images
            vision_out = model.vision_tower(
                pixel_values=pixel_values.to(self.dtype),
                pixel_position_ids=image_position_ids,
            )
            image_features = vision_out.last_hidden_state
            visual_embeds = model.embed_vision(image_features)

            # Replace image placeholder tokens with visual embeddings
            if mm_token_type_ids is not None:
                # mm_token_type_ids: 1 where image tokens should go
                image_mask = mm_token_type_ids == 1  # (B, S)
                B, S, D = text_embeds.shape
                combined = text_embeds.clone()
                # Scatter visual embeddings into image positions
                img_positions = image_mask.nonzero(as_tuple=False)
                if img_positions.shape[0] > 0 and visual_embeds.shape[0] > 0:
                    n_vis = visual_embeds.shape[1]
                    for b in range(B):
                        pos = (img_positions[:, 0] == b).nonzero(as_tuple=False).squeeze(-1)
                        n = min(len(pos), n_vis)
                        for i in range(n):
                            combined[b, img_positions[pos[i], 1]] = visual_embeds[b, i]
                return combined.to(dtype=self.dtype)

        return text_embeds.to(dtype=self.dtype)

    def preprocess(self, pil_images: list, text: str = "") -> dict:
        """Preprocess PIL images + text using Gemma 4's chat template.

        Returns dict with pixel_values, image_position_ids, input_ids, etc.
        """
        # Build chat message with image placeholders
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text or "describe"})

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        return self.processor(
            text=prompt,
            images=pil_images if pil_images else None,
            return_tensors="pt",
        )
