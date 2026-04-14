"""Gemma 4 Vision-Language backbone.

Loads a pretrained Gemma 4 model and extracts hidden states
for the action expert to cross-attend to.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor


# Gemma 4 variant dimensions (hidden_size)
GEMMA4_DIMS = {
    "google/gemma-4-E2B": 1536,
    "google/gemma-4-E4B": 2560,
    "google/gemma-4-E2B-it": 1536,
    "google/gemma-4-E4B-it": 2560,
}


class Gemma4Backbone(nn.Module):
    """Frozen Gemma 4 backbone that produces vision-language features.

    Outputs hidden states from the last decoder layer, which the action
    expert cross-attends to for action generation.
    """

    def __init__(self, model_name: str = "google/gemma-4-E4B-it", device: str = "cuda", dtype=torch.bfloat16):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = GEMMA4_DIMS.get(model_name, 2560)
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, images: list, text: str, device: str = "cuda") -> torch.Tensor:
        """Encode images + text into hidden states.

        Args:
            images: List of PIL images (multi-view cameras).
            text: Language instruction.
            device: Target device.

        Returns:
            Hidden states tensor (1, seq_len, hidden_size).
        """
        # Build prompt with images
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
        ).to(device=device, dtype=self.dtype)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states is a tuple of (num_layers+1) tensors
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_size)
        return last_hidden.to(dtype=self.dtype)
