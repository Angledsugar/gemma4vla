"""Export Gemma 4 vision encoder + projector to standalone PyTorch files.

Run this once from the gemma4-sa-pi05 venv (which has transformers>=5.5):
    cd ~/project/gemma4-sa-pi05
    CUDA_VISIBLE_DEVICES=1 uv run python ~/project/gemma4vla/scripts/export_gemma4_vision.py

This saves vision_tower and embed_vision as separate .pt files that can be
loaded by openpi's venv without needing transformers>=5.5.
"""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output_dir", default=os.path.expanduser("~/.cache/gemma4vla/vision_export"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoConfig

    print(f"Loading {args.model}...")
    config = AutoConfig.from_pretrained(args.model)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save vision tower
    print("Saving vision_tower...")
    torch.save(model.model.vision_tower.state_dict(), os.path.join(args.output_dir, "vision_tower.pt"))

    # Save vision embedder (projector)
    print("Saving embed_vision...")
    torch.save(model.model.embed_vision.state_dict(), os.path.join(args.output_dir, "embed_vision.pt"))

    # Save config info
    import json
    info = {
        "model": args.model,
        "vision_hidden_size": config.vision_config.hidden_size,
        "text_hidden_size": config.text_config.hidden_size,
        "vision_config": {k: v for k, v in config.vision_config.to_dict().items()
                         if isinstance(v, (int, float, str, bool, list))},
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Exported to {args.output_dir}")
    print(f"  vision_hidden_size: {info['vision_hidden_size']}")
    print(f"  text_hidden_size: {info['text_hidden_size']}")
    print(f"  Files: vision_tower.pt, embed_vision.pt, config.json")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
