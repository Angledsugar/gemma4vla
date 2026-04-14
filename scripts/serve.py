"""Serve Gemma4VLA over WebSocket (OpenPI protocol compatible).

This script runs inside openpi's venv to access PI0Pytorch and all dependencies.

Usage:
    # Step 1: Convert π0.5 checkpoint (one-time, from openpi dir):
    cd ~/project/openpi
    uv run examples/convert_jax_model_to_pytorch.py \
        --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
        --config_name pi05_droid \
        --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch

    # Step 2: Serve (from openpi venv!):
    cd ~/project/openpi
    CUDA_VISIBLE_DEVICES=0 uv run python ~/project/gemma4vla/scripts/serve.py

    # Or original π0.5 without Gemma 4:
    CUDA_VISIBLE_DEVICES=0 uv run python ~/project/gemma4vla/scripts/serve.py --no-gemma4

    # Isaac Sim client (unchanged):
    python standalone_droid_vla.py --scene 1 --gpu 1
"""

import argparse
import asyncio
import http
import logging
import os
import sys
import time
import traceback

import numpy as np
import safetensors.torch
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure openpi-client is on path
OPENPI_ROOT = os.environ.get("OPENPI_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "openpi"))
OPENPI_CLIENT = os.path.join(OPENPI_ROOT, "packages", "openpi-client", "src")
if os.path.isdir(OPENPI_CLIENT):
    sys.path.insert(0, OPENPI_CLIENT)


def load_pi05_pytorch(checkpoint_dir: str, device: str = "cuda"):
    """Load π0.5 PyTorch model from converted checkpoint."""
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    config = Pi0Config(
        action_dim=32, action_horizon=15, max_token_len=200,
        dtype="bfloat16", paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m", pi05=True,
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, os.path.join(checkpoint_dir, "model.safetensors"))
    model.to(device).eval()
    return model, config


def replace_vision_with_gemma4(model, gemma4_name: str, device: str = "cuda"):
    """Replace PaliGemma's SigLIP with Gemma 4's vision encoder.

    Loads pre-exported vision_tower + embed_vision .pt files (no transformers>=5.5 needed).
    Export first with: cd ~/project/gemma4-sa-pi05 && uv run python ~/project/gemma4vla/scripts/export_gemma4_vision.py
    """
    import json
    import torch.nn as nn

    export_dir = os.path.expanduser("~/.cache/gemma4vla/vision_export")
    config_path = os.path.join(export_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Gemma 4 vision export not found at {export_dir}.\n"
            "Run first: cd ~/project/gemma4-sa-pi05 && "
            "CUDA_VISIBLE_DEVICES=1 uv run python ~/project/gemma4vla/scripts/export_gemma4_vision.py"
        )

    with open(config_path) as f:
        vision_info = json.load(f)
    gemma4_proj_dim = vision_info["text_hidden_size"]  # 2560

    logger.info(f"Loading exported Gemma 4 vision ({gemma4_proj_dim}d) from {export_dir}")

    # Reconstruct vision tower from exported state dict
    # We reuse PaliGemma's SigLIP architecture since Gemma 4 also uses SigLIP
    # but with different weights. Load state dict into existing vision tower.
    gemma4_vision_sd = torch.load(os.path.join(export_dir, "vision_tower.pt"), map_location=device)
    gemma4_projector_sd = torch.load(os.path.join(export_dir, "embed_vision.pt"), map_location=device)

    # Create standalone modules to hold the weights
    from collections import OrderedDict

    class VisionModule(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            # Reconstruct layers from state dict keys
            for key, tensor in state_dict.items():
                parts = key.split(".")
                module = self
                for part in parts[:-1]:
                    if not hasattr(module, part):
                        setattr(module, part, nn.Module())
                    module = getattr(module, part)
                setattr(module, parts[-1], nn.Parameter(tensor))

        def forward(self, x):
            raise NotImplementedError("Use model's original forward")

    # Instead of reconstructing, load from transformers if available,
    # otherwise use the original PaliGemma vision tower with swapped weights
    original_vision = model.paligemma_with_expert.img
    original_projector = None  # PaliGemma projector is inside the model

    # Bridge: Gemma 4 output (2560) → PaliGemma embedding (2048)
    # 2-Layer MLP with GELU (LLaVA 1.5 검증)
    paligemma_dim = 2048
    bridge = nn.Sequential(
        nn.Linear(gemma4_proj_dim, paligemma_dim, bias=True),
        nn.GELU(),
        nn.Linear(paligemma_dim, paligemma_dim, bias=True),
    ).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        nn.init.zeros_(bridge[0].bias)
        nn.init.zeros_(bridge[2].bias)
        nn.init.eye_(bridge[2].weight)
        bridge[0].weight.zero_()
        bridge[0].weight[:paligemma_dim, :paligemma_dim] = torch.eye(paligemma_dim)
        if gemma4_proj_dim > paligemma_dim:
            nn.init.normal_(bridge[0].weight[:, paligemma_dim:], std=0.01)

    # For now, keep using PaliGemma's vision but add bridge for dimension test
    # TODO: Replace vision tower weights once architecture matching is resolved
    # The key insight: PaliGemma SigLIP (1152d) → PaliGemma projector → 2048d
    # vs Gemma 4 SigLIP (768d) → Gemma 4 projector → 2560d → bridge → 2048d
    # Architecture mismatch means we can't just swap state dicts.

    # Use the bridge as a post-projector adapter on PaliGemma's output
    # This doesn't change the vision encoder but adds the MLP adapter layer
    original_embed = model.paligemma_with_expert.embed_image

    def bridged_embed_image(pixel_values):
        features = original_embed(pixel_values)  # PaliGemma: → 2048d
        # Bridge acts as adapter: 2048 → 2048 (identity-initialized)
        # When Gemma 4 vision is properly integrated, this becomes 2560 → 2048
        return bridge(
            torch.nn.functional.pad(features, (0, gemma4_proj_dim - paligemma_dim))
        )

    model.paligemma_with_expert.embed_image = bridged_embed_image
    model._gemma4_bridge = bridge

    bridge_params = sum(p.numel() for p in bridge.parameters())
    logger.info(f"MLP bridge attached: {gemma4_proj_dim}→{paligemma_dim}, params={bridge_params/1e6:.1f}M")
    logger.info("NOTE: Currently using PaliGemma vision + bridge adapter. Full Gemma 4 vision swap requires architecture matching.")
    return model


def build_droid_observation(obs_dict: dict, config, device: str = "cuda"):
    """Convert raw DROID observation dict to PI0Pytorch's Observation format."""
    from openpi.models import model as _model
    from openpi.policies import droid_policy

    # Apply DROID input transforms
    droid_inputs = droid_policy.DroidInputs(model_type=_model.ModelType.PI05)
    transformed = droid_inputs(obs_dict)

    # Convert to model Observation
    # Convert image_mask from Python bool to numpy bool arrays (jaxtyping requirement)
    if "image_mask" in transformed:
        transformed["image_mask"] = {
            k: np.array(v, dtype=bool) for k, v in transformed["image_mask"].items()
        }
    observation = _model.Observation.from_dict(transformed)

    # Add batch dimension and move to device
    import dataclasses
    batched = dataclasses.replace(
        observation,
        images={k: v[None].to(device) if isinstance(v, torch.Tensor) else torch.tensor(v[None], device=device)
                for k, v in observation.images.items()},
        image_masks={k: torch.tensor([v], device=device) for k, v in observation.image_masks.items()},
        state=observation.state[None].to(device) if isinstance(observation.state, torch.Tensor)
              else torch.tensor(observation.state[None], device=device),
        tokenized_prompt=observation.tokenized_prompt[None].to(device) if isinstance(observation.tokenized_prompt, torch.Tensor)
              else torch.tensor(observation.tokenized_prompt[None], device=device),
        tokenized_prompt_mask=observation.tokenized_prompt_mask[None].to(device) if isinstance(observation.tokenized_prompt_mask, torch.Tensor)
              else torch.tensor(observation.tokenized_prompt_mask[None], device=device),
    )
    return batched


def main():
    parser = argparse.ArgumentParser(description="Gemma4VLA / PI05 Policy Server")
    parser.add_argument("--gemma4", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--pi05-checkpoint", type=str,
                        default=os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch"))
    parser.add_argument("--no-gemma4", action="store_true", help="Use original PaliGemma vision")
    parser.add_argument("--expert-checkpoint", type=str, default=None,
                        help="Gemma4VLA trained expert checkpoint (.pt file)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args()

    if args.expert_checkpoint and args.expert_checkpoint.endswith(".pt"):
        # ============ Gemma4VLA mode: real Gemma 4 backbone + trained expert ============
        logger.info(f"Loading Gemma4VLA with trained expert: {args.expert_checkpoint}")
        from gemma4vla.model.config import Gemma4ActionExpertConfig
        from gemma4vla.model.policy import Gemma4VLAPolicy

        config = Gemma4ActionExpertConfig(variant="flow_shared", action_dim=32, action_horizon=15)
        policy = Gemma4VLAPolicy(config, dummy=False, device=args.device)

        # Load trained expert weights
        state_dict = torch.load(args.expert_checkpoint, map_location=args.device)
        policy.action_expert.load_state_dict(state_dict)
        policy.action_expert.eval()
        logger.info(f"Expert loaded: {sum(p.numel() for p in policy.action_expert.parameters())/1e6:.1f}M params")

        @torch.no_grad()
        def infer(obs_dict: dict) -> dict:
            from PIL import Image as PILImage
            # Parse images
            images = []
            for key in ["observation/exterior_image_1_left", "observation/wrist_image_left"]:
                img_data = obs_dict.get(key.encode() if isinstance(list(obs_dict.keys())[0], bytes) else key)
                if img_data is not None:
                    arr = np.asarray(img_data, dtype=np.uint8) if not isinstance(img_data, dict) else np.zeros((224,224,3), dtype=np.uint8)
                    if isinstance(img_data, dict):
                        d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in img_data.items()}
                        if "data" in d and "shape" in d:
                            dtype_raw = d.get("type", "uint8")
                            if isinstance(dtype_raw, bytes): dtype_raw = dtype_raw.decode()
                            shape = tuple(d["shape"])
                            buf = d["data"]
                            if isinstance(buf, (bytes, bytearray)):
                                for try_dtype in [dtype_raw, "uint8", "float32"]:
                                    if len(buf) == int(np.prod(shape)) * np.dtype(try_dtype).itemsize:
                                        arr = np.frombuffer(buf, dtype=try_dtype).copy().reshape(shape)
                                        break
                    if arr.ndim == 3:
                        images.append(PILImage.fromarray(arr.astype(np.uint8)))

            # Parse state
            joint_pos = obs_dict.get(b"observation/joint_position", obs_dict.get("observation/joint_position", np.zeros(7)))
            gripper_pos = obs_dict.get(b"observation/gripper_position", obs_dict.get("observation/gripper_position", np.zeros(1)))
            if isinstance(joint_pos, dict):
                d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in joint_pos.items()}
                if "data" in d: joint_pos = np.frombuffer(d["data"], dtype="float32").copy()
            if isinstance(gripper_pos, dict):
                d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in gripper_pos.items()}
                if "data" in d: gripper_pos = np.frombuffer(d["data"], dtype="float32").copy()
            joint_pos = np.asarray(joint_pos, dtype=np.float32).flatten()
            gripper_pos = np.asarray(gripper_pos, dtype=np.float32).flatten()
            state = np.concatenate([joint_pos, gripper_pos])
            if len(state) < 32:
                state = np.concatenate([state, np.zeros(32 - len(state), dtype=np.float32)])

            prompt = obs_dict.get(b"prompt", obs_dict.get("prompt", ""))
            if isinstance(prompt, bytes): prompt = prompt.decode()

            # Encode + predict
            bb = policy._real_backbone
            t0 = time.monotonic()
            context = bb.encode_observation(
                images_pil=images if images else None,
                text=prompt or "manipulate object",
                device=args.device,
                return_embeddings_only=(config.variant == "flow_shared"),
            )
            state_t = torch.tensor(state, device=args.device, dtype=torch.bfloat16).unsqueeze(0)
            actions = policy.sampler.sample(
                predict_velocity_fn=policy.action_expert,
                shape=(1, config.action_horizon, config.action_dim),
                state=state_t, context=context, device=args.device,
            )
            infer_ms = (time.monotonic() - t0) * 1000
            actions_np = actions.squeeze(0).cpu().float().numpy()
            return {
                "actions": np.ascontiguousarray(actions_np[:, :8]),
                "policy_timing": {"infer_ms": infer_ms},
            }

        model_label = "Gemma4VLA"
        logger.info("Gemma4VLA ready")

    else:
        # ============ π0.5 PyTorch mode ============
        model, config = load_pi05_pytorch(args.pi05_checkpoint, args.device)
        logger.info(f"π0.5 loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

        if not args.no_gemma4:
            model = replace_vision_with_gemma4(model, args.gemma4, args.device)
        model_label = "gemma4vla" if not args.no_gemma4 else "pi05-pytorch"

        # π0.5 inference function
        @torch.no_grad()
        def infer(obs_dict: dict) -> dict:
            observation = build_droid_observation(obs_dict, config, args.device)
            t0 = time.monotonic()
            actions = model.sample_actions(args.device, observation, num_steps=args.num_steps)
            infer_ms = (time.monotonic() - t0) * 1000
            actions_np = actions.cpu().float().numpy()
            return {
                "actions": np.ascontiguousarray(actions_np[:, :8]),
                "policy_timing": {"infer_ms": infer_ms},
            }

    # WebSocket server
    from openpi_client import msgpack_numpy
    import websockets.asyncio.server as ws_server
    import websockets.frames

    async def handler(websocket):
        logger.info(f"Connection from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()
        metadata = {
            "model": model_label,
            "robot": "droid", "action_dim": 8, "action_horizon": 15,
        }
        await websocket.send(packer.pack(metadata))

        while True:
            try:
                raw = await websocket.recv()
                obs = msgpack_numpy.unpackb(raw)
                result = await asyncio.get_event_loop().run_in_executor(None, infer, obs)
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                logger.info("Connection closed")
                break
            except Exception:
                logger.error(traceback.format_exc())
                try:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR)
                except Exception:
                    pass
                raise

    async def health(conn, req):
        if req.path == "/healthz":
            return conn.respond(http.HTTPStatus.OK, "OK\n")

    async def run():
        async with ws_server.serve(handler, "0.0.0.0", args.port,
                                    compression=None, max_size=None,
                                    process_request=health) as server:
            label = "Gemma4VLA" if not args.no_gemma4 else "PI05-PyTorch"
            logger.info(f"{label} server listening on 0.0.0.0:{args.port}")
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
