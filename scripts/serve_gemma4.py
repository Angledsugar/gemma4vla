"""Serve trained Gemma4VLA model over WebSocket (OpenPI protocol compatible).

Usage:
    # Serve on GPU 0 (Isaac Sim on GPU 1):
    cd ~/project/gemma4vla
    CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_gemma4.py \
        --checkpoint checkpoints/droid/expert_step1000.pt

    # Isaac Sim client (unchanged):
    cd ~/project/isaacsim-raise
    CUDA_VISIBLE_DEVICES=1 python ../raise-vla/scripts/standalone/standalone_droid_vla.py \
        --scene 1 --debug
"""

import argparse
import asyncio
import http
import logging
import os
import time
import traceback

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Gemma4VLA Policy Server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained action expert checkpoint (.pt)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=5, help="Flow matching ODE steps")
    args = parser.parse_args()

    # Load model
    from gemma4vla.model.config import Gemma4ActionExpertConfig
    from gemma4vla.model.policy import Gemma4VLAPolicy
    from gemma4vla.model.flow_matching import FlowMatchingSampler

    config = Gemma4ActionExpertConfig(
        variant="flow_shared",
        action_dim=32,
        action_horizon=15,
        flow_matching_steps=args.num_steps,
    )

    logger.info("Loading Gemma 4 backbone + action expert...")
    policy = Gemma4VLAPolicy(config, dummy=False, device=args.device)

    # Load trained checkpoint
    state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    policy.action_expert.load_state_dict(state_dict)
    policy.action_expert.to(args.device, dtype=config.torch_dtype)
    policy.action_expert.eval()

    expert_params = sum(p.numel() for p in policy.action_expert.parameters())
    logger.info(f"Expert loaded: {expert_params/1e6:.1f}M params from {args.checkpoint}")

    sampler = FlowMatchingSampler(num_steps=args.num_steps)

    # Inference function
    @torch.no_grad()
    def infer(obs_dict: dict) -> dict:
        t0 = time.monotonic()

        # Extract images
        from PIL import Image
        images_pil = []
        for key in ["observation/exterior_image_1_left", "observation/wrist_image_left"]:
            if key in obs_dict:
                img_data = obs_dict[key]
                if isinstance(img_data, dict):
                    # msgpack-numpy dict format
                    d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in img_data.items()}
                    dtype = d.get("type", "uint8")
                    if isinstance(dtype, bytes):
                        dtype = dtype.decode()
                    img_data = np.frombuffer(d["data"], dtype=dtype).copy().reshape(tuple(d["shape"]))
                if isinstance(img_data, np.ndarray):
                    images_pil.append(Image.fromarray(img_data[:, :, :3].astype(np.uint8)))

        if not images_pil:
            logger.warning("No images in observation, using blank")
            images_pil = [Image.new("RGB", (224, 224))]

        # Extract state (joint_position + gripper)
        joint_pos = obs_dict.get("observation/joint_position", np.zeros(7, dtype=np.float32))
        gripper_pos = obs_dict.get("observation/gripper_position", np.zeros(1, dtype=np.float32))
        if isinstance(joint_pos, dict):
            d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in joint_pos.items()}
            joint_pos = np.frombuffer(d["data"], dtype="float32").copy()
        if isinstance(gripper_pos, dict):
            d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in gripper_pos.items()}
            gripper_pos = np.frombuffer(d["data"], dtype="float32").copy()

        state_np = np.concatenate([
            np.array(joint_pos, dtype=np.float32),
            np.array(gripper_pos, dtype=np.float32),
        ])
        # Pad to 32 dims
        if len(state_np) < 32:
            state_np = np.concatenate([state_np, np.zeros(32 - len(state_np), dtype=np.float32)])

        # Extract prompt
        prompt = obs_dict.get("prompt", "manipulate object")
        if isinstance(prompt, bytes):
            prompt = prompt.decode()

        # Encode observation with Gemma 4 backbone
        context = policy._real_backbone.encode_observation(
            images_pil=images_pil,
            text=prompt,
            device=args.device,
            return_embeddings_only=(config.variant == "flow_shared"),
        )

        state_tensor = torch.tensor(state_np, dtype=config.torch_dtype, device=args.device).unsqueeze(0)

        # Flow matching sampling
        B = 1
        shape = (B, config.action_horizon, config.action_dim)
        actions = sampler.sample(
            predict_velocity_fn=policy.action_expert,
            shape=shape,
            state=state_tensor,
            context=context,
            device=args.device,
        )

        infer_ms = (time.monotonic() - t0) * 1000

        # Extract first 8 dims (7 joints + 1 gripper) from 32-dim internal
        actions_np = actions[0].cpu().float().numpy()[:, :8]  # (H, 8)

        logger.info(
            f"Inferred: shape={actions_np.shape}, "
            f"arm_range=[{actions_np[:,:7].min():.3f}, {actions_np[:,:7].max():.3f}], "
            f"grip=[{actions_np[:,7].min():.2f}, {actions_np[:,7].max():.2f}], "
            f"time={infer_ms:.0f}ms"
        )

        return {
            "actions": np.ascontiguousarray(actions_np),
            "policy_timing": {"infer_ms": infer_ms},
        }

    # WebSocket server (OpenPI protocol)
    import msgpack
    import struct

    try:
        # Try openpi_client's msgpack_numpy first
        import sys
        openpi_client_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "openpi", "packages", "openpi-client", "src"
        )
        if os.path.isdir(openpi_client_path):
            sys.path.insert(0, openpi_client_path)
        from openpi_client import msgpack_numpy
        pack_fn = msgpack_numpy.Packer().pack
        unpack_fn = msgpack_numpy.unpackb
        logger.info("Using openpi_client msgpack_numpy")
    except ImportError:
        # Fallback: plain msgpack with numpy handling
        def _encode_numpy(obj):
            if isinstance(obj, np.ndarray):
                return {
                    b"data": obj.tobytes(),
                    b"shape": list(obj.shape),
                    b"type": str(obj.dtype),
                }
            return obj

        def _decode_numpy(obj):
            if isinstance(obj, dict) and b"data" in obj:
                dtype = obj.get(b"type", b"float32")
                if isinstance(dtype, bytes):
                    dtype = dtype.decode()
                return np.frombuffer(obj[b"data"], dtype=dtype).copy().reshape(
                    tuple(obj[b"shape"])
                )
            return obj

        packer = msgpack.Packer(default=_encode_numpy)
        pack_fn = packer.pack
        unpack_fn = lambda raw: msgpack.unpackb(raw, object_hook=_decode_numpy, raw=True)
        logger.info("Using plain msgpack with numpy fallback")

    import websockets.asyncio.server as ws_server
    import websockets.frames

    async def handler(websocket):
        logger.info(f"Connection from {websocket.remote_address}")
        metadata = {
            "model": "gemma4vla",
            "robot": "droid",
            "action_dim": 8,
            "action_horizon": 15,
        }
        await websocket.send(pack_fn(metadata))

        while True:
            try:
                raw = await websocket.recv()
                obs = unpack_fn(raw)
                result = await asyncio.get_event_loop().run_in_executor(None, infer, obs)
                await websocket.send(pack_fn(result))
            except Exception as e:
                if "ConnectionClosed" in type(e).__name__:
                    logger.info("Connection closed")
                    break
                logger.error(traceback.format_exc())
                try:
                    await websocket.send(traceback.format_exc().encode())
                except Exception:
                    pass
                break

    async def health(conn, req):
        if req.path == "/healthz":
            return conn.respond(http.HTTPStatus.OK, "OK\n")

    async def run():
        async with ws_server.serve(
            handler, "0.0.0.0", args.port,
            compression=None, max_size=None,
            process_request=health,
        ) as server:
            logger.info(f"Gemma4VLA server listening on 0.0.0.0:{args.port}")
            logger.info(f"  Checkpoint: {args.checkpoint}")
            logger.info(f"  Flow steps: {args.num_steps}")
            logger.info(f"  Device: {args.device}")
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
