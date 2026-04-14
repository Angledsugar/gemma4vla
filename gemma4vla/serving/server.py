"""WebSocket policy server — OpenPI client protocol compatible.

Serves the Gemma4-VLA model over WebSocket so existing Isaac Sim
scripts (standalone_droid_vla.py) can connect without modification.

Expected observation format (from client):
    {
        "observation/exterior_image_1_left": np.ndarray (H, W, 3),
        "observation/wrist_image_left": np.ndarray (H, W, 3),
        "observation/joint_position": np.ndarray (7,),
        "observation/gripper_position": np.ndarray (1,),
        "prompt": str,
    }

Response format:
    {
        "actions": np.ndarray (action_horizon, 8),
        "policy_timing": {"infer_ms": float},
    }
"""

import asyncio
import http
import logging
import time
import traceback

import numpy as np
import torch
import websockets.asyncio.server as ws_server
import websockets.frames
from msgpack_numpy import Packer, unpackb
from PIL import Image

from ..model.vla import Gemma4VLA, Gemma4VLAConfig

logger = logging.getLogger(__name__)


class Gemma4VLAServer:
    def __init__(
        self,
        config: Gemma4VLAConfig,
        host: str = "0.0.0.0",
        port: int = 8000,
        device: str = "cuda",
    ):
        self.host = host
        self.port = port
        self.device = device

        logger.info("Loading Gemma4-VLA model...")
        self.model = Gemma4VLA(config)
        self.model.load_backbone(device=device)
        self.model.action_expert.to(device=device, dtype=config.dtype)
        self.model.action_expert.eval()
        logger.info(
            f"Model loaded. Trainable params: {self.model.num_trainable_params()/1e6:.1f}M, "
            f"Backbone: {config.backbone_model}"
        )

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        """Convert msgpack-deserialized data to numpy array.

        msgpack-numpy encodes arrays as dicts with bytes keys:
        {b'nd': True, b'type': b'<f4', b'kind': b'', b'shape': (7,), b'data': bytes}
        """
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, dict):
            # Normalize keys: bytes → str
            d = {(k.decode() if isinstance(k, bytes) else k): v for k, v in data.items()}
            if "data" in d and "shape" in d:
                dtype_raw = d.get("type", "float32")
                if isinstance(dtype_raw, bytes):
                    dtype_raw = dtype_raw.decode()
                shape = tuple(d["shape"])
                buf = d["data"]
                if isinstance(buf, (bytes, bytearray)):
                    expected_bytes = int(np.prod(shape)) * np.dtype(dtype_raw).itemsize
                    if len(buf) != expected_bytes:
                        # dtype/shape mismatch — infer dtype from buffer size
                        for try_dtype in [dtype_raw, "uint8", "float32", "float64"]:
                            if len(buf) == int(np.prod(shape)) * np.dtype(try_dtype).itemsize:
                                dtype_raw = try_dtype
                                break
                    return np.frombuffer(buf, dtype=dtype_raw).copy().reshape(shape)
        if isinstance(data, (list, tuple)):
            return np.array(data)
        return np.atleast_1d(np.asarray(data))

    @staticmethod
    def _normalize_obs_keys(obs: dict) -> dict:
        """Normalize observation dict keys from bytes to str."""
        return {(k.decode() if isinstance(k, bytes) else k): v for k, v in obs.items()}

    def _obs_to_inputs(self, obs: dict):
        """Convert OpenPI observation dict to model inputs."""
        obs = self._normalize_obs_keys(obs)

        # Images → PIL
        images = []
        for key in ["observation/exterior_image_1_left", "observation/wrist_image_left"]:
            img_data = obs.get(key)
            if img_data is not None:
                img_arr = self._to_numpy(img_data)
                if img_arr is not None and img_arr.ndim == 3:
                    images.append(Image.fromarray(img_arr.astype(np.uint8)))

        # Proprioception state
        joint_pos = self._to_numpy(obs.get("observation/joint_position", np.zeros(7)))
        gripper_pos = self._to_numpy(obs.get("observation/gripper_position", np.zeros(1)))
        state = np.concatenate([joint_pos.flatten(), gripper_pos.flatten()]).astype(np.float32)

        # Instruction
        prompt = obs.get("prompt", "")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        return images, prompt, torch.tensor(state)

    def infer(self, obs: dict) -> dict:
        images, prompt, state = self._obs_to_inputs(obs)

        t0 = time.monotonic()
        actions = self.model.predict_actions(
            images=images,
            instruction=prompt,
            state=state,
            device=self.device,
        )
        infer_ms = (time.monotonic() - t0) * 1000

        actions_np = actions.cpu().float().numpy()
        # Ensure (horizon, action_dim) shape
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)
        elif actions_np.ndim == 0:
            actions_np = actions_np.reshape(1, 1)

        return {
            "actions": np.ascontiguousarray(actions_np),
            "policy_timing": {"infer_ms": infer_ms},
        }

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with ws_server.serve(
            self._handler, self.host, self.port,
            compression=None, max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"Gemma4-VLA server listening on {self.host}:{self.port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address}")
        packer = Packer()

        # Send metadata (OpenPI protocol)
        metadata = {
            "model": f"gemma4vla ({self.model.config.backbone_model})",
            "robot": "droid",
            "action_dim": self.model.config.action_dim,
            "action_horizon": self.model.config.action_horizon,
        }
        await websocket.send(packer.pack(metadata))

        while True:
            try:
                raw = await websocket.recv()
                obs = unpackb(raw)
                result = await asyncio.get_event_loop().run_in_executor(None, self.infer, obs)
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                logger.info(f"Connection closed: {websocket.remote_address}")
                break
            except Exception:
                tb = traceback.format_exc()
                logger.error(tb)
                try:
                    await websocket.send(tb)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                except Exception:
                    pass
                raise

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None
