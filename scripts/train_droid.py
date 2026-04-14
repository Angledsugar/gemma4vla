"""Train Gemma4VLA on DROID 100 dataset (LeRobot v3).

Training-Time RTC (arXiv:2512.05964):
  Conditions on action prefix from previous chunk during training.
  Per-token timestep via AdaLN-zero: prefix τ=1, postfix τ~U(0,1).
  Loss computed only on postfix tokens.

Usage:
    # Standard training
    CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py --steps 1000

    # Train directly from Hugging Face Hub (auto-download + cache)
    CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py \
        --repo_id lerobot/droid_100 --steps 1000

    # With Training-Time RTC
    CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py --steps 10000 --rtc

    # Full pipeline (RTC + Knowledge Insulation)
    CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py --steps 10000 --rtc --knowledge_insulation
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Normalization statistics
# ============================================================================

class ActionStateNormalizer:
    """Per-dimension normalization for actions and states.

    DROID actions are already in [-1, 1] (normalized joint velocity),
    but we compute dataset statistics for proper z-score normalization
    which helps flow matching converge faster.
    """

    def __init__(self, stats_path: str = None):
        self.action_mean = None
        self.action_std = None
        self.state_mean = None
        self.state_std = None

        if stats_path and os.path.isfile(stats_path):
            self.load(stats_path)

    def compute_from_dataset(self, dataset, max_samples: int = 10000):
        """Compute normalization stats from dataset."""
        n = min(len(dataset), max_samples)
        indices = random.sample(range(len(dataset)), n)

        all_actions = []
        all_states = []
        for idx in indices:
            item = dataset[idx]
            all_actions.append(item["actions"].numpy())
            all_states.append(item["state"].numpy())

        actions = np.stack(all_actions)  # (N, H, D)
        states = np.stack(all_states)    # (N, D)

        # Per-dim stats (flatten horizon for actions)
        actions_flat = actions.reshape(-1, actions.shape[-1])
        self.action_mean = torch.tensor(actions_flat.mean(axis=0), dtype=torch.float32)
        self.action_std = torch.tensor(actions_flat.std(axis=0).clip(min=1e-6), dtype=torch.float32)
        self.state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
        self.state_std = torch.tensor(states.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        logger.info(
            f"Norm stats computed from {n} samples: "
            f"action_mean=[{self.action_mean[:8].tolist()}], "
            f"action_std=[{self.action_std[:8].tolist()}]"
        )

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None:
            return actions
        mean = self.action_mean.to(actions.device, dtype=actions.dtype)
        std = self.action_std.to(actions.device, dtype=actions.dtype)
        return (actions - mean) / std

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None:
            return actions
        mean = self.action_mean.to(actions.device, dtype=actions.dtype)
        std = self.action_std.to(actions.device, dtype=actions.dtype)
        return actions * std + mean

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_mean is None:
            return state
        mean = self.state_mean.to(state.device, dtype=state.dtype)
        std = self.state_std.to(state.device, dtype=state.dtype)
        return (state - mean) / std

    def save(self, path: str):
        torch.save({
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
        }, path)
        logger.info(f"Norm stats saved: {path}")

    def load(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.action_mean = data["action_mean"]
        self.action_std = data["action_std"]
        self.state_mean = data["state_mean"]
        self.state_std = data["state_std"]
        logger.info(f"Norm stats loaded: {path}")


# ============================================================================
# Dataset
# ============================================================================

class DroidV3Dataset(Dataset):
    """Load DROID LeRobot v3 dataset with video frame extraction.

    Reads parquet for state/action, decodes video frames for images.
    """

    def __init__(
        self,
        data_dir: str,
        action_horizon: int = 15,
        image_size: int = 448,
        cameras: list[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.action_horizon = action_horizon
        self.image_size = image_size

        # Load info
        with open(self.data_dir / "meta" / "info.json") as f:
            self.info = json.load(f)

        self.fps = self.info["fps"]
        self.state_dim = self.info["features"]["observation.state"]["shape"][0]
        self.action_dim = self.info["features"]["action"]["shape"][0]

        # Camera keys
        if cameras is None:
            self.cameras = [k for k, v in self.info["features"].items()
                           if v["dtype"] in ("video", "image") and "images" in k]
        else:
            self.cameras = cameras

        # Load parquet
        parquet_files = sorted((self.data_dir / "data").glob("**/*.parquet"))
        tables = [pq.read_table(str(f)) for f in parquet_files]
        import pyarrow as pa
        self.table = pa.concat_tables(tables).to_pandas()

        # Build valid indices (full action chunk within same episode)
        episodes = self.table["episode_index"].values
        self.valid_indices = []
        for i in range(len(episodes) - action_horizon):
            if episodes[i] == episodes[i + action_horizon - 1]:
                self.valid_indices.append(i)

        # Video reader cache
        self._video_cache = {}

        # Load tasks
        try:
            tasks_df = pq.read_table(str(self.data_dir / "meta" / "tasks.parquet")).to_pandas()
            self.tasks = {row["task_index"]: row["task"] for _, row in tasks_df.iterrows()}
        except Exception:
            self.tasks = {0: ""}

        logger.info(
            f"DROID dataset: {len(self.valid_indices)} samples, "
            f"{self.info['total_episodes']} episodes, "
            f"state={self.state_dim}, action={self.action_dim}, "
            f"cameras={self.cameras}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def _read_video_frame(self, camera: str, episode_idx: int, frame_idx: int) -> np.ndarray:
        """Read a single frame from video file."""
        try:
            import av
        except ImportError:
            # Fallback: return random image (for testing without av)
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

        # Find video file
        video_template = self.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        chunk_size = self.info.get("chunks_size", 1000)
        chunk_idx = episode_idx // chunk_size
        file_idx = chunk_idx  # typically same

        video_path = self.data_dir / video_template.format(
            video_key=camera,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )

        if not video_path.exists():
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

        cache_key = str(video_path)
        if cache_key not in self._video_cache:
            container = av.open(str(video_path))
            self._video_cache[cache_key] = container

        container = self._video_cache[cache_key]
        stream = container.streams.video[0]

        # Seek to frame
        try:
            container.seek(frame_idx, stream=stream)
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="rgb24")
                # Resize
                pil_img = Image.fromarray(img).resize((self.image_size, self.image_size), Image.BILINEAR)
                return np.array(pil_img)
        except Exception:
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        frame_idx = self.valid_indices[idx]
        row = self.table.iloc[frame_idx]

        # State
        state = np.array(row["observation.state"], dtype=np.float32)

        # Action chunk
        actions = []
        for i in range(self.action_horizon):
            a = np.array(self.table.iloc[frame_idx + i]["action"], dtype=np.float32)
            actions.append(a)
        actions = np.stack(actions)  # (H, 7)

        # Pad action to 32 dims (π0.5 internal format)
        if actions.shape[-1] < 32:
            pad = np.zeros((self.action_horizon, 32 - actions.shape[-1]), dtype=np.float32)
            actions = np.concatenate([actions, pad], axis=-1)

        # Pad state similarly
        if len(state) < 32:
            state = np.concatenate([state, np.zeros(32 - len(state), dtype=np.float32)])

        # Images (PIL for Gemma 4 processor)
        episode_idx = int(row["episode_index"])
        global_frame = int(row["frame_index"])
        images = []
        for cam in self.cameras[:2]:  # Use first 2 cameras
            img = self._read_video_frame(cam, episode_idx, global_frame)
            images.append(Image.fromarray(img))

        # Task
        task_idx = int(row.get("task_index", 0))
        task = self.tasks.get(task_idx, "")

        return {
            "images": images,
            "state": torch.tensor(state, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "task": task,
        }


def collate_droid(batch):
    """Collate: keep images as list of PIL, stack tensors."""
    return {
        "images": [b["images"] for b in batch],  # list of list of PIL
        "state": torch.stack([b["state"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "tasks": [b["task"] for b in batch],
    }


# ============================================================================
# Training
# ============================================================================

def download_from_hub(repo_id: str, cache_dir: str = None) -> str:
    """Download a LeRobot dataset from Hugging Face Hub. Returns local path."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    logger.info(f"Dataset ready: {repo_id} → {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace dataset repo (e.g. lerobot/droid_100). Auto-downloads.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Local dataset path. If --repo_id is given, this is ignored.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where to cache downloaded datasets (default: HF cache)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--knowledge_insulation", action="store_true")
    parser.add_argument("--rtc", action="store_true", help="Enable Training-Time RTC")
    parser.add_argument("--rtc_max_delay", type=int, default=4,
                        help="Max prefix delay for RTC (sampled from {0,...,max_delay})")
    parser.add_argument("--dummy", action="store_true", help="Use dummy backbone (CPU testing)")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="checkpoints/droid")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--normalize", action="store_true", help="Z-score normalize actions/states")
    args = parser.parse_args()

    # Resolve data source: --repo_id (Hub) or --data_dir (local)
    if args.repo_id:
        args.data_dir = download_from_hub(args.repo_id, cache_dir=args.cache_dir)
    elif args.data_dir is None:
        args.data_dir = os.path.expanduser("~/.cache/huggingface/lerobot/lerobot/droid_100")

    from gemma4vla.model.config import Gemma4ActionExpertConfig
    from gemma4vla.model.policy import Gemma4VLAPolicy

    config = Gemma4ActionExpertConfig(
        variant="flow_shared",
        action_dim=32,
        action_horizon=15,
    )
    logger.info(f"Config: variant={config.variant}, depth={config.expert_depth}, width={config.expert_width}")
    if args.rtc:
        logger.info(f"Training-Time RTC enabled: max_delay={args.rtc_max_delay}")

    # Model
    if args.dummy:
        logger.info("Using dummy backbone (CPU testing)")
        policy = Gemma4VLAPolicy(config, dummy=True)
        policy.to(args.device)
    else:
        logger.info("Loading real Gemma 4 backbone...")
        policy = Gemma4VLAPolicy(config, dummy=False, device=args.device)
        policy.action_expert.to(args.device, dtype=config.torch_dtype)

    # Resume from checkpoint
    if args.resume:
        state_dict = torch.load(args.resume, map_location=args.device, weights_only=True)
        policy.action_expert.load_state_dict(state_dict)
        logger.info(f"Resumed from: {args.resume}")

    expert_params = sum(p.numel() for p in policy.action_expert.parameters())
    logger.info(f"Expert params: {expert_params/1e6:.1f}M")

    # Dataset
    dataset = DroidV3Dataset(args.data_dir, action_horizon=config.action_horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_droid, num_workers=0)

    # Normalization
    normalizer = ActionStateNormalizer()
    stats_path = os.path.join(args.output_dir, "norm_stats.pt")
    if args.normalize:
        if os.path.isfile(stats_path):
            normalizer.load(stats_path)
        else:
            logger.info("Computing normalization statistics...")
            normalizer.compute_from_dataset(dataset)
            os.makedirs(args.output_dir, exist_ok=True)
            normalizer.save(stats_path)

    # Training setup
    if args.knowledge_insulation:
        from gemma4vla.training.knowledge_insulation import KnowledgeInsulationTrainer
        ki = KnowledgeInsulationTrainer(
            policy, lr_backbone=args.lr_backbone, lr_expert=args.lr,
            use_fast_backbone=False, device=args.device,
        )
        logger.info("Using Knowledge Insulation training")
    else:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                policy.get_action_expert_params(), lr=args.lr, weight_decay=0.01,
            )
            logger.info("Using 8-bit AdamW (saves ~5GB VRAM)")
        except ImportError:
            optimizer = torch.optim.AdamW(
                policy.get_action_expert_params(), lr=args.lr, weight_decay=0.01,
            )
            logger.info("Using standard AdamW (bitsandbytes not available)")
        ki = None
        logger.info("Using standard expert-only training")

    # Learning rate scheduler (linear warmup + cosine decay)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    if ki is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    os.makedirs(args.output_dir, exist_ok=True)

    # RTC delay sampling weights (exponentially decreasing, per paper)
    if args.rtc:
        rtc_delays = list(range(args.rtc_max_delay + 1))  # {0, 1, 2, 3, 4}
        rtc_weights = [0.5 ** d for d in rtc_delays]  # exponential decay
        rtc_weight_sum = sum(rtc_weights)
        rtc_weights = [w / rtc_weight_sum for w in rtc_weights]
        logger.info(f"RTC delay distribution: delays={rtc_delays}, weights={[f'{w:.3f}' for w in rtc_weights]}")

    # Training loop
    step = 0
    total_loss = 0.0
    policy.action_expert.train()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            t0 = time.time()

            if args.dummy:
                # Dummy mode: use random tensors
                B = batch["state"].shape[0]
                images = [torch.randn(B, 3, 448, 448, device=args.device)]
                lang_tokens = torch.randint(0, 1000, (B, 16), device=args.device)
                state = batch["state"].to(args.device)
                actions = batch["actions"].to(args.device)
            else:
                # Real mode: preprocess images with Gemma 4
                B = batch["state"].shape[0]
                bb = policy._real_backbone

                # Encode each sample's images
                all_contexts = []
                for b_idx in range(B):
                    pil_imgs = batch["images"][b_idx]
                    task = batch["tasks"][b_idx] or "manipulate object"
                    ctx = bb.encode_observation(
                        images_pil=pil_imgs, text=task, device=args.device,
                        return_embeddings_only=(config.variant == "flow_shared"),
                    )
                    all_contexts.append(ctx)

                context = torch.cat(all_contexts, dim=0)  # (B, S, 2560)
                state = batch["state"].to(args.device, dtype=config.torch_dtype)
                actions = batch["actions"].to(args.device, dtype=config.torch_dtype)

            # Normalize
            if args.normalize:
                actions = normalizer.normalize_actions(actions)
                state = normalizer.normalize_state(state)

            # Sample RTC delay
            prefix_len = 0
            if args.rtc:
                prefix_len = random.choices(rtc_delays, weights=rtc_weights, k=1)[0]

            if ki is not None and args.dummy:
                losses = ki.train_step(images, lang_tokens, state, actions)
                loss_val = losses["total_loss"]
            elif ki is not None:
                # KI with real backbone
                context_detached = context.detach()
                expert_loss = policy.flow_loss(
                    predict_velocity_fn=policy.action_expert,
                    target_actions=actions,
                    state=state,
                    context=context_detached,
                    prefix_len=prefix_len,
                )
                optimizer_expert = ki.optimizer_expert
                optimizer_expert.zero_grad()
                expert_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.action_expert.parameters(), 1.0)
                optimizer_expert.step()
                loss_val = expert_loss.item()
            else:
                if args.dummy:
                    loss = policy.compute_loss(images, lang_tokens, state, actions)
                else:
                    loss = policy.flow_loss(
                        predict_velocity_fn=policy.action_expert,
                        target_actions=actions,
                        state=state,
                        context=context.detach(),
                        prefix_len=prefix_len,
                    )
                loss_val = loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.get_action_expert_params(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            dt = (time.time() - t0) * 1000
            total_loss += loss_val
            step += 1

            if step % 10 == 0 or step == 1:
                avg = total_loss / step
                lr_cur = scheduler.get_last_lr()[0] if scheduler else args.lr
                rtc_str = f" d={prefix_len}" if args.rtc else ""
                logger.info(
                    f"step={step}/{args.steps} loss={loss_val:.4f} avg={avg:.4f} "
                    f"lr={lr_cur:.2e} time={dt:.0f}ms{rtc_str}"
                )

            if step > 0 and step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"expert_step{step}.pt")
                torch.save(policy.action_expert.state_dict(), ckpt)
                logger.info(f"Saved: {ckpt}")

    # Final save
    ckpt = os.path.join(args.output_dir, f"expert_step{step}.pt")
    torch.save(policy.action_expert.state_dict(), ckpt)
    logger.info(f"Training complete. Final avg loss: {total_loss/max(step,1):.4f}")
    logger.info(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
