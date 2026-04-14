"""Train Gemma4VLA base model on multiple LeRobot datasets from Hugging Face Hub.

Automatically downloads any LeRobot-format dataset, handles different robot
morphologies (action/state dims, cameras), and trains a single base model.

Usage:
    # From YAML config (recommended)
    uv run scripts/train_base.py --config configs/base_mix.yaml

    # Quick: single dataset
    uv run scripts/train_base.py --datasets lerobot/droid_100

    # Multiple datasets with weights
    uv run scripts/train_base.py \
        --datasets lerobot/droid_100:0.5 lerobot/pusht:0.3 lerobot/aloha_sim_insertion_human:0.2

    # Dummy mode (CPU, no GPU needed)
    uv run scripts/train_base.py --datasets lerobot/pusht --dummy --device cpu --steps 5
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Hub download
# ============================================================================

def download_from_hub(repo_id: str, cache_dir: str = None) -> str:
    """Download a LeRobot dataset from Hugging Face Hub. Returns local path."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading {repo_id} from Hub...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    logger.info(f"  → {local_dir}")
    return local_dir


# ============================================================================
# Universal LeRobot dataset (handles any robot morphology)
# ============================================================================

class UniversalLeRobotDataset(Dataset):
    """Load any LeRobot v3 dataset, pad action/state to common dims.

    Handles:
    - Different action dims (DROID=7, ALOHA=14, PushT=2, etc.)
    - Different state dims
    - Different camera configurations
    - Video or image data
    """

    def __init__(
        self,
        data_dir: str,
        action_dim: int = 32,
        action_horizon: int = 15,
        image_size: int = 448,
    ):
        import json
        import pyarrow.parquet as pq

        self.data_dir = Path(data_dir)
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.image_size = image_size

        # Load metadata
        info_path = self.data_dir / "meta" / "info.json"
        if not info_path.exists():
            # snapshot_download may put files in a subdirectory
            candidates = list(self.data_dir.rglob("meta/info.json"))
            if candidates:
                info_path = candidates[0]
                self.data_dir = info_path.parent.parent
            else:
                raise FileNotFoundError(f"No meta/info.json found in {data_dir}")

        with open(info_path) as f:
            self.info = json.load(f)

        self.repo_name = self.info.get("repo_id", self.data_dir.name)
        self.fps = self.info.get("fps", 10)
        features = self.info.get("features", {})

        # Detect action/state keys and dims
        self.action_key = "action"
        self.state_key = "observation.state"
        self.raw_action_dim = features.get(self.action_key, {}).get("shape", [0])[0]
        self.raw_state_dim = features.get(self.state_key, {}).get("shape", [0])[0]

        # Detect camera keys
        self.cameras = [
            k for k, v in features.items()
            if v.get("dtype") in ("video", "image") and "images" in k
        ]

        # Load parquet data
        parquet_files = sorted((self.data_dir / "data").glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {self.data_dir / 'data'}")

        import pyarrow as pa
        tables = [pq.read_table(str(f)) for f in parquet_files]
        self.df = pa.concat_tables(tables).to_pandas()

        # Build valid indices (full action chunk within same episode)
        episodes = self.df["episode_index"].values
        self.valid_indices = []
        for i in range(len(episodes) - action_horizon):
            if episodes[i] == episodes[i + action_horizon - 1]:
                self.valid_indices.append(i)

        # Load tasks
        self.tasks = {}
        try:
            tasks_path = self.data_dir / "meta" / "tasks.parquet"
            if tasks_path.exists():
                tasks_df = pq.read_table(str(tasks_path)).to_pandas()
                for _, row in tasks_df.iterrows():
                    self.tasks[row["task_index"]] = row["task"]
        except Exception:
            pass
        if not self.tasks:
            self.tasks[0] = ""

        # Video reader cache
        self._video_cache = {}

        logger.info(
            f"  {self.repo_name}: {len(self.valid_indices)} samples, "
            f"{self.info.get('total_episodes', '?')} episodes, "
            f"action={self.raw_action_dim}→{self.action_dim}, "
            f"state={self.raw_state_dim}→{self.action_dim}, "
            f"cameras={self.cameras}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def _read_video_frame(self, camera: str, episode_idx: int, frame_idx: int) -> np.ndarray:
        """Read a single frame from video file."""
        from PIL import Image as PILImage
        try:
            import av
        except ImportError:
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

        video_template = self.info.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        chunk_size = self.info.get("chunks_size", 1000)
        chunk_idx = episode_idx // chunk_size

        video_path = self.data_dir / video_template.format(
            video_key=camera,
            chunk_index=chunk_idx,
            file_index=chunk_idx,
        )

        if not video_path.exists():
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

        cache_key = str(video_path)
        if cache_key not in self._video_cache:
            self._video_cache[cache_key] = av.open(str(video_path))

        container = self._video_cache[cache_key]
        stream = container.streams.video[0]

        try:
            container.seek(frame_idx, stream=stream)
            for frame in container.decode(stream):
                img = frame.to_ndarray(format="rgb24")
                pil_img = PILImage.fromarray(img).resize(
                    (self.image_size, self.image_size), PILImage.BILINEAR
                )
                return np.array(pil_img)
        except Exception:
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

    def __getitem__(self, idx):
        from PIL import Image as PILImage

        frame_idx = self.valid_indices[idx]
        row = self.df.iloc[frame_idx]

        # State → pad to action_dim
        raw_state = np.array(row.get(self.state_key, [0.0] * self.raw_state_dim), dtype=np.float32)
        state = np.zeros(self.action_dim, dtype=np.float32)
        state[:len(raw_state)] = raw_state

        # Action chunk → pad to action_dim
        actions = []
        for i in range(self.action_horizon):
            raw_a = np.array(self.df.iloc[frame_idx + i][self.action_key], dtype=np.float32)
            a = np.zeros(self.action_dim, dtype=np.float32)
            a[:len(raw_a)] = raw_a
            actions.append(a)
        actions = np.stack(actions)

        # Images
        episode_idx = int(row["episode_index"])
        global_frame = int(row["frame_index"])
        images = []
        for cam in self.cameras[:2]:
            img = self._read_video_frame(cam, episode_idx, global_frame)
            images.append(PILImage.fromarray(img))

        # If no cameras, use a blank image
        if not images:
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            images.append(PILImage.fromarray(blank))

        # Task
        task_idx = int(row.get("task_index", 0))
        task = self.tasks.get(task_idx, self.tasks.get(0, ""))

        return {
            "images": images,
            "state": torch.tensor(state, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "task": task,
            "dataset": self.repo_name,
        }


class MixedRobotDataset(Dataset):
    """Combine multiple UniversalLeRobotDatasets with weighted sampling."""

    def __init__(self, datasets: list[UniversalLeRobotDataset], weights: list[float]):
        self.datasets = datasets
        self.weights = weights
        self.cumulative = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative.append(total)
        self._total = total

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        for i, cum in enumerate(self.cumulative):
            if idx < cum:
                local_idx = idx - (self.cumulative[i - 1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        raise IndexError(idx)

    def get_sampler(self) -> WeightedRandomSampler:
        """Weighted sampler: respects dataset mixture ratios."""
        sample_weights = []
        for i, ds in enumerate(self.datasets):
            w = self.weights[i] / len(ds)
            sample_weights.extend([w] * len(ds))
        return WeightedRandomSampler(sample_weights, num_samples=self._total, replacement=True)


def collate_mixed(batch):
    return {
        "images": [b["images"] for b in batch],
        "state": torch.stack([b["state"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "tasks": [b["task"] for b in batch],
        "datasets": [b["dataset"] for b in batch],
    }


# ============================================================================
# Normalization (same as train_droid.py)
# ============================================================================

class ActionStateNormalizer:
    def __init__(self):
        self.action_mean = None
        self.action_std = None
        self.state_mean = None
        self.state_std = None

    def compute_from_dataset(self, dataset, max_samples=10000):
        n = min(len(dataset), max_samples)
        indices = random.sample(range(len(dataset)), n)
        all_actions, all_states = [], []
        for idx in indices:
            item = dataset[idx]
            all_actions.append(item["actions"].numpy())
            all_states.append(item["state"].numpy())
        actions = np.stack(all_actions).reshape(-1, all_actions[0].shape[-1])
        states = np.stack(all_states)
        self.action_mean = torch.tensor(actions.mean(0), dtype=torch.float32)
        self.action_std = torch.tensor(actions.std(0).clip(min=1e-6), dtype=torch.float32)
        self.state_mean = torch.tensor(states.mean(0), dtype=torch.float32)
        self.state_std = torch.tensor(states.std(0).clip(min=1e-6), dtype=torch.float32)
        logger.info(f"Norm stats from {n} samples")

    def normalize_actions(self, a):
        if self.action_mean is None: return a
        return (a - self.action_mean.to(a.device, a.dtype)) / self.action_std.to(a.device, a.dtype)

    def normalize_state(self, s):
        if self.state_mean is None: return s
        return (s - self.state_mean.to(s.device, s.dtype)) / self.state_std.to(s.device, s.dtype)

    def save(self, path):
        torch.save({"action_mean": self.action_mean, "action_std": self.action_std,
                     "state_mean": self.state_mean, "state_std": self.state_std}, path)

    def load(self, path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        self.action_mean, self.action_std = d["action_mean"], d["action_std"]
        self.state_mean, self.state_std = d["state_mean"], d["state_std"]


# ============================================================================
# Main
# ============================================================================

def parse_dataset_arg(s: str) -> tuple[str, float]:
    """Parse 'repo_id:weight' or 'repo_id' (default weight=1.0)."""
    if ":" in s and not s.startswith("hf://"):
        parts = s.rsplit(":", 1)
        try:
            return parts[0], float(parts[1])
        except ValueError:
            return s, 1.0
    return s, 1.0


def main():
    parser = argparse.ArgumentParser(description="Train base model on multiple LeRobot datasets")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="repo_id:weight pairs (e.g. lerobot/droid_100:0.5 lerobot/pusht:0.3)")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--action_dim", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=15)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--knowledge_insulation", action="store_true")
    parser.add_argument("--rtc", action="store_true")
    parser.add_argument("--rtc_max_delay", type=int, default=4)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="checkpoints/base_mix")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load YAML config (CLI args override)
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Apply config values as defaults (CLI takes priority)
        train_cfg = cfg.get("training", {})
        for k, v in train_cfg.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
        if args.cache_dir is None:
            args.cache_dir = cfg.get("cache_dir")
        if args.action_dim == 32 and "action_dim" in cfg:
            args.action_dim = cfg["action_dim"]
        if args.action_horizon == 15 and "action_horizon" in cfg:
            args.action_horizon = cfg["action_horizon"]

    # ---- Resolve datasets ----
    dataset_specs = []  # [(repo_id, weight, local_dir)]

    if args.datasets:
        # From CLI: --datasets lerobot/droid_100:0.5 lerobot/pusht:0.3
        for ds_str in args.datasets:
            repo_id, weight = parse_dataset_arg(ds_str)
            dataset_specs.append((repo_id, weight))
    elif args.config:
        # From YAML
        for entry in cfg.get("datasets", []):
            dataset_specs.append((entry["repo_id"], entry.get("weight", 1.0)))
    else:
        # Default: just droid_100
        dataset_specs.append(("lerobot/droid_100", 1.0))

    # Normalize weights
    total_w = sum(w for _, w in dataset_specs)
    dataset_specs = [(r, w / total_w) for r, w in dataset_specs]

    logger.info(f"=== Base Model Training: {len(dataset_specs)} datasets ===")
    for repo_id, weight in dataset_specs:
        logger.info(f"  {repo_id} (weight={weight:.2f})")

    # ---- Download all datasets from Hub ----
    datasets = []
    weights = []
    for repo_id, weight in dataset_specs:
        local_dir = download_from_hub(repo_id, cache_dir=args.cache_dir)
        ds = UniversalLeRobotDataset(
            data_dir=local_dir,
            action_dim=args.action_dim,
            action_horizon=args.action_horizon,
        )
        datasets.append(ds)
        weights.append(weight)

    mixed = MixedRobotDataset(datasets, weights)
    sampler = mixed.get_sampler()
    loader = DataLoader(
        mixed, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_mixed, num_workers=0,
    )

    logger.info(f"Total samples: {len(mixed)}")

    # ---- Normalization ----
    normalizer = ActionStateNormalizer()
    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, "norm_stats.pt")
    if args.normalize:
        if os.path.isfile(stats_path):
            normalizer.load(stats_path)
        else:
            logger.info("Computing normalization statistics across all datasets...")
            normalizer.compute_from_dataset(mixed)
            normalizer.save(stats_path)

    # ---- Model ----
    from gemma4vla.model.config import Gemma4ActionExpertConfig
    from gemma4vla.model.policy import Gemma4VLAPolicy

    config = Gemma4ActionExpertConfig(
        variant="flow_shared",
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
    )

    if args.dummy:
        policy = Gemma4VLAPolicy(config, dummy=True)
        policy.to(args.device)
    else:
        policy = Gemma4VLAPolicy(config, dummy=False, device=args.device)
        policy.action_expert.to(args.device, dtype=config.torch_dtype)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=args.device, weights_only=True)
        policy.action_expert.load_state_dict(state_dict)
        logger.info(f"Resumed from: {args.resume}")

    expert_params = sum(p.numel() for p in policy.action_expert.parameters())
    logger.info(f"Expert params: {expert_params / 1e6:.1f}M")

    # ---- Optimizer ----
    if args.knowledge_insulation:
        from gemma4vla.training.knowledge_insulation import KnowledgeInsulationTrainer
        ki = KnowledgeInsulationTrainer(
            policy, lr_backbone=args.lr_backbone, lr_expert=args.lr,
            use_fast_backbone=False, device=args.device,
        )
    else:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                policy.get_action_expert_params(), lr=args.lr, weight_decay=0.01,
            )
        except ImportError:
            optimizer = torch.optim.AdamW(
                policy.get_action_expert_params(), lr=args.lr, weight_decay=0.01,
            )
        ki = None

    # LR scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if ki is None else None

    # RTC
    if args.rtc:
        rtc_delays = list(range(args.rtc_max_delay + 1))
        rtc_weights = [0.5 ** d for d in rtc_delays]
        rtc_sum = sum(rtc_weights)
        rtc_weights = [w / rtc_sum for w in rtc_weights]

    # ---- Training loop ----
    step = 0
    total_loss = 0.0
    policy.action_expert.train()

    logger.info(f"=== Training starts: {args.steps} steps ===")

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            t0 = time.time()

            if args.dummy:
                B = batch["state"].shape[0]
                images = [torch.randn(B, 3, 448, 448, device=args.device)]
                lang_tokens = torch.randint(0, 1000, (B, 16), device=args.device)
                state = batch["state"].to(args.device)
                actions = batch["actions"].to(args.device)
            else:
                B = batch["state"].shape[0]
                bb = policy._real_backbone
                all_contexts = []
                for b_idx in range(B):
                    pil_imgs = batch["images"][b_idx]
                    task = batch["tasks"][b_idx] or "manipulate object"
                    ctx = bb.encode_observation(
                        images_pil=pil_imgs, text=task, device=args.device,
                        return_embeddings_only=(config.variant == "flow_shared"),
                    )
                    all_contexts.append(ctx)
                context = torch.cat(all_contexts, dim=0)
                state = batch["state"].to(args.device, dtype=config.torch_dtype)
                actions = batch["actions"].to(args.device, dtype=config.torch_dtype)

            if args.normalize:
                actions = normalizer.normalize_actions(actions)
                state = normalizer.normalize_state(state)

            prefix_len = 0
            if args.rtc:
                prefix_len = random.choices(rtc_delays, weights=rtc_weights, k=1)[0]

            # Forward + backward
            if ki is not None and args.dummy:
                losses = ki.train_step(images, lang_tokens, state, actions)
                loss_val = losses["total_loss"]
            elif ki is not None:
                context_detached = context.detach()
                expert_loss = policy.flow_loss(
                    predict_velocity_fn=policy.action_expert,
                    target_actions=actions, state=state,
                    context=context_detached, prefix_len=prefix_len,
                )
                ki.optimizer_expert.zero_grad()
                expert_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.action_expert.parameters(), 1.0)
                ki.optimizer_expert.step()
                loss_val = expert_loss.item()
            else:
                if args.dummy:
                    loss = policy.compute_loss(images, lang_tokens, state, actions)
                else:
                    loss = policy.flow_loss(
                        predict_velocity_fn=policy.action_expert,
                        target_actions=actions, state=state,
                        context=context.detach(), prefix_len=prefix_len,
                    )
                loss_val = loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.get_action_expert_params(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            dt = (time.time() - t0) * 1000
            total_loss += loss_val
            step += 1

            if step % 10 == 0 or step == 1:
                avg = total_loss / step
                lr_cur = scheduler.get_last_lr()[0] if scheduler else args.lr
                ds_names = set(batch["datasets"])
                logger.info(
                    f"step={step}/{args.steps} loss={loss_val:.4f} avg={avg:.4f} "
                    f"lr={lr_cur:.2e} time={dt:.0f}ms ds={ds_names}"
                )

            if step > 0 and step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"expert_step{step}.pt")
                torch.save(policy.action_expert.state_dict(), ckpt)
                logger.info(f"Saved: {ckpt}")

    # Final save
    ckpt = os.path.join(args.output_dir, f"expert_step{step}.pt")
    torch.save(policy.action_expert.state_dict(), ckpt)
    logger.info(f"Training complete. Final avg loss: {total_loss / max(step, 1):.4f}")
    logger.info(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
