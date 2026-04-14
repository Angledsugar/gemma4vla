"""LeRobot v3 dataset loader for Gemma4VLA training.

Supports any LeRobot v3.0 dataset (local or Hub) with:
  - Arbitrary robot morphologies (state/action dims vary per dataset)
  - Multiple camera views (observation.images.*)
  - Task-conditioned training (natural language prompts)
  - Action chunking (consecutive actions as target)

Compatible with:
  - DROID, Bridge, OXE, Aloha, PushT, etc.
  - Any dataset following LeRobot v3 format

Usage:
    from gemma4vla.data.lerobot_v3 import LeRobotV3Dataset

    # Single dataset
    dataset = LeRobotV3Dataset("lerobot/pusht_image", action_horizon=15)

    # Multiple datasets (mixed training)
    dataset = LeRobotV3MixedDataset([
        ("lerobot/droid_100", 0.5),       # 50% DROID
        ("lerobot/bridge_v2", 0.3),       # 30% Bridge
        ("lerobot/aloha_sim_cube", 0.2),  # 20% ALOHA
    ], action_horizon=15)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class LeRobotV3Dataset(Dataset):
    """Load a single LeRobot v3 dataset for VLA training.

    Each sample returns:
        - images: dict of camera tensors {cam_name: (C, H, W)}
        - state: (state_dim,) proprioception
        - actions: (action_horizon, action_dim) target action chunk
        - task: str — natural language task instruction
        - episode_index: int
        - frame_index: int
    """

    def __init__(
        self,
        repo_id: str,
        action_horizon: int = 15,
        action_dim: int = None,      # None = auto from dataset
        state_key: str = "observation.state",
        action_key: str = "action",
        image_keys: list[str] = None,  # None = auto-detect
        task_key: str = None,         # column with task string
        split: str = "train",
        root: str = None,             # local path override
        max_episodes: int = None,
        image_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.state_key = state_key
        self.action_key = action_key
        self.image_size = image_size

        # Load dataset
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self._lerobot = LeRobotDataset(repo_id, root=root, split=split)
            self._use_lerobot = True
        except ImportError:
            # Fallback: load parquet directly
            logger.warning("lerobot not installed, using direct parquet loading")
            self._lerobot = None
            self._use_lerobot = False
            self._load_parquet(repo_id, root)

        # Auto-detect features from info.json
        if self._use_lerobot:
            self._info = self._lerobot.meta.info
            self._features = self._info.get("features", {})
        else:
            self._info = self._meta_info
            self._features = self._info.get("features", {})

        # Resolve image keys
        if image_keys is None:
            self.image_keys = [k for k, v in self._features.items()
                               if v.get("dtype") == "image" or "images" in k]
        else:
            self.image_keys = image_keys

        # Resolve dimensions
        state_feat = self._features.get(state_key, {})
        action_feat = self._features.get(action_key, {})
        self.state_dim = state_feat.get("shape", [0])[0] if state_feat else 0
        self.dataset_action_dim = action_feat.get("shape", [0])[0] if action_feat else 0
        self.action_dim = action_dim or self.dataset_action_dim

        # Task descriptions
        self.task_key = task_key
        self._tasks = self._load_tasks(repo_id, root)

        # Build valid indices (frames where full action chunk is available)
        self._build_index(max_episodes)

        logger.info(
            f"LeRobotV3: {repo_id} | "
            f"{len(self._valid_indices)} samples | "
            f"state_dim={self.state_dim} action_dim={self.action_dim} | "
            f"cameras={self.image_keys} | "
            f"tasks={len(self._tasks)}"
        )

    def _load_parquet(self, repo_id: str, root: str = None):
        """Fallback: load data directly from parquet files."""
        import pyarrow.parquet as pq

        if root:
            base = Path(root)
        else:
            cache = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id.replace("/", "/")
            base = cache

        # Load info.json
        info_path = base / "meta" / "info.json"
        with open(info_path) as f:
            self._meta_info = json.load(f)

        # Load all parquet files
        data_dir = base / "data"
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {data_dir}")

        tables = [pq.read_table(str(f)) for f in parquet_files]
        import pyarrow as pa
        self._table = pa.concat_tables(tables)
        self._df = self._table.to_pandas()

    def _load_tasks(self, repo_id: str, root: str = None) -> dict:
        """Load task descriptions from tasks.jsonl or tasks.parquet."""
        tasks = {}
        try:
            if self._use_lerobot and hasattr(self._lerobot.meta, 'tasks'):
                for task in self._lerobot.meta.tasks.values():
                    tasks[task.get("task_index", 0)] = task.get("task", "")
        except Exception:
            pass

        if not tasks:
            tasks[0] = ""  # default empty task
        return tasks

    def _build_index(self, max_episodes: int = None):
        """Build list of valid (episode, frame) pairs for action chunking."""
        self._valid_indices = []

        if self._use_lerobot:
            n_frames = len(self._lerobot)
            for idx in range(n_frames - self.action_horizon):
                # Check same episode for action chunk
                ep_current = self._lerobot[idx].get("episode_index", 0)
                ep_end = self._lerobot[idx + self.action_horizon - 1].get("episode_index", 0)
                if isinstance(ep_current, torch.Tensor):
                    ep_current = ep_current.item()
                if isinstance(ep_end, torch.Tensor):
                    ep_end = ep_end.item()
                if ep_current == ep_end:
                    self._valid_indices.append(idx)

                if max_episodes and ep_current >= max_episodes:
                    break
        else:
            # Direct parquet
            episodes = self._df["episode_index"].values
            for i in range(len(episodes) - self.action_horizon):
                if episodes[i] == episodes[i + self.action_horizon - 1]:
                    self._valid_indices.append(i)
                if max_episodes and episodes[i] >= max_episodes:
                    break

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx) -> dict:
        frame_idx = self._valid_indices[idx]

        if self._use_lerobot:
            return self._get_lerobot(frame_idx)
        return self._get_parquet(frame_idx)

    def _get_lerobot(self, frame_idx: int) -> dict:
        """Get sample from lerobot dataset."""
        sample = self._lerobot[frame_idx]

        # State
        state = sample.get(self.state_key, torch.zeros(self.state_dim))
        if isinstance(state, torch.Tensor):
            state = state.float()

        # Action chunk
        actions = []
        for i in range(self.action_horizon):
            s = self._lerobot[frame_idx + i]
            a = s.get(self.action_key, torch.zeros(self.dataset_action_dim))
            if isinstance(a, torch.Tensor):
                a = a.float()
            actions.append(a)
        actions = torch.stack(actions)  # (H, action_dim)

        # Pad/truncate action dim
        if self.action_dim != self.dataset_action_dim:
            if self.action_dim > self.dataset_action_dim:
                pad = torch.zeros(self.action_horizon, self.action_dim - self.dataset_action_dim)
                actions = torch.cat([actions, pad], dim=-1)
            else:
                actions = actions[:, :self.action_dim]

        # Images
        images = {}
        for key in self.image_keys:
            img = sample.get(key)
            if img is not None:
                if isinstance(img, torch.Tensor):
                    # Ensure (C, H, W) format
                    if img.dim() == 3 and img.shape[-1] in (1, 3):
                        img = img.permute(2, 0, 1)  # HWC → CHW
                    images[key] = img.float()

        # Task
        ep_idx = sample.get("episode_index", 0)
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        task_idx = sample.get("task_index", 0)
        if isinstance(task_idx, torch.Tensor):
            task_idx = task_idx.item()
        task = self._tasks.get(task_idx, self._tasks.get(0, ""))

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "task": task,
            "episode_index": ep_idx,
            "frame_index": frame_idx,
        }

    def _get_parquet(self, frame_idx: int) -> dict:
        """Get sample from direct parquet loading."""
        row = self._df.iloc[frame_idx]

        state = torch.tensor(row.get(self.state_key, [0.0] * self.state_dim), dtype=torch.float32)

        actions = []
        for i in range(self.action_horizon):
            a = self._df.iloc[frame_idx + i].get(self.action_key, [0.0] * self.dataset_action_dim)
            actions.append(torch.tensor(a, dtype=torch.float32))
        actions = torch.stack(actions)

        return {
            "images": {},  # parquet doesn't have images inline
            "state": state,
            "actions": actions,
            "task": "",
            "episode_index": int(row.get("episode_index", 0)),
            "frame_index": frame_idx,
        }


class LeRobotV3MixedDataset(Dataset):
    """Mixed training from multiple LeRobot v3 datasets with weighted sampling.

    Normalizes state/action dimensions across datasets by padding to max dim.

    Usage:
        dataset = LeRobotV3MixedDataset([
            ("lerobot/droid_100", 0.5),
            ("lerobot/bridge_v2", 0.3),
            ("lerobot/aloha_sim", 0.2),
        ], action_horizon=15, action_dim=32)
    """

    def __init__(
        self,
        datasets_with_weights: list[tuple[str, float]],
        action_horizon: int = 15,
        action_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.datasets = []
        self.weights = []
        self.cumulative_sizes = []

        total = 0
        for repo_id, weight in datasets_with_weights:
            ds = LeRobotV3Dataset(
                repo_id,
                action_horizon=action_horizon,
                action_dim=action_dim,
                **kwargs,
            )
            self.datasets.append(ds)
            self.weights.append(weight)
            total += len(ds)
            self.cumulative_sizes.append(total)

        self._total = total
        logger.info(f"MixedDataset: {len(self.datasets)} sources, {self._total} total samples")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        # Find which dataset
        for i, cum in enumerate(self.cumulative_sizes):
            if idx < cum:
                local_idx = idx - (self.cumulative_sizes[i - 1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")

    def get_sampler(self, num_samples: int = None) -> WeightedRandomSampler:
        """Create weighted sampler respecting dataset mixture ratios."""
        sample_weights = []
        for i, ds in enumerate(self.datasets):
            w = self.weights[i] / len(ds)
            sample_weights.extend([w] * len(ds))

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples or self._total,
            replacement=True,
        )


def collate_vla(batch: list[dict]) -> dict:
    """Collate function for VLA training.

    Handles variable image keys across samples by using the
    union of all camera keys, padding missing cameras with zeros.
    """
    # Collect all image keys
    all_img_keys = set()
    for sample in batch:
        all_img_keys.update(sample["images"].keys())

    # Determine image shape from first available
    img_shape = None
    for sample in batch:
        for v in sample["images"].values():
            if isinstance(v, torch.Tensor) and v.dim() == 3:
                img_shape = v.shape
                break
        if img_shape:
            break

    result = {
        "state": torch.stack([s["state"] for s in batch]),
        "actions": torch.stack([s["actions"] for s in batch]),
        "tasks": [s["task"] for s in batch],
    }

    # Images: pad missing cameras with zeros
    if all_img_keys and img_shape:
        result["images"] = {}
        for key in sorted(all_img_keys):
            imgs = []
            for sample in batch:
                img = sample["images"].get(key)
                if img is None:
                    img = torch.zeros(img_shape)
                imgs.append(img)
            result["images"][key] = torch.stack(imgs)

    return result
