"""DROID dataset loader for training.

Supports two modes:
1. Online: Load raw data + compute backbone features on-the-fly (slow)
2. Offline: Load pre-computed backbone features (fast, recommended)

Pre-compute features:
    python scripts/precompute_features.py --data_dir /path/to/droid_lerobot
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DroidPrecomputedDataset(Dataset):
    """Load pre-computed backbone features + action labels.

    Expected directory structure:
        data_dir/
            episode_0/
                features.npz   — backbone hidden states per timestep
                actions.npz    — action chunks
                states.npz     — proprioception
            episode_1/
            ...
    """

    def __init__(self, data_dir: str, action_horizon: int = 15, action_dim: int = 8):
        self.data_dir = Path(data_dir)
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        for ep_dir in sorted(self.data_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            features_path = ep_dir / "features.npz"
            actions_path = ep_dir / "actions.npz"
            states_path = ep_dir / "states.npz"
            if not all(p.exists() for p in [features_path, actions_path, states_path]):
                continue

            features = np.load(features_path)
            n_steps = len(features.files)

            for t in range(n_steps - self.action_horizon):
                samples.append((ep_dir, t))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_dir, t = self.samples[idx]

        features = np.load(ep_dir / "features.npz")
        actions = np.load(ep_dir / "actions.npz")
        states = np.load(ep_dir / "states.npz")

        # Backbone features at timestep t
        context = features[f"step_{t}"]  # (seq_len, hidden_size)

        # Action chunk: actions[t:t+H]
        action_chunk = np.stack([
            actions[f"step_{t+i}"] for i in range(self.action_horizon)
        ])  # (H, action_dim)

        # Proprioception at timestep t
        state = states[f"step_{t}"]  # (action_dim,)

        return {
            "context": torch.tensor(context, dtype=torch.float32),
            "actions": torch.tensor(action_chunk, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
        }
