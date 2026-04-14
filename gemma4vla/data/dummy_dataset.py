"""Dummy dataset for testing the training loop.

Generates random observations and actions matching DROID format.
"""

import torch
from torch.utils.data import Dataset


class DummyDROIDDataset(Dataset):
    """Random data matching DROID VLA format.

    Each sample contains:
        - 2 camera images (224×224×3)
        - Language tokens (padded to max_len)
        - State (action_dim)
        - Target action chunk (action_horizon × action_dim)
    """

    def __init__(
        self,
        num_samples: int = 1000,
        action_dim: int = 32,
        action_horizon: int = 15,
        max_token_len: int = 32,
        image_size: int = 448,
        num_cameras: int = 2,
    ):
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.image_size = image_size
        self.num_cameras = num_cameras

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "images": [
                torch.randn(3, self.image_size, self.image_size)
                for _ in range(self.num_cameras)
            ],
            "lang_tokens": torch.randint(0, 1000, (self.max_token_len,)),
            "state": torch.randn(self.action_dim),
            "actions": torch.randn(self.action_horizon, self.action_dim),
        }


def collate_fn(batch):
    """Custom collate: stack images per camera, pad language tokens."""
    num_cameras = len(batch[0]["images"])
    return {
        "images": [
            torch.stack([b["images"][i] for b in batch])
            for i in range(num_cameras)
        ],
        "lang_tokens": torch.stack([b["lang_tokens"] for b in batch]),
        "state": torch.stack([b["state"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
    }
