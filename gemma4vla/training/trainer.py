"""Training loop for Gemma4-VLA.

Only the action expert is trained. The backbone is frozen and used
to pre-compute features for efficiency.
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.vla import Gemma4VLA, Gemma4VLAConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: Gemma4VLAConfig,
        output_dir: str = "checkpoints",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 50_000,
        save_every: int = 5000,
        log_every: int = 100,
        device: str = "cuda",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.save_every = save_every
        self.log_every = log_every
        self.device = device

        # Model
        self.model = Gemma4VLA(config)
        self.model.load_backbone(device=device)
        self.model.action_expert.to(device=device, dtype=config.dtype)

        # Optimizer (only action expert params)
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=max_steps,
            pct_start=warmup_steps / max_steps,
        )

        logger.info(f"Trainable params: {self.model.num_trainable_params()/1e6:.1f}M")

    def train(self, dataloader: DataLoader):
        """Main training loop.

        Expected batch format from dataloader:
            {
                "actions": (B, H, action_dim),       — target action chunks
                "state": (B, action_dim),             — proprioception
                "context": (B, S, backbone_hidden),   — pre-computed backbone features
            }

        For efficiency, backbone features should be pre-computed offline
        and stored in the dataset, so training only runs the action expert.
        """
        self.model.action_expert.train()
        step = 0

        while step < self.max_steps:
            for batch in dataloader:
                if step >= self.max_steps:
                    break

                actions = batch["actions"].to(self.device, dtype=self.config.dtype)
                state = batch["state"].to(self.device, dtype=self.config.dtype)
                context = batch["context"].to(self.device, dtype=self.config.dtype)

                loss = self.model.compute_loss(actions, state, context)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                if step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"step={step} loss={loss.item():.4f} lr={lr:.2e}")

                if step > 0 and step % self.save_every == 0:
                    self._save_checkpoint(step)

                step += 1

        self._save_checkpoint(step)
        logger.info(f"Training complete. {step} steps.")

    def _save_checkpoint(self, step: int):
        path = self.output_dir / f"action_expert_step{step}.pt"
        torch.save(self.model.action_expert.state_dict(), path)
        logger.info(f"Saved checkpoint: {path}")
