"""Knowledge Insulation training strategy — π0.6 aligned.

π0.6 model card:
  "The model is trained with Knowledge Insulation: the vision-language
   backbone predicts FAST action tokens and co-training examples, such
   as multi-modal web data. The action expert predicts continuous actions,
   and the gradient from the action expert does not flow back to the
   main VLM backbone."

Two gradient paths:
  Path 1 (Backbone): FAST token prediction + optional web co-training
    → gradient flows into backbone only
  Path 2 (Action Expert): Flow matching on continuous actions
    → gradient flows into action expert only (context.detach())

This replaces the simple discrete binning head from earlier versions
with the actual FAST tokenizer used in π0.6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.fast_tokenizer import FASTActionTokenizer, FASTActionHead, ActionTokenSequence


class ActionTokenizer:
    """Simple binning tokenizer (fallback when FAST unavailable)."""

    def __init__(self, num_bins: int = 256, action_low: float = -1.0, action_high: float = 1.0):
        self.num_bins = num_bins
        self.action_low = action_low
        self.action_high = action_high

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        normed = (actions - self.action_low) / (self.action_high - self.action_low)
        normed = normed.clamp(0, 1 - 1e-6)
        return (normed * self.num_bins).long()


class KnowledgeInsulationTrainer:
    """π0.6-style Knowledge Insulation training.

    Backbone path: predicts FAST action tokens (cross-entropy loss)
    Expert path:   predicts continuous actions via flow matching (MSE loss)
    Gradients are isolated via stop-gradient (context.detach()).
    """

    def __init__(
        self,
        policy,
        lr_backbone: float = 1e-5,
        lr_expert: float = 1e-4,
        use_fast_backbone: bool = True,
        num_bins: int = 256,
        expert_loss_weight: float = 1.0,
        device: str = "cuda",
    ):
        self.policy = policy
        self.device = device
        self.expert_loss_weight = expert_loss_weight
        self.use_fast_backbone = use_fast_backbone

        config = policy.config

        if use_fast_backbone:
            # π0.6 style: backbone predicts FAST tokens
            self.fast_head = FASTActionHead(
                hidden_size=config.backbone.hidden_size,
                vocab_size=config.backbone.vocab_size,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
            ).to(device)
            self.fast_tokenizer = FASTActionTokenizer()
            self.fast_sequence = ActionTokenSequence(
                fast_tokenizer=self.fast_tokenizer,
                vocab_size=config.backbone.vocab_size,
                max_seq_len=config.fast_max_seq_len,
            )
        else:
            # Fallback: simple discrete binning
            self.fast_head = None
            self.binning_head = nn.Sequential(
                nn.Linear(config.backbone.hidden_size, config.backbone.hidden_size),
                nn.SiLU(),
                nn.Linear(config.backbone.hidden_size, config.action_horizon * config.action_dim * num_bins),
            ).to(device)
            self.tokenizer = ActionTokenizer(num_bins=num_bins)
            self.num_bins = num_bins

        # Separate optimizers (π0.6: separate gradient paths)
        backbone_params = self._get_backbone_params()
        expert_params = list(policy.action_expert.parameters())

        self.optimizer_backbone = torch.optim.AdamW(backbone_params, lr=lr_backbone, weight_decay=0.01)
        self.optimizer_expert = torch.optim.AdamW(expert_params, lr=lr_expert, weight_decay=0.01)

    def _get_backbone_params(self):
        """Collect backbone + head parameters."""
        params = []
        if self.policy.use_dummy:
            params += list(self.policy.backbone.parameters())
            params += list(self.policy.vision_encoder.parameters())
            params += list(self.policy.vision_projector.parameters())
        # Don't add real backbone params (frozen)

        if self.fast_head is not None:
            params += list(self.fast_head.parameters())
        elif hasattr(self, 'binning_head'):
            params += list(self.binning_head.parameters())
        return params

    def train_step(
        self,
        images: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        state: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> dict:
        """One training step with Knowledge Insulation.

        Path 1: Backbone → FAST token prediction
        Path 2: Expert → flow matching (context detached)
        """
        self.policy.train()

        # ============ Forward through backbone ============
        context = self.policy.encode_observation(images, lang_tokens)

        # ============ Path 1: Backbone loss (FAST or binning) ============
        if self.use_fast_backbone and self.fast_head is not None:
            # π0.6 style: FAST token prediction
            # Build target FAST token sequence from actions
            import numpy as np
            B = target_actions.shape[0]
            backbone_loss = torch.tensor(0.0, device=self.device)

            for b in range(B):
                actions_np = target_actions[b].detach().cpu().numpy()
                prefix = np.array([1], dtype=np.int64)  # minimal prefix
                seq = self.fast_sequence.build_training_sequence(prefix, actions_np)

                target_tokens = torch.tensor(seq["tokens"], device=self.device).unsqueeze(0)
                loss_mask = torch.tensor(seq["loss_mask"], device=self.device).unsqueeze(0)

                # Use backbone hidden states for FAST prediction
                # Truncate/pad context to match sequence length
                ctx = context[b:b+1]
                S = ctx.shape[1]
                T = target_tokens.shape[1]
                if S < T:
                    ctx = F.pad(ctx, (0, 0, 0, T - S))
                else:
                    ctx = ctx[:, :T]

                backbone_loss = backbone_loss + self.fast_head.compute_loss(ctx, target_tokens, loss_mask)

            backbone_loss = backbone_loss / B
        else:
            # Fallback: simple binning
            action_tokens = self.tokenizer.encode(target_actions)
            logits = self.binning_head(context.mean(dim=1))
            logits = logits.view(-1, self.policy.config.action_horizon,
                                self.policy.config.action_dim, self.num_bins)
            backbone_loss = F.cross_entropy(
                logits.reshape(-1, self.num_bins),
                action_tokens.reshape(-1),
            )

        # ============ Path 2: Expert loss (flow matching, detached) ============
        context_detached = context.detach()
        expert_loss = self.policy.flow_loss(
            predict_velocity_fn=self.policy.action_expert,
            target_actions=target_actions,
            state=state,
            context=context_detached,
        )

        # ============ Backward (separate) ============
        self.optimizer_backbone.zero_grad()
        self.optimizer_expert.zero_grad()

        backbone_loss.backward(retain_graph=True)
        (self.expert_loss_weight * expert_loss).backward()

        torch.nn.utils.clip_grad_norm_(self._get_backbone_params(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy.action_expert.parameters(), 1.0)

        self.optimizer_backbone.step()
        self.optimizer_expert.step()

        return {
            "backbone_loss": backbone_loss.item(),
            "expert_loss": expert_loss.item(),
            "total_loss": backbone_loss.item() + self.expert_loss_weight * expert_loss.item(),
        }
