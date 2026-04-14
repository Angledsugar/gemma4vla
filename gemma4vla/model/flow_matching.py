"""Conditional Flow Matching for action generation.

Implements the flow matching training objective and ODE sampling
used in π0/π0.5 for continuous action chunk generation.

Flow: x_t = (1-t) * x_0 + t * x_1
      where x_0 ~ N(0,1), x_1 = target action, t ∈ [0,1]
Velocity: v = x_1 - x_0  (constant velocity straight-line flow)
Loss: ||v_θ(x_t, t) - v||²

Training-Time RTC (arXiv:2512.05964):
  Prefix tokens (d steps) use ground-truth actions with τ=1 (no noise).
  Postfix tokens (H-d steps) use standard flow matching.
  Loss is only computed on postfix tokens.
  Per-token timestep conditioning via AdaLN-zero.
"""

import torch
import torch.nn as nn


class FlowMatchingLoss(nn.Module):
    """Conditional flow matching training loss with optional RTC."""

    def forward(
        self,
        predict_velocity_fn,
        target_actions: torch.Tensor,   # (B, H, D) — ground truth actions
        state: torch.Tensor,            # (B, D) — proprioception
        context: torch.Tensor,          # (B, S, C) — backbone features
        prefix_len: int = 0,            # RTC: number of prefix steps (d)
    ) -> torch.Tensor:
        B, H, D = target_actions.shape
        device = target_actions.device

        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # Sample noise x_0 ~ N(0, 1)
        x_0 = torch.randn_like(target_actions)

        # Target velocity: v = x_1 - x_0
        x_1 = target_actions
        velocity_target = x_1 - x_0

        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        t_expand = t[:, None, None]  # (B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        if prefix_len > 0:
            # Training-Time RTC: per-token timestep
            # Prefix: use ground-truth actions, τ=1
            # Postfix: use noised actions, τ=sampled
            x_t[:, :prefix_len] = target_actions[:, :prefix_len]

            # Per-token timestep: (B, H)
            per_token_t = t[:, None].expand(B, H).clone()  # default: sampled t
            per_token_t[:, :prefix_len] = 1.0  # prefix: τ=1 (fully denoised)

            # Predict velocity with per-token timestep
            velocity_pred = predict_velocity_fn(
                noised_actions=x_t,
                timestep=per_token_t,  # (B, H) per-token
                state=state,
                context=context,
            )

            # Masked loss: only on postfix
            postfix_pred = velocity_pred[:, prefix_len:]
            postfix_target = velocity_target[:, prefix_len:]
            loss = nn.functional.mse_loss(postfix_pred, postfix_target)
        else:
            # Standard flow matching (no RTC)
            velocity_pred = predict_velocity_fn(
                noised_actions=x_t,
                timestep=t,  # (B,) global
                state=state,
                context=context,
            )
            loss = nn.functional.mse_loss(velocity_pred, velocity_target)

        return loss


class FlowMatchingSampler:
    """ODE sampler for generating actions from noise via Euler integration.

    Supports RTC inference: condition on action prefix from previous chunk.
    """

    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        predict_velocity_fn,
        shape: tuple,              # (B, H, D)
        state: torch.Tensor,      # (B, D)
        context: torch.Tensor,    # (B, S, C)
        device: str = "cuda",
        action_prefix: torch.Tensor = None,  # (B, d, D) — RTC prefix
    ) -> torch.Tensor:
        """Generate action chunk by integrating the learned velocity field.

        Euler method: x_{t+dt} = x_t + dt * v_θ(x_t, t)

        With RTC: prefix tokens are held fixed (ground-truth from previous chunk),
        only postfix is denoised.
        """
        dtype = context.dtype  # match backbone/expert dtype
        dt = 1.0 / self.num_steps
        B, H, D = shape

        prefix_len = 0
        if action_prefix is not None:
            prefix_len = action_prefix.shape[1]

        # Start from noise (match dtype of model)
        x_t = torch.randn(shape, device=device, dtype=dtype)

        # Set prefix to ground-truth
        if prefix_len > 0:
            x_t[:, :prefix_len] = action_prefix.to(dtype)

        for i in range(self.num_steps):
            if prefix_len > 0:
                # Per-token timestep: prefix=1.0, postfix=current t
                per_token_t = torch.full((B, H), i * dt, device=device, dtype=dtype)
                per_token_t[:, :prefix_len] = 1.0
                velocity = predict_velocity_fn(
                    noised_actions=x_t,
                    timestep=per_token_t,
                    state=state,
                    context=context,
                )
                # Only update postfix
                x_t[:, prefix_len:] = x_t[:, prefix_len:] + dt * velocity[:, prefix_len:]
            else:
                t = torch.full((B,), i * dt, device=device, dtype=dtype)
                velocity = predict_velocity_fn(
                    noised_actions=x_t,
                    timestep=t,
                    state=state,
                    context=context,
                )
                x_t = x_t + dt * velocity

        return x_t  # (B, H, D) — denoised actions
