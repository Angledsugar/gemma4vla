"""Loss functions for VLA training.

Primary: Flow matching MSE loss on velocity prediction.
Optional: Auxiliary losses for stability.
"""

import torch
import torch.nn.functional as F


def flow_matching_loss(
    predicted_velocity: torch.Tensor,  # (B, H, D)
    target_velocity: torch.Tensor,     # (B, H, D)
    mask: torch.Tensor = None,         # (B, H) optional per-step mask
) -> torch.Tensor:
    """MSE loss between predicted and target velocity fields.

    Following π0.5: loss = ||v_θ(x_t, t) - (noise - actions)||²
    """
    loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")  # (B, H, D)

    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
        return loss.sum() / mask.sum().clamp(min=1) / loss.shape[-1]

    return loss.mean()


def action_chunk_smoothness_loss(actions: torch.Tensor) -> torch.Tensor:
    """Optional regularizer: penalize jerky action sequences.

    Encourages smooth trajectories by penalizing acceleration (second derivative).
    """
    if actions.shape[1] < 3:
        return torch.tensor(0.0, device=actions.device)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return accel.pow(2).mean()
