"""Minimal training loop for Gemma 4 Action Expert.

Trains with dummy data to verify:
1. Forward pass produces correct shapes
2. Loss computes and backpropagates
3. Optimizer updates parameters
4. All three expert variants work

Usage:
    cd ~/project/gemma4vla
    uv run scripts/train.py --variant flow_transformer --steps 50
    uv run scripts/train.py --variant flow_shared --steps 50
    uv run scripts/train.py --variant flow_mlp --steps 50
"""

import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from gemma4vla.model.config import Gemma4ActionExpertConfig
from gemma4vla.model.policy import Gemma4VLAPolicy
from gemma4vla.data.dummy_dataset import DummyDROIDDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="flow_transformer",
                        choices=["flow_transformer", "flow_shared", "flow_mlp"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action_dim", type=int, default=32)
    parser.add_argument("--action_horizon", type=int, default=15)
    parser.add_argument("--knowledge_insulation", action="store_true",
                        help="Use Knowledge Insulation training (separate backbone/expert gradients)")
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    args = parser.parse_args()

    # Config
    config = Gemma4ActionExpertConfig(
        variant=args.variant,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
    )
    logger.info(f"Expert variant: {config.variant}")
    logger.info(f"Expert width: {config.expert_width}, depth: {config.expert_depth}")
    logger.info(f"Backbone hidden_size: {config.backbone.hidden_size}")

    # Model
    model = Gemma4VLAPolicy(config).to(args.device)

    # Parameter summary
    summary = model.param_summary()
    for name, count in summary.items():
        logger.info(f"  {name}: {count/1e6:.1f}M params")

    # Training strategy
    ki_trainer = None
    if args.knowledge_insulation:
        from gemma4vla.training.knowledge_insulation import KnowledgeInsulationTrainer
        ki_trainer = KnowledgeInsulationTrainer(
            policy=model,
            lr_backbone=args.lr_backbone,
            lr_expert=args.lr,
            device=args.device,
        )
        logger.info("Using Knowledge Insulation training (7.5x faster convergence)")
        logger.info(f"  Backbone LR: {args.lr_backbone}, Expert LR: {args.lr}")
    else:
        logger.info("Using standard joint training")

    # Optimizer (standard mode: only action expert)
    optimizer = torch.optim.AdamW(model.get_action_expert_params(), lr=args.lr, weight_decay=0.01)

    # Data
    dataset = DummyDROIDDataset(
        num_samples=args.steps * args.batch_size,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0

    for batch in loader:
        if step >= args.steps:
            break

        images = [img.to(args.device) for img in batch["images"]]
        lang_tokens = batch["lang_tokens"].to(args.device)
        state = batch["state"].to(args.device)
        actions = batch["actions"].to(args.device)

        t0 = time.time()

        if ki_trainer is not None:
            # Knowledge Insulation: separate gradient paths
            losses = ki_trainer.train_step(images, lang_tokens, state, actions)
            loss_val = losses["total_loss"]
            dt = time.time() - t0
            total_loss += loss_val
            step += 1

            if step % 10 == 0 or step == 1:
                logger.info(
                    f"step={step}/{args.steps}  "
                    f"backbone={losses['backbone_loss']:.4f}  "
                    f"expert={losses['expert_loss']:.4f}  "
                    f"total={loss_val:.4f}  time={dt*1000:.0f}ms"
                )
        else:
            # Standard joint training
            loss = model.compute_loss(images, lang_tokens, state, actions)
            loss_val = loss.mean()

            optimizer.zero_grad()
            loss_val.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.get_action_expert_params(), 1.0)
            optimizer.step()
            dt = time.time() - t0

            total_loss += loss_val.item()
            step += 1

            if step % 10 == 0 or step == 1:
                avg_loss = total_loss / step
                logger.info(
                    f"step={step}/{args.steps}  loss={loss_val.item():.4f}  "
                    f"avg_loss={avg_loss:.4f}  grad_norm={grad_norm:.3f}  "
                    f"time={dt*1000:.0f}ms"
                )

    # Final inference test
    model.eval()
    with torch.no_grad():
        test_images = [torch.randn(1, 3, 224, 224, device=args.device)]
        test_tokens = torch.randint(0, 1000, (1, 16), device=args.device)
        test_state = torch.randn(1, args.action_dim, device=args.device)

        t0 = time.time()
        actions = model.predict_actions(test_images, test_tokens, test_state)
        infer_time = (time.time() - t0) * 1000

        logger.info(f"\nInference test:")
        logger.info(f"  Actions shape: {actions.shape}")
        logger.info(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
        logger.info(f"  Inference time: {infer_time:.0f}ms")

    logger.info(f"\nTraining complete. Final avg loss: {total_loss/step:.4f}")


if __name__ == "__main__":
    main()
