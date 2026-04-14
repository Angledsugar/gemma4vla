# Gemma 4 + π0.6-style Action Expert VLA

Gemma 4 E4B backbone + π0.6-aligned action expert for robotic manipulation.
42-layer shared attention, flow matching + FAST, Knowledge Insulation training.

## Architecture

```
Gemma 4 E4B (8B, frozen)              Action Expert (~834M, trainable)
┌────────────────────────┐             ┌────────────────────────┐
│ SigLIP Vision (768d)   │             │ 42 layers              │
│ Gemma 4 Decoder        │             │ width=896              │
│ 42 layers, width=2560  │── shared ───│ 35 sliding (hd=256)    │
│ 35 sliding (hd=256)    │  attention  │  7 global  (hd=512)    │
│  7 global  (hd=512)    │             │ adaRMS + SwiGLU        │
│ 448×448 images, 4 cams │             │ flow matching 5-step   │
└────────────────────────┘             └────────────────────────┘
```

### π0.6 alignment

| Item | π0.6 | Ours |
|------|------|------|
| Backbone | Gemma 3 4B | **Gemma 4 E4B** |
| Expert depth | = backbone layers | **42 (= Gemma 4)** |
| Expert params | ~860M | **~834M** |
| Shared attention | ✅ | ✅ (all 42 layers) |
| Sliding + global | Gemma 3 pattern | **Gemma 4 pattern** |
| Flow + FAST | ✅ | ✅ |
| Knowledge Insulation | ✅ | ✅ |
| Image resolution | 448×448 | 448×448 |
| Denoise steps | 5 | 5 |

## Quick Start

### 1. Install

```bash
cd ~/project/gemma4vla
uv sync
uv pip install pyarrow pandas bitsandbytes
```

### 2. Download data

```bash
# DROID 100 episodes (small, for testing)
huggingface-cli download lerobot/droid_100 --repo-type dataset \
    --local-dir /media/engineer/DATA/datasets/droid_100

# Full DROID (76k episodes, ~1.8TB — optional)
huggingface-cli download lerobot/droid --repo-type dataset \
    --local-dir /media/engineer/DATA/datasets/droid
```

### 3. Train

```bash
# GPU 0: Train with real Gemma 4 backbone + DROID data
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py \
    --data_dir /media/engineer/DATA/datasets/droid_100 \
    --steps 1000 \
    --batch_size 1 \
    --lr 1e-4 \
    --output_dir /media/engineer/DATA/checkpoints/gemma4vla

# With Knowledge Insulation (π0.6 style, recommended for longer training)
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py \
    --data_dir /media/engineer/DATA/datasets/droid_100 \
    --steps 10000 \
    --batch_size 1 \
    --knowledge_insulation \
    --output_dir /media/engineer/DATA/checkpoints/gemma4vla

# Dummy mode (CPU, shape testing only)
uv run scripts/train_droid.py --dummy --steps 5 --device cpu
```

### 4. Serve (for Isaac Sim evaluation)

```bash
# Start policy server (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve.py \
    --checkpoint /media/engineer/DATA/checkpoints/gemma4vla/expert_step1000.pt

# Or from openpi venv (original π0.5 for comparison)
cd ~/project/openpi
CUDA_VISIBLE_DEVICES=0 uv run python ~/project/gemma4vla/scripts/serve.py \
    --no-gemma4
```

### 5. Evaluate in Isaac Sim (DROID environment)

```bash
# Terminal 1: Policy server (above)

# Terminal 2: Isaac Sim DROID environment (GPU 1)
cd ~/project/isaacsim-raise
python3 ../raise-vla/scripts/standalone/standalone_droid_vla.py \
    --scene 1 --gpu 1 --debug

# Normal mode (automatic control loop)
python3 ../raise-vla/scripts/standalone/standalone_droid_vla.py \
    --scene 1 --gpu 1 --action_mode velocity
```

### 6. Run tests

```bash
cd ~/project/gemma4vla
uv run python -m pytest tests/ -v
# 20 tests: shapes, gradients, FAST tokenizer, knowledge insulation
```

## Training Details

### Knowledge Insulation (π0.6 method)

```
Path 1 (Backbone): FAST token prediction → cross-entropy loss
    → gradient updates backbone only
    → teaches backbone to understand robot actions

Path 2 (Expert):   Flow matching → MSE loss on velocity field  
    → gradient updates expert only (context.detach())
    → backbone features are read but not corrupted

Total: backbone_loss + expert_loss (separate optimizers, separate LR)
```

### Memory Usage (RTX 3090 24GB)

| Component | VRAM |
|-----------|------|
| Gemma 4 backbone (frozen, bf16) | ~16GB |
| Action expert (bf16) | ~1.7GB |
| Optimizer states (8-bit AdamW) | ~1.7GB |
| Activations + buffers | ~3GB |
| **Total** | **~22GB** |

### Expected Training Metrics

```
step=1     loss=1.18   (random init)
step=100   loss=1.02   (learning)
step=500   loss=0.92   (converging)
step=1000  loss=0.85   (100 episodes)
```

## Project Structure

```
gemma4vla/
├── gemma4vla/
│   ├── model/
│   │   ├── config.py              # π0.6-aligned config
│   │   ├── action_expert.py       # 3 variants: shared, cross-attn, mlp
│   │   ├── backbone_gemma4.py     # Real Gemma 4 E4B backbone
│   │   ├── fast_tokenizer.py      # FAST action tokenizer
│   │   ├── flow_matching.py       # Flow matching loss + ODE sampler
│   │   ├── policy.py              # Full VLA policy
│   │   └── projector.py           # Vision/state projectors
│   ├── data/
│   │   ├── lerobot_v3.py          # LeRobot v3 loader (multi-dataset)
│   │   └── dummy_dataset.py       # Testing
│   ├── training/
│   │   ├── knowledge_insulation.py # KI trainer
│   │   └── loss.py                # Loss functions
│   └── serving/
│       └── server.py              # WebSocket server (OpenPI compatible)
├── scripts/
│   ├── train_droid.py             # DROID training entry point
│   ├── train.py                   # Dummy data training
│   └── serve.py                   # Policy serving
├── tests/                         # 20 tests (shapes, gradients, KI, FAST)
├── configs/                       # YAML configs
└── checkpoints/                   # Saved models
```

## Mixed Dataset Training

```python
from gemma4vla.data.lerobot_v3 import LeRobotV3MixedDataset

dataset = LeRobotV3MixedDataset([
    ("/media/engineer/DATA/datasets/droid_100", 0.5),
    ("/media/engineer/DATA/datasets/aloha_sim", 0.3),
    ("/media/engineer/DATA/datasets/bridge_v2", 0.2),
], action_horizon=15, action_dim=32)
```
