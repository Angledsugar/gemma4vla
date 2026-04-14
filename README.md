# Gemma4VLA

## Gemma 4 E4B + π style action expert 
Gemma 4 E4B backbone + π style action expert for robotic manipulation.
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

### Comparison with π0 / π0.5 / π0.6

| Item | π0 | π0.5 | π0.6 | Ours |
|------|-----|------|------|------|
| Backbone | PaliGemma 2B | PaliGemma 2B | Gemma 3 4B | **Gemma 4 E4B 8B** |
| Vision encoder | SigLIP 400M | SigLIP 400M | SigLIP 400M | **SigLIP (Gemma 4)** |
| Expert architecture | Flow matching | Flow matching | Flow + FAST | **Flow + FAST** |
| Expert depth | < backbone | < backbone | = backbone | **42 (= backbone)** |
| Expert params | ~300M | ~300M | ~860M | **~834M** |
| Shared attention | ✅ | ✅ | ✅ | **✅ (all 42 layers)** |
| Attention pattern | Gemma 2 | Gemma 2 | Gemma 3 sliding+global | **Gemma 4 sliding+global** |
| Knowledge Insulation | - | - | ✅ | **✅** |
| FAST tokenizer | - | - | ✅ (backbone path) | **✅ (backbone path)** |
| Training-Time RTC | - | - | - | **✅** |
| Image resolution | 224x224 | 224x224 | 448x448 | **448x448** |
| Max cameras | 1 | 1 | 4 | **4** |
| Denoise steps | 10 | 10 | 5 | **5** |
| Action representation | Continuous only | Continuous only | Continuous + discrete | **Continuous + discrete** |

**Key differences from π0 / π0.5**:
- **Backbone scale**: π0/π0.5 use PaliGemma 2B. We use Gemma 4 E4B (8B), which brings stronger vision-language grounding.
- **Expert depth**: π0/π0.5 use a shallower expert. Following π0.6, our expert has the same depth as the backbone (42 layers), enabling shared attention at every layer.
- **Knowledge Insulation**: π0/π0.5 train backbone and expert jointly. π0.6 and ours isolate gradients — backbone learns FAST tokens (discrete), expert learns flow matching (continuous), preventing backbone degradation.
- **FAST tokenizer**: π0/π0.5 only predict continuous actions. π0.6 and ours add discrete FAST token prediction as an auxiliary backbone objective.

**Key differences from π0.6**:
- **Backbone**: π0.6 uses Gemma 3 4B. We use Gemma 4 E4B (8B), a newer and larger backbone with improved vision-language capabilities.
- **Attention pattern**: π0.6 follows Gemma 3's sliding/global pattern. We follow Gemma 4's pattern (35 sliding layers with head_dim=256, 7 global layers with head_dim=512).
- **Training-Time RTC**: We add RTC (arXiv:2512.05964) for smooth action chunk transitions, not present in π0.6.

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
# Base model: multi-dataset from YAML config (recommended)
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_base.py --config configs/base_mix.yaml

# Base model: CLI로 직접 데이터셋 지정
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_base.py \
    --datasets lerobot/droid_100:0.5 lerobot/pusht:0.3 \
    --steps 50000 --rtc --knowledge_insulation --normalize

# DROID 단일 fine-tuning
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py \
    --data_dir /media/engineer/DATA/datasets/droid_100 \
    --steps 10000 --rtc --knowledge_insulation

# Dummy mode (CPU, shape testing only)
uv run scripts/train_base.py --datasets lerobot/pusht --dummy --device cpu --steps 5
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

## Training Pipeline

### Overview

Knowledge Insulation + RTC + FAST 전체가 기본 학습 파이프라인입니다. Gemma 4 backbone (8B)은 frozen 상태에서 두 개의 분리된 gradient path로 학습합니다.

```
┌──────────────────────────────────────────────────────────────────┐
│ Training Step (Knowledge Insulation + RTC)                       │
│                                                                  │
│  Images + Text ──→ [Gemma 4 backbone (frozen)] ──→ context       │
│                                                                  │
│  ┌─── Path 1: Backbone ──────────────────────────────────────┐   │
│  │  actions → FAST encode → discrete tokens                  │   │
│  │  context → FASTActionHead → predicted tokens              │   │
│  │  Loss = CrossEntropy (action token positions only)        │   │
│  │  → gradient updates backbone head only                    │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─── Path 2: Action Expert (with RTC) ──────────────────────┐   │
│  │  context.detach() (stop gradient to backbone)             │   │
│  │                                                           │   │
│  │  RTC split: [prefix d steps | postfix H-d steps]         │   │
│  │    prefix:  ground-truth actions, τ=1 (no noise)         │   │
│  │    postfix: x_t = (1-t)·x_0 + t·x_1, τ~U(0,1)          │   │
│  │                                                           │   │
│  │  context + x_t + per-token τ → [Action Expert]            │   │
│  │                                → predicted velocity v_θ   │   │
│  │  Loss = MSE(v_θ, x_1 - x_0) on postfix only             │   │
│  │  → gradient updates expert only                           │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Total: backbone_loss + expert_loss (separate optimizers)        │
└──────────────────────────────────────────────────────────────────┘
```

### Default training (full pipeline)

base 모델은 모든 로봇 데이터셋을 혼합 학습합니다. `train_base.py`가 기본 학습 스크립트이며, YAML config로 데이터셋 구성을 관리합니다.

```bash
# YAML config로 multi-dataset 학습 (권장)
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_base.py --config configs/base_mix.yaml

# CLI로 직접 지정
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_base.py \
    --datasets lerobot/droid_100:0.4 lerobot/aloha_sim_insertion_human:0.15 \
               lerobot/pusht:0.1 lerobot/unitreeh1_fold_clothes:0.1 \
    --steps 50000 \
    --rtc --rtc_max_delay 4 \
    --knowledge_insulation \
    --lr 1e-4 --lr_backbone 1e-5 \
    --normalize
```

`configs/base_mix.yaml` 예시:

```yaml
datasets:
  - repo_id: lerobot/droid_100
    weight: 0.4
  - repo_id: lerobot/aloha_sim_insertion_human
    weight: 0.15
  - repo_id: lerobot/aloha_sim_transfer_cube_human
    weight: 0.15
  - repo_id: lerobot/pusht
    weight: 0.1
  - repo_id: lerobot/unitreeh1_fold_clothes
    weight: 0.1
  - repo_id: lerobot/columbia_cairlab_pusht_real
    weight: 0.1

action_dim: 32
action_horizon: 15

training:
  steps: 50000
  batch_size: 2
  lr: 1e-4
  lr_backbone: 1e-5
  knowledge_insulation: true
  rtc: true
  rtc_max_delay: 4
  normalize: true
```

- **Multi-dataset**: `UniversalLeRobotDataset`이 서로 다른 로봇의 action/state 차원을 공통 차원(32)으로 패딩하여 통합. `WeightedRandomSampler`로 데이터셋별 샘플링 비율 제어.
- **Knowledge Insulation**: backbone은 FAST 토큰 예측 (cross-entropy, lr=1e-5), expert는 flow matching (MSE, lr=1e-4). `context.detach()`로 gradient 분리.
- **RTC**: 매 학습 step에서 prefix 길이 d를 {0,1,2,3,4}에서 샘플링 (P(d) ~ 0.5^d). prefix는 ground-truth + τ=1, postfix만 loss 계산.
- **FAST**: backbone이 action을 이해하도록 discrete token prediction을 auxiliary objective로 사용.
- **Auto-download**: HuggingFace Hub에서 자동 다운로드. 로컬 캐시 경로는 `cache_dir`로 지정.

### Training scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_base.py` | **Base 모델 학습** (multi-dataset, YAML config) |
| `scripts/train_droid.py` | DROID 단일 데이터셋 학습 (fine-tuning, 빠른 실험용) |
| `scripts/train.py` | Dummy data shape 테스트 (GPU 불필요) |

### Simplified modes

개별 기능만 사용하거나 단일 데이터셋으로 빠른 실험을 할 수 있습니다.

```bash
# 단일 데이터셋 (DROID만)
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_base.py \
    --datasets lerobot/droid_100 --steps 1000

# Expert-only (flow matching만, KI/RTC 없음)
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_base.py \
    --datasets lerobot/droid_100 --steps 1000 --lr 1e-4

# DROID fine-tuning
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_droid.py \
    --data_dir /path/to/droid_100 --steps 10000 \
    --rtc --knowledge_insulation

# Dummy mode (CPU, shape testing only)
uv run scripts/train_base.py --datasets lerobot/pusht --dummy --device cpu --steps 5
```

### Optimizer and scheduler

- **Optimizer**: 8-bit AdamW (bitsandbytes) — saves ~5GB VRAM compared to standard AdamW
- **Scheduler**: Linear warmup (default 1000 steps) + cosine decay
- **Gradient clipping**: max_norm=1.0
- **Normalization**: Per-dimension z-score normalization on actions/states (`--normalize`)

### Checkpointing

Expert weights are saved every `--save_every` steps (default 500) and at the end of training. Only expert state_dict is saved (not the frozen backbone).

```bash
# Resume from checkpoint
uv run scripts/train_droid.py --resume checkpoints/droid/expert_step500.pt --steps 2000
```

## Inference Pipeline

### Overview

Inference generates an action chunk by integrating the learned velocity field from noise to actions via Euler ODE integration (5 steps by default).

```
┌─────────────────────────────────────────────────────────────┐
│ Inference (5-step ODE)                                      │
│                                                             │
│  Images + Text ──→ [Gemma 4 backbone] ──→ context           │
│                                                             │
│  x_0 ~ N(0,1)  (random noise, shape = action chunk)        │
│                                                             │
│  for t in [0.0, 0.2, 0.4, 0.6, 0.8]:                       │
│      v = ActionExpert(x_t, t, context)                      │
│      x_{t+dt} = x_t + dt * v          (Euler step)         │
│                                                             │
│  x_1 = predicted action chunk  (B, 15, action_dim)          │
└─────────────────────────────────────────────────────────────┘
```

### Inference with RTC

At inference, the last d steps from the previous action chunk become the prefix of the next chunk. Prefix tokens are held fixed (not denoised) and receive per-token timestep τ=1.0. Only postfix tokens are integrated from noise.

```
Previous chunk: [a_1 ... a_11, a_12, a_13, a_14, a_15]
                                └─── prefix (d=4) ───┘
                                         │
Next chunk:     [a_12, a_13, a_14, a_15, â_5, â_6, ..., â_15]
                 ├── fixed (from prev) ──┤ ├── denoised ──┤
                 τ = 1.0                   τ = 0 → 1
```

This ensures smooth temporal transitions between consecutive action chunks, avoiding jerky behavior at chunk boundaries.

### Serving for Isaac Sim

The policy runs as a WebSocket server (OpenPI protocol compatible). Existing Isaac Sim clients connect without modification.

```bash
# Gemma4VLA mode (trained expert + Gemma 4 backbone)
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve.py \
    --expert-checkpoint /path/to/expert_step1000.pt

# π0.5 PyTorch mode (for comparison, requires openpi venv)
cd ~/project/openpi
CUDA_VISIBLE_DEVICES=0 uv run python ~/project/gemma4vla/scripts/serve.py --no-gemma4

# Health check
curl http://localhost:8000/healthz
```

The server accepts msgpack-encoded observations over WebSocket:

```
Client → Server: {
    "observation/exterior_image_1_left": np.ndarray (H, W, 3),
    "observation/wrist_image_left": np.ndarray (H, W, 3),
    "observation/joint_position": np.ndarray (7,),
    "observation/gripper_position": np.ndarray (1,),
    "prompt": str
}

Server → Client: {
    "actions": np.ndarray (15, 8),       # action chunk
    "policy_timing": {"infer_ms": float}  # latency
}
```

## Training Details

### Real-Time Chunking (RTC)

Training-time RTC (arXiv:2512.05964) for smooth action chunk transitions.
Action horizon H is split into prefix (d steps) and postfix (H-d steps):

```
Action chunk: [a_1, a_2, ..., a_d, a_{d+1}, ..., a_H]
               ├─── prefix ───┤  ├──── postfix ────┤
               ground-truth        flow matching
               τ = 1 (no noise)    τ ~ U(0,1)
               loss = 0            loss = MSE
```

- **Training**: prefix tokens get ground-truth actions with τ=1, postfix tokens get standard noised interpolation. Loss is computed only on postfix. Per-token timestep conditioning via AdaRMSNorm.
- **Inference**: previous chunk's last d actions become the next chunk's prefix (held fixed during ODE integration), ensuring temporal consistency.

```python
# Training with RTC (prefix_len=4)
loss = flow_loss(expert, actions, state, context, prefix_len=4)

# Inference with RTC prefix
actions = sampler.sample(expert, shape, state, context,
                         action_prefix=prev_actions[:, -4:])
```

### FAST Action Tokenizer (π0-FAST)

FAST (Fast Action STructured Tokenizer) converts continuous actions into discrete tokens for autoregressive prediction by the backbone.

```
Encode: actions (H, D) → DCT → quantize → token sequence
Decode: token sequence → dequantize → inverse DCT → actions (H, D)
```

- Uses pretrained HuggingFace tokenizer (`physical-intelligence/fast`)
- Action tokens are mapped to the high end of the backbone vocabulary
- Attention: prefix (instruction + state) = bidirectional, action tokens = causal
- Loss: cross-entropy on action token positions only

Used in Knowledge Insulation where the backbone predicts FAST tokens (Path 1) while the action expert predicts continuous actions via flow matching (Path 2).

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

