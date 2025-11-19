# Dreamer Literature Review

## DreamerV2 vs DreamerV3 vs Dreamer 4 Comparison

### Overview

| Aspect | DreamerV2 (2021) | DreamerV3 (2023) | Dreamer 4 (Sept 2025) |
|--------|------------------|------------------|------------------------|
| **Core Architecture** | RSSM (GRU-based) | RSSM (GRU-based) | **Transformer** + Causal Tokenizer |
| **World Model** | Recurrent state-space | Recurrent state-space | **Block-causal transformer** |
| **Training Paradigm** | Online RL + replay | Online RL + replay | **Offline-first** (learns from videos) |
| **Key Innovation** | Discrete latents | Robustness techniques | **Shortcut forcing** |

---

### Technical Details

| Feature | DreamerV2 | DreamerV3 | Dreamer 4 |
|---------|-----------|-----------|-----------|
| **Reward Handling** | Per-task scaling | Symlog (auto-scales) | Symlog inherited |
| **Hyperparameters** | Tuned per domain | Single fixed config | Single fixed config |
| **State Representation** | Discrete latents (32 categories) | Discrete + Unimix (1% uniform) | **Latent tokens** (causal tokenizer) |
| **Sequence Model** | GRU | GRU + LayerNorm | **Transformer** (axial attention) |
| **Sampling** | N/A (deterministic rollouts) | N/A | **Shortcut forcing** (4 steps vs 64 for diffusion) |
| **Action Conditioning** | Fully supervised | Fully supervised | **Semi-supervised** (100h labeled → 85% performance) |

---

### Performance

| Benchmark | DreamerV2 | DreamerV3 | Dreamer 4 |
|-----------|-----------|-----------|-----------|
| **Atari (26 games)** | Good (human-level) | +30% over V2 | N/A (different focus) |
| **Minecraft Diamonds** | Fails | First w/o human data (online) | First from **offline only** |
| **Craftax-Classic** | ~45% | ~53% | Not benchmarked yet |
| **Data Efficiency** | Baseline | ~2x V2 | **100x less data than VPT** |
| **Inference Speed** | Fast (GRU) | Fast (GRU) | **Real-time** (≥20 FPS on H100) |

---

### Architecture Comparison

```
DreamerV2/V3:
┌─────────┐     ┌──────┐     ┌────────┐
│ Encoder │ ──▶ │ RSSM │ ──▶ │ Decoder│
└─────────┘     │ (GRU)│     └────────┘
                └──────┘
                   ▼
              Actor-Critic

Dreamer 4:
┌───────────────┐     ┌─────────────────┐     ┌──────────┐
│ Causal        │ ──▶ │ Block-Causal    │ ──▶ │ Policy   │
│ Tokenizer     │     │ Transformer     │     │ Heads    │
└───────────────┘     │ (Axial Attn)    │     └──────────┘
                      │ + Shortcut      │
                      │   Forcing       │
                      └─────────────────┘
```

---

### Key Innovations Per Version

**DreamerV2** (ICLR 2021):
- Discrete latent representations
- First to match human on Atari via world models
- Introduced categorical distributions for state
- Paper: https://arxiv.org/abs/2010.02193

**DreamerV3** (Nature 2025):
- Symlog transformations (handles any reward scale)
- Two-hot critic (255 bins)
- Unimix categoricals (1% uniform mixing)
- Works out-of-the-box on 150+ tasks
- Paper: https://arxiv.org/abs/2301.04104

**Dreamer 4** (Sept 2025):
- **Transformer replaces GRU** (better long-range dependencies)
- **Shortcut forcing**: 16x faster than diffusion sampling
- **Learns from unlabeled videos** (minimal action labels needed)
- **Axial attention**: O(n) instead of O(n²) for video
- **Sparse temporal attention**: Every 4 layers only
- Paper: https://arxiv.org/abs/2509.24527
- Project: https://danijar.com/project/dreamer4/

---

## DreamerV3 Technical Deep Dive

### Core Technical Changes from V2

1. **Symlog Predictions** (Most Important)
   - `symlog(x) = sign(x) * ln(|x| + 1)`
   - Compresses large values, preserves small ones
   - Applied to: decoder, rewards, critic, encoder inputs
   - **Why it matters**: Works on Atari (rewards: 0-100) AND Minecraft (rewards: 0-10000) without changes

2. **Two-Hot Critic**
   - DreamerV2: Predicts single return value
   - DreamerV3: Predicts distribution over 255 bins
   - **Why it matters**: Handles multimodal return distributions better

3. **Unimix Categoricals**
   - 1% uniform + 99% network output for categorical distributions
   - Prevents KL spikes and overconfident predictions
   - More stable training

4. **Return Normalization**
   - Normalizes by 5th-95th percentile range (exponential moving average)
   - Automatically adjusts to reward scale
   - Prefers downscaling large rewards over upscaling small ones

5. **KL Regularization Changes**
   - Split into "dynamic loss" (0.5) and "representation loss" (0.1)
   - Free bits clamping to prevent degenerate solutions

6. **Architecture Polish**
   - SiLU activation (smoother gradients)
   - LayerNorm on final dimensions
   - Xavier normal initialization

---

## Craftax Context

### Why Craftax Was Created

**The core problem**: Original Crafter (Python-native) was too slow.
- CPU-bound simulation
- Limited to ~1M environment steps
- Required industrial compute resources

**Craftax solution**:
- Written entirely in JAX
- 250-257x faster than Crafter
- Enables 4096+ parallel environment workers
- 1B step PPO run takes <1 hour on single GPU

### Craftax Variants

| Feature | Crafter | Craftax-Classic | Craftax (Full) |
|---------|---------|-----------------|----------------|
| Speed | Baseline | 257x faster | 169x faster |
| Mechanics | Basic survival/crafting | Same as Crafter | Extended (NetHack-inspired) |
| Depth | Single layer world | Same | Multiple dungeon floors |
| Combat | Basic | Same | Enhanced enemy types, behaviors |
| Items | Basic | Same | Potions, enchantments, magic |
| Achievements | 22 | 22 | 65 across 4 difficulty tiers |

### Benchmark Results

**Craftax-1B (1 billion steps)**:
- PPO-GTrXL: 18.3%
- PQN-RNN: 16.0%
- PPO-RNN: 15.3%
- RND: 12.0%
- PPO: 11.9%

**Craftax-1M (1 million steps)**:
- Simulus: 6.6%
- Efficient MBRL: 5.4%
- PPO-RNN: 2.3%

---

## Implementation Options for Craftax

### DreamerV2 JAX (Recommended Starting Point)
- Repository: https://github.com/kenjyoung/dreamerv2_JAX
- Has gymnax wrappers (Craftax uses gymnax interface)
- Multi-seed training on single GPU
- Tested on A100 GPUs
- Dependencies: JAX, Haiku, NumPy, gymnax, gym, wandb

### DreamerV3 Official
- Repository: https://github.com/danijar/dreamerv3
- No native gymnax support (needs wrapper)
- Python 3.11+ required
- Better performance (~53% on Craftax-classic vs ~45% for V2)

### Craftax Baselines (Alternative)
- Repository: https://github.com/MichaelTMatthews/Craftax_Baselines
- PPO, PPO-RNN, RND implementations
- JAX-native, documented
- Good for understanding environment first

---

## Hardware Requirements

### GPU Requirements
- DreamerV3 200M model: RTX 3090 (24GB) had memory issues
- DreamerV2 JAX benchmarked on A100 (40-80GB VRAM)
- Craftax PPO: RTX 4090 + i9-13900K (20k+ steps/second)

### Apple Silicon (M3 Mac)
- JAX-Metal is experimental
- No float64 support
- 3-10x slower than NVIDIA GPUs
- OK for testing, not for serious training
- 36GB unified memory helps with model size but not throughput

### Recommended Setup
- Local (M3): Small test runs, debugging, code development
- Cluster (A100/V100/RTX 4090): Actual training runs
- Slurm job submission for cluster work

---

## References

### Papers
- DreamerV2: "Mastering Atari with Discrete World Models" (ICLR 2021)
- DreamerV3: "Mastering Diverse Domains through World Models" (Nature 2025)
- Dreamer 4: "Training Agents Inside of Scalable World Models" (arXiv 2025)
- Craftax: "A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning" (ICML 2024)

### Code Repositories
- Craftax: https://github.com/MichaelTMatthews/Craftax
- Craftax Baselines: https://github.com/MichaelTMatthews/Craftax_Baselines
- DreamerV3 Official: https://github.com/danijar/dreamerv3
- DreamerV2 JAX: https://github.com/kenjyoung/dreamerv2_JAX
- Gymnax: https://github.com/RobertTLange/gymnax

### Useful Resources
- Craftax Blog: https://craftaxenv.github.io/
- Craftax Paper: https://arxiv.org/abs/2402.16801
- DreamerV3 Explained: https://eclecticsheep.ai/2023/08/10/dreamer_v3.html
- Dreamer 4 Project: https://danijar.com/project/dreamer4/
