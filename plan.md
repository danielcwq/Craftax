# DreamerV2 + Craftax Implementation Plan

## The Big Picture: Repository Strategy

### Recommended Approach: **Separate Experiment Repository**

```
Your Structure:
craftax-dreamer/              # YOUR new repo
├── craftax/                  # Git submodule or pip install
├── dreamerv2_JAX/           # Git submodule or pip install
└── experiments/             # YOUR code lives here
    ├── configs/
    ├── train.py
    └── wrappers/
```

**Why this approach?**
- Craftax = environment (stays unchanged)
- DreamerV2 = algorithm (mostly unchanged)
- Your repo = glue code + experiments + configs
- Clean separation for version control
- Easy to swap components later

---

## Step-by-Step Setup Plan

### Phase 1: Local Development Setup (Week 1)

```bash
# 1. Create your experiment repo
mkdir ~/craftax-dreamer
cd ~/craftax-dreamer
git init

# 2. Set up environment with uv
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install craftax  # The environment
uv pip install jax flax optax gymnax wandb  # Core deps

# 4. Clone DreamerV2 JAX (as submodule or separate clone)
git clone https://github.com/kenjyoung/dreamerv2_JAX.git
# OR: git submodule add https://github.com/kenjyoung/dreamerv2_JAX.git
```

**Directory structure after setup:**

```
craftax-dreamer/
├── .venv/                          # Virtual environment
├── dreamerv2_JAX/                  # Cloned repo (read-only mostly)
│   ├── dreamer_single_seed.py
│   ├── environments.py
│   └── ...
├── experiments/                    # YOUR CODE
│   ├── __init__.py
│   ├── train.py                   # Main training script
│   ├── configs/
│   │   └── craftax_config.json   # Hyperparameters
│   ├── wrappers/
│   │   └── craftax_wrapper.py    # Adapt Craftax to DreamerV2 interface
│   └── utils/
│       └── logging.py
├── scripts/                        # Utility scripts
│   ├── test_env.py               # Verify Craftax works
│   └── visualize.py              # Render episodes
├── slurm/                          # Cluster jobs (later)
│   └── train_job.sh
├── pyproject.toml                  # Your dependencies
├── README.md
└── .gitignore
```

---

### Phase 2: Pre-Implementation Checklist

Before writing DreamerV2 code, verify these work:

#### Checkpoint 1: Craftax Works

```python
# scripts/test_env.py
import jax
from craftax import make_craftax_env_from_name

def test_craftax():
    """Verify Craftax environment works"""
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)

    print(f"Observation shape: {jax.tree_map(lambda x: x.shape, obs)}")
    print(f"Action space: {env.action_space(env_params)}")

    # Random rollout
    for _ in range(100):
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_action)
        obs, state, reward, done, info = env.step(rng_step, state, action, env_params)

    print("✓ Craftax works!")

if __name__ == "__main__":
    test_craftax()
```

Run: `python scripts/test_env.py`

---

#### Checkpoint 2: DreamerV2 Interface Understanding

```python
# scripts/test_dreamer.py
"""
Understand what DreamerV2 expects from environments
"""

# Look at dreamerv2_JAX/environments.py to see:
# - What methods are needed? (reset, step, observation_space, action_space)
# - What shapes?
# - How are episodes handled?

# Key question: Does DreamerV2 expect gym or gymnax interface?
# Answer: Check environments.py wrapper
```

---

#### Checkpoint 3: Environment Wrapper

This is the **critical glue code**:

```python
# experiments/wrappers/craftax_wrapper.py
"""
Adapts Craftax (gymnax) to DreamerV2's expected interface
"""
import jax
import jax.numpy as jnp
from craftax import make_craftax_env_from_name

class CraftaxToDreamer:
    """Wrapper to make Craftax compatible with DreamerV2"""

    def __init__(self, env_name="Craftax-Symbolic-v1"):
        self.env = make_craftax_env_from_name(env_name, auto_reset=True)
        self.env_params = self.env.default_params
        self._state = None

    def reset(self, rng):
        """DreamerV2 expects: obs"""
        obs, self._state = self.env.reset(rng, self.env_params)
        return self._process_obs(obs)

    def step(self, rng, action):
        """DreamerV2 expects: obs, reward, done"""
        obs, self._state, reward, done, info = self.env.step(
            rng, self._state, action, self.env_params
        )
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        """
        Transform Craftax obs to what DreamerV2 expects

        Craftax gives dict/pytree, DreamerV2 might expect:
        - Flat vector (for symbolic)
        - Image (for pixels)

        Check DreamerV2 code to see what format it wants!
        """
        # Example: flatten if needed
        if isinstance(obs, dict):
            # Flatten all arrays in dict
            flat_obs = jnp.concatenate([
                v.flatten() for v in jax.tree_leaves(obs)
            ])
            return flat_obs
        return obs

    @property
    def observation_space(self):
        """Return obs space in DreamerV2's expected format"""
        # TODO: Check what DreamerV2 needs
        pass

    @property
    def action_space(self):
        """Return action space"""
        return self.env.action_space(self.env_params)
```

**Key task**: Study `dreamerv2_JAX/environments.py` to see exactly what interface is expected, then implement wrapper accordingly.

---

### Phase 3: Minimal Training Script

```python
# experiments/train.py
"""
Minimal DreamerV2 + Craftax training script
"""
import jax
import json
from pathlib import Path

# Import DreamerV2 components (adjust path as needed)
import sys
sys.path.append(str(Path(__file__).parent.parent / "dreamerv2_JAX"))
from dreamer_single_seed import train  # Or whatever the entry point is

# Import your wrapper
from wrappers.craftax_wrapper import CraftaxToDreamer

def main():
    # 1. Load config
    config_path = Path(__file__).parent / "configs" / "craftax_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # 2. Create environment
    env = CraftaxToDreamer(env_name="Craftax-Symbolic-v1")

    # 3. Setup logging
    import wandb
    wandb.init(
        project="craftax-dreamer",
        config=config,
        name=f"run_{config['seed']}"
    )

    # 4. Run training (using DreamerV2's train function)
    # This depends on how dreamerv2_JAX is structured
    train(
        env=env,
        config=config,
        seed=config['seed'],
        output_dir=config['output_dir']
    )

if __name__ == "__main__":
    main()
```

---

### Phase 4: Configuration

```json
// experiments/configs/craftax_config.json
{
  "env_name": "Craftax-Symbolic-v1",
  "jax_env": true,
  "seed": 0,

  // Training
  "total_steps": 100000,
  "batch_size": 16,
  "sequence_length": 50,
  "train_every": 1000,

  // World model
  "hidden_size": 256,
  "latent_size": 32,
  "num_categories": 32,

  // Actor-critic
  "actor_lr": 3e-5,
  "critic_lr": 3e-5,
  "world_lr": 1e-4,

  // Replay
  "replay_capacity": 1000000,

  // Logging
  "output_dir": "./outputs",
  "log_every": 1000,
  "wandb_project": "craftax-dreamer"
}
```

---

### Phase 5: Cluster Setup (Later)

```bash
# slurm/train_job.sh
#!/bin/bash
#SBATCH --job-name=craftax_dreamer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out

# Load modules (adjust for your cluster)
module load python/3.11
module load cuda/12.1

# Activate environment
source ~/craftax-dreamer/.venv/bin/activate

# Run training
cd ~/craftax-dreamer
python experiments/train.py \
    --config experiments/configs/craftax_config.json \
    --seed $SLURM_ARRAY_TASK_ID

# Optional: sync results
# rsync -av outputs/ $SCRATCH/craftax_results/
```

Submit: `sbatch --array=0-4 slurm/train_job.sh`  (5 seeds)

---

## Development Workflow

### Local (M3 Mac)
```bash
# 1. Test environment
python scripts/test_env.py

# 2. Test wrapper
python scripts/test_wrapper.py

# 3. Quick training run (100 steps)
python experiments/train.py --debug --steps 100

# 4. Visualize results
python scripts/visualize.py --checkpoint outputs/latest
```

### Cluster (When Ready)
```bash
# 1. Push code to cluster
git push
# SSH to cluster
ssh cluster

# 2. Pull and setup
cd ~/craftax-dreamer
git pull
source .venv/bin/activate

# 3. Test on cluster CPU first
python experiments/train.py --debug --steps 100 --device cpu

# 4. Submit GPU job
sbatch slurm/train_job.sh

# 5. Monitor
squeue -u $USER
tail -f logs/job_*.out
```

---

## Implementation Priority

### Week 1: Foundations
- [ ] Set up repository structure
- [ ] Verify Craftax works (`test_env.py`)
- [ ] Study DreamerV2 interface
- [ ] Write wrapper (`craftax_wrapper.py`)
- [ ] Test wrapper in isolation

### Week 2: Integration
- [ ] Adapt DreamerV2 code to use your wrapper
- [ ] Get single training step working
- [ ] Debug shape mismatches
- [ ] Add logging/visualization

### Week 3: Local Training
- [ ] Run 10k step training locally
- [ ] Verify loss curves look reasonable
- [ ] Check memory usage
- [ ] Profile performance

### Week 4: Cluster Migration
- [ ] Set up cluster environment
- [ ] Write Slurm scripts
- [ ] Test small job
- [ ] Launch full training run

---

## Quick Start Commands

```bash
# Today (30 min setup):
mkdir ~/craftax-dreamer && cd ~/craftax-dreamer
git init
uv venv && source .venv/bin/activate
uv pip install craftax jax flax optax gymnax wandb
git clone https://github.com/kenjyoung/dreamerv2_JAX.git
mkdir -p experiments/{configs,wrappers,scripts,utils}
touch experiments/__init__.py

# Test Craftax works
cat > scripts/test_env.py << 'EOF'
import jax
from craftax import make_craftax_env_from_name

env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng, env.default_params)
print(f"✓ Obs shape: {jax.tree_map(lambda x: x.shape, obs)}")
EOF

python scripts/test_env.py

# Next: Study dreamerv2_JAX/environments.py and build wrapper
```

---

## Critical Success Factors

### The 80/20 Rule
**Getting the wrapper right is 80% of the work.** Everything else is configuration.

### Key Questions to Answer Early
1. What observation format does DreamerV2 expect?
   - Flat vector? Dict? Nested structure?
   - What dtype? (float32, int32?)

2. How does DreamerV2 handle episode resets?
   - Auto-reset? Manual reset?
   - How are episode boundaries communicated?

3. What's the action space format?
   - Discrete int? One-hot? Continuous?

### Study These Files First
1. `dreamerv2_JAX/environments.py` - Environment interface
2. `dreamerv2_JAX/dreamer_single_seed.py` - Training loop entry point
3. `craftax/craftax/envs/craftax_symbolic_env.py` - Craftax implementation

### Debugging Strategy
1. Start with smallest possible test
2. Test each component in isolation
3. Add complexity incrementally
4. Use print statements liberally (JAX debugging is hard)
5. Visualize intermediate outputs

---

## Common Pitfalls to Avoid

1. **Don't modify Craftax or DreamerV2 code directly**
   - Keep them as dependencies
   - All your changes go in `experiments/`

2. **Don't train on M3 for real runs**
   - Local = testing only
   - Cluster = actual training

3. **Don't skip the wrapper testing**
   - Spend time getting this right
   - Most bugs will be here

4. **Don't forget JAX is functional**
   - No in-place updates
   - Everything is immutable
   - Use jit/vmap where possible

5. **Don't ignore shapes early**
   - Print all shapes immediately
   - Mismatches compound quickly

---

## Success Metrics

### Week 1 Success
- [ ] Craftax runs locally
- [ ] DreamerV2 code understood
- [ ] Wrapper compiles without errors

### Week 2 Success
- [ ] Single training step completes
- [ ] Losses are computed (even if wrong)
- [ ] No shape errors

### Week 3 Success
- [ ] 10k steps complete locally
- [ ] Loss curves trend downward
- [ ] Agent performs better than random

### Week 4 Success
- [ ] Cluster job runs successfully
- [ ] Training scales to 100k+ steps
- [ ] Results logged to wandb

---

## Resources

### Documentation
- Craftax: https://github.com/MichaelTMatthews/Craftax
- DreamerV2 JAX: https://github.com/kenjyoung/dreamerv2_JAX
- Gymnax: https://github.com/RobertTLange/gymnax
- JAX: https://jax.readthedocs.io/

### Papers
- DreamerV2: https://arxiv.org/abs/2010.02193
- Craftax: https://arxiv.org/abs/2402.16801

### Community
- JAX Discord
- Craftax GitHub Issues
- r/MachineLearning
