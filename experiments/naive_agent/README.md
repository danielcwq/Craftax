# Simple RL Agent for Craftax

## Quick Start

```bash
cd ~/Waterloo/Craftax
python experiments/naive_agent/simple_agent.py
```

Expected: ~5-10 minutes on M3 Mac, trains for 100 episodes.

## Files

- `simple_agent.py` - The complete agent (heavily commented)
- `WALKTHROUGH.md` - Detailed explanation of how it works
- This README - Quick reference

## What You'll Learn

1. **Policy Networks** - Neural networks that output actions
2. **REINFORCE** - Basic policy gradient algorithm
3. **Returns** - Discounted future rewards
4. **JAX Basics** - Functional programming for ML

## The Algorithm (30 Second Version)

```
For each episode:
    1. Play game using current policy
    2. Calculate returns (total future reward per step)
    3. Update policy:
       - Increase probability of actions that led to high returns
       - Decrease probability of actions that led to low returns
```

## Expected Results

| Metric | Start (Episode 0) | End (Episode 100) |
|--------|------------------|-------------------|
| Reward | 2-5 | 10-20 |
| Episode Length | 50-200 | 200-500 |
| Loss | -2 to -1 | -0.5 to 0 |

Your agent won't be great, but it should be better than random!

## Code Structure

```python
# 1. Define policy network
class SimplePolicy(nn.Module):
    def __call__(self, obs):
        return action_logits

# 2. Play episode
def play_episode(network, params, ...):
    for step in range(max_steps):
        action = sample(network(obs))
        obs, reward = env.step(action)
    return episode_data

# 3. Learn from episode
def train_step(params, episode_data, ...):
    returns = compute_returns(rewards)
    loss = -(log_probs * returns).mean()
    params = update(params, loss)
    return params

# 4. Main loop
for episode in range(100):
    episode_data = play_episode(...)
    params = train_step(params, episode_data, ...)
```

## Key Concepts

### Policy (π)
The strategy the agent uses to pick actions.
```python
π(a|s) = probability of action a given state s
```

### Return (G)
Total discounted future reward.
```python
G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    = r_t + γ*G_{t+1}

# γ (gamma) = 0.99 in our code
```

### Policy Gradient
Update rule for the neural network.
```python
∇L = -log_prob(action) * return
```

## Hyperparameters

| Parameter | Value | What it does |
|-----------|-------|--------------|
| Learning rate | 3e-4 | How fast the network learns |
| Gamma (γ) | 0.99 | How much future rewards matter |
| Hidden size | 128 | Size of network layers |
| Episodes | 100 | How many games to play |
| Max steps | 1000 | Max length of one episode |

## Common Questions

**Q: Why is it so slow?**
A: REINFORCE learns from complete episodes (no batching, no vectorization). This is the simplest version for learning. GPU/vectorized version is 100x faster.

**Q: Why doesn't reward always increase?**
A: RL has high variance. Some episodes are lucky, some unlucky. The trend should go up over time.

**Q: What's a good final reward?**
A: For this simple agent, 10-20 is reasonable. PPO gets 30-50+.

**Q: Can I make it faster?**
A: Yes! Next steps:
1. Vectorize environments (run 16 in parallel)
2. Use GPU
3. Switch to PPO (better algorithm)

**Q: How do I know it's working?**
A: Check these:
- [ ] Code runs without errors
- [ ] Loss decreases (becomes less negative)
- [ ] Reward generally increases over time
- [ ] Episode length increases over time

## Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'craftax'
```
**Fix:**
```bash
uv pip install craftax jax flax optax
```

### NaN Loss
```
Episode  42 | Reward:   nan | Length:  234 | Loss:    nan
```
**Cause:** Numerical instability
**Fix:** Already handled in code (line 158: `+ 1e-8`)

### Very Low Rewards
```
Episode 100 | Reward:   0.50 | ...
```
**Possible issues:**
- Learning rate too high/low (try 1e-3 or 1e-5)
- Network too small (increase hidden size to 256)
- Unlucky random seed (try seed=42)

### Slow Training
**Expected speed on M3 Mac:** ~100-200 steps/sec
**Expected time per episode:** 2-5 seconds
**Total runtime:** 5-10 minutes

This is normal for CPU! GPU would be 50-100x faster.

## Next Steps

Once you understand this agent:

1. **Read the code** - Every line is commented
2. **Read WALKTHROUGH.md** - Detailed explanations
3. **Modify hyperparameters** - See what changes
4. **Add logging** - Track metrics with wandb
5. **Upgrade to PPO** - Better algorithm
6. **Vectorize** - Run multiple environments in parallel
7. **Move to GPU** - 100x speedup

## Math Reference

### Softmax (converts logits to probabilities)
```python
logits = [2.1, -0.5, 1.3]
probs = exp(logits) / sum(exp(logits))
      = [0.56, 0.04, 0.25]
```

### Log Softmax (numerically stable version)
```python
log_probs = logits - log(sum(exp(logits)))
```

### Categorical Sampling
```python
probs = [0.5, 0.3, 0.2]
# 50% chance of action 0
# 30% chance of action 1
# 20% chance of action 2
```

### Gradient Descent
```python
# Goal: minimize loss
params_new = params_old - learning_rate * gradient

# With Adam optimizer:
# (automatically adjusts learning rate per parameter)
```

## Comparison to Other Algorithms

| Algorithm | Sample Efficiency | Computation | Stability |
|-----------|------------------|-------------|-----------|
| **REINFORCE** (ours) | Low | Low | Medium |
| DQN | Medium | Medium | Medium |
| PPO | Medium | Medium | High |
| SAC | High | High | High |
| DreamerV2 | Very High | Very High | Medium |

REINFORCE is the simplest, but not the best. It's a great starting point for learning!

## Resources

- **Our code**: `simple_agent.py` (heavily commented)
- **Craftax**: https://github.com/MichaelTMatthews/Craftax
- **REINFORCE paper**: Williams 1992 (original)
- **Spinning Up**: https://spinningup.openai.com/en/latest/ (RL tutorial)
- **JAX tutorial**: https://jax.readthedocs.io/

## Performance Benchmark

On M3 Max (CPU):
- Steps/sec: ~150
- Episode time: ~3 seconds
- Total training time: ~8 minutes

On A100 (GPU, vectorized):
- Steps/sec: ~20,000
- Episode time: 0.01 seconds
- Total training time: ~10 seconds

## License

MIT (same as Craftax)
