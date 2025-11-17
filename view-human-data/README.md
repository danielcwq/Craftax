# View Human Data

Simple tools for analyzing human gameplay trajectories from Craftax.

## Quick Start

```bash
cd view-human-data

# View the most recent trajectory (auto-finds latest in play_data/)
python view_trajectory.py

# Or view a specific trajectory file
python view_trajectory.py ../play_data/trajectories_1763341398.pkl.pbz2
```

### Alternative: Simple Viewer (if you get import errors)

```bash
# Lighter version that works with minimal dependencies
python simple_viewer.py
```

## What It Shows

The script analyzes your gameplay and shows:

- **Basic Stats**: Steps taken, total reward, average reward per step
- **Action Frequency**: Which controls you used most often
- **Reward Timeline**: When you earned achievements
- **Health & Achievements**: Starting vs ending state

## Trajectory Data Structure

Each trajectory file contains:

```python
{
    'state': [EnvState, EnvState, ...],    # Complete game states
    'action': [int, int, ...],              # Action IDs (0-42)
    'reward': [float, float, ...],          # Rewards earned
    'done': [bool, bool, ...]               # Episode termination
}
```

## Requirements

- Python 3.7+
- Craftax installed (with JAX dependencies)
- The trajectory files must be in `../play_data/` (relative to this directory)

**Note:** Run this script in the same Python environment where you installed Craftax:
```bash
# If you installed Craftax in a specific environment, activate it first
# Example: conda activate craftax_env  or  source venv/bin/activate
python view_trajectory.py
```

## Tips

- Trajectories are only saved if you run: `play_craftax --save_trajectories`
- Exit the game by closing the window (not Ctrl+C) to ensure saving
- Each trajectory is timestamped, so you can track multiple gameplay sessions

