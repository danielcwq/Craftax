#!/usr/bin/env python3
"""
Simple trajectory viewer that works WITHOUT needing JAX installed.

This viewer shows basic statistics by inspecting the pickle structure
without fully unpickling the state objects.
"""

import bz2
import pickle
import sys
from pathlib import Path
from collections import Counter


class StateStub:
    """Stub class to handle state unpickling without full dependencies"""
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __reduce__(self):
        return (StateStub, (), self.__dict__)


def load_trajectory_simple(filepath):
    """
    Load trajectory with minimal dependencies.
    
    This works by stubbing out the complex state objects and just
    extracting the action/reward/done arrays which are simple types.
    """
    # Custom unpickler that stubs out complex classes
    class SimpleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Stub out all craftax classes with our simple stub
            if 'craftax' in module or 'jax' in module:
                return StateStub
            return super().find_class(module, name)
    
    with bz2.BZ2File(filepath, 'rb') as f:
        unpickler = SimpleUnpickler(f)
        try:
            trajectory = unpickler.load()
            return trajectory
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            return None


def analyze_simple(trajectory):
    """Analyze trajectory with just the basic data"""
    
    if trajectory is None:
        return
    
    num_steps = len(trajectory['action'])
    total_reward = sum(trajectory['reward'])
    episode_ended = any(trajectory['done'])
    
    print("\n" + "=" * 70)
    print("TRAJECTORY QUICK VIEW")
    print("=" * 70)
    print("\n📊 Basic Statistics:")
    print(f"   Total steps:       {num_steps:,}")
    print(f"   Total reward:      {total_reward:.4f}")
    print(f"   Avg reward/step:   {total_reward/num_steps:.6f}")
    print(f"   Episode ended:     {episode_ended}")
    
    # Action analysis
    print(f"\n🎮 Top 10 Most Used Actions:")
    action_counts = Counter(trajectory['action'])
    for action_id, count in action_counts.most_common(10):
        percentage = 100 * count / num_steps
        print(f"   Action {action_id:2d}: {count:6,} times ({percentage:5.1f}%)")
    
    # Reward events
    reward_events = [(i, r) for i, r in enumerate(trajectory['reward']) if r > 0]
    print(f"\n🏆 Reward Events: {len(reward_events)}")
    if reward_events:
        for step, reward in reward_events[:10]:
            print(f"   Step {step:5d}: +{reward:.2f}")
        if len(reward_events) > 10:
            print(f"   ... and {len(reward_events) - 10} more")
    
    print("\n" + "=" * 70)
    print("\n💡 Note: For full state analysis, use view_trajectory.py")
    print("   (requires Craftax + JAX environment)")


def main():
    # Find trajectory file
    if len(sys.argv) >= 2:
        filepath = Path(sys.argv[1])
    else:
        script_dir = Path(__file__).parent
        play_data_dir = script_dir.parent / "play_data"
        
        if not play_data_dir.exists():
            print("❌ No play_data directory found")
            sys.exit(1)
        
        trajectories = sorted(play_data_dir.glob("trajectories_*.pbz2"))
        if not trajectories:
            print("❌ No trajectory files found in play_data/")
            sys.exit(1)
        
        filepath = trajectories[-1]
        print(f"📂 Using latest: {filepath.name}")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    
    print(f"\n⏳ Loading trajectory from: {filepath}")
    trajectory = load_trajectory_simple(filepath)
    
    if trajectory:
        analyze_simple(trajectory)


if __name__ == "__main__":
    main()

