#!/usr/bin/env python3
"""
Simple script to view and analyze Craftax human gameplay trajectories.

This script loads trajectory files from the play_data directory and provides
a clear analysis of the gameplay session.
"""

import bz2
import pickle
import sys
from collections import Counter
from pathlib import Path

# Add parent directory to path so we can import craftax
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trajectory(filepath):
    """
    Load a compressed pickle trajectory file.
    
    Args:
        filepath: Path to the .pbz2 trajectory file
        
    Returns:
        Dictionary with keys: 'state', 'action', 'reward', 'done'
        
    Note:
        Requires craftax and JAX to be installed to unpickle the state objects.
    """
    try:
        with bz2.BZ2File(filepath, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        print(f"\n❌ Error: Missing required module: {e}")
        print("\n💡 Make sure you're running this in the same Python environment")
        print("   where Craftax is installed (with JAX dependencies).")
        print("\n   If using conda: conda activate <your_env>")
        print("   If using venv: source <your_venv>/bin/activate")
        sys.exit(1)


def print_basic_stats(trajectory):
    """Print basic statistics about the trajectory."""
    num_steps = len(trajectory['action'])
    total_reward = sum(trajectory['reward'])
    episode_ended = any(trajectory['done'])
    
    print("\n" + "=" * 70)
    print("TRAJECTORY OVERVIEW")
    print("=" * 70)
    
    print(f"\n📊 Basic Statistics:")
    print(f"   Total steps taken:     {num_steps:,}")
    print(f"   Total reward earned:   {total_reward:.4f}")
    print(f"   Average reward/step:   {total_reward/num_steps:.6f}")
    print(f"   Episode ended (died):  {episode_ended}")


def print_action_analysis(trajectory):
    """Analyze and print action usage statistics."""
    num_steps = len(trajectory['action'])
    action_counts = Counter(trajectory['action'])
    
    print(f"\n🎮 Action Frequency (Top 10):")
    print(f"   {'Action ID':<12} {'Count':<10} {'Percentage'}")
    print(f"   {'-'*12} {'-'*10} {'-'*10}")
    
    for action_id, count in action_counts.most_common(10):
        percentage = 100 * count / num_steps
        print(f"   Action {action_id:2d}    {count:6,}     {percentage:5.1f}%")


def print_reward_timeline(trajectory):
    """Show when rewards were earned during gameplay."""
    reward_events = [(step, reward) for step, reward in enumerate(trajectory['reward']) if reward > 0]
    
    print(f"\n🏆 Reward Events: {len(reward_events)} total")
    
    if reward_events:
        print(f"   {'Step':<10} {'Reward'}")
        print(f"   {'-'*10} {'-'*10}")
        
        # Show first 10 reward events
        for step, reward in reward_events[:10]:
            print(f"   {step:<10} +{reward:.2f}")
        
        if len(reward_events) > 10:
            print(f"   ... and {len(reward_events) - 10} more reward events")
    else:
        print("   No rewards earned in this trajectory")


def print_state_info(trajectory):
    """Print information about starting and ending game states."""
    first_state = trajectory['state'][0]
    last_state = trajectory['state'][-1]
    
    print(f"\n💚 Health:")
    print(f"   Starting: {first_state.player_health}")
    print(f"   Ending:   {last_state.player_health}")
    
    # Check if achievements exist in the state
    if hasattr(first_state, 'achievements'):
        start_achievements = int(first_state.achievements.sum())
        end_achievements = int(last_state.achievements.sum())
        new_achievements = end_achievements - start_achievements
        
        print(f"\n🎯 Achievements:")
        print(f"   Starting:     {start_achievements}")
        print(f"   Ending:       {end_achievements}")
        print(f"   New unlocked: {new_achievements}")


def find_latest_trajectory():
    """
    Find the most recent trajectory file in the play_data directory.
    
    Returns:
        Path to the latest trajectory file, or None if not found
    """
    # Look for play_data directory relative to this script
    script_dir = Path(__file__).parent
    play_data_dir = script_dir.parent / "play_data"
    
    if not play_data_dir.exists():
        return None
    
    trajectories = sorted(play_data_dir.glob("trajectories_*.pbz2"))
    
    if trajectories:
        return trajectories[-1]  # Return most recent
    
    return None


def analyze_trajectory(trajectory):
    """
    Main analysis function - calls all analysis functions.
    
    Args:
        trajectory: Dictionary containing trajectory data
    """
    print_basic_stats(trajectory)
    print_action_analysis(trajectory)
    print_reward_timeline(trajectory)
    print_state_info(trajectory)
    print("\n" + "=" * 70)


def main():
    """Main entry point for the trajectory viewer."""
    
    # Determine which trajectory file to load
    if len(sys.argv) >= 2:
        # User specified a file
        filepath = Path(sys.argv[1])
    else:
        # Try to find the latest trajectory
        filepath = find_latest_trajectory()
        
        if filepath is None:
            print("❌ No trajectory files found in play_data/")
            print("\nUsage:")
            print("  python view_trajectory.py                           # Use latest trajectory")
            print("  python view_trajectory.py <path/to/trajectory.pbz2> # Use specific file")
            sys.exit(1)
        
        print(f"📂 Using most recent trajectory: {filepath.name}")
    
    # Check if file exists
    if not filepath.exists():
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)
    
    # Load and analyze
    print(f"\n⏳ Loading trajectory from: {filepath}")
    trajectory = load_trajectory(filepath)
    
    analyze_trajectory(trajectory)
    
    # Helpful tips
    print("\n💡 Tips:")
    print("   • To explore the data interactively, use Python's interactive mode")
    print("   • Trajectory contains: 'state', 'action', 'reward', 'done' keys")
    print(f"   • To load: import bz2, pickle; t = pickle.load(bz2.BZ2File('{filepath}', 'rb'))")


if __name__ == "__main__":
    main()

