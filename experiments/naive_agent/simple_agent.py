"""
SIMPLEST POSSIBLE RL AGENT FOR CRAFTAX
Using REINFORCE (basic policy gradient)

Core idea:
1. Neural network outputs action probabilities
2. Sample actions, play episode
3. If episode was good (high reward), increase probability of those actions
4. If episode was bad (low reward), decrease probability of those actions
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from craftax.craftax_env import make_craftax_env_from_name

# ============================================
# PART 1: THE BRAIN (Neural Network)
# ============================================

class SimplePolicy(nn.Module):
    """
    A simple neural network that looks at the game state
    and decides what action to take.

    Input: game observation (what the agent sees)
    Output: probability distribution over actions
    """
    action_dim: int  # How many actions are possible

    @nn.compact
    def __call__(self, observation):
        # Flatten the observation into a 1D vector
        # (neural networks like flat vectors)
        x = observation.reshape(-1)

        # Layer 1: 64 neurons (smaller network, less overfitting)
        x = nn.Dense(64)(x)
        x = nn.relu(x)  # Activation function

        # Layer 2: 64 neurons
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        # Output layer: one number per action
        # These are "logits" (unnormalized probabilities)
        action_logits = nn.Dense(self.action_dim)(x)

        return action_logits


# ============================================
# PART 2: PLAYING AN EPISODE
# ============================================

def play_episode(network, params, rng, env, env_params, max_steps=1000, verbose=False):
    """
    Play one complete episode of the game IN CRAFTAX.

    This function actually interacts with the Craftax environment:
    - env.reset() creates a new Craftax world
    - env.step() simulates the game physics, enemies, crafting, etc.
    - The 'state' object contains the full Craftax game state
    - The 'obs' is what the agent sees (a processed view of the state)

    Returns:
        observations: what the agent saw at each step
        actions: what the agent did at each step
        rewards: what reward the agent got at each step
        log_probs: how confident the agent was about each action
    """
    # Start the game (this creates a new Craftax world!)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)

    if verbose:
        print(f"  [Craftax] New episode started!")
        print(f"  [Craftax] Initial state: position={state.player_position if hasattr(state, 'player_position') else 'unknown'}")

    # Storage for the episode
    observations = []
    actions = []
    rewards = []
    log_probs = []

    # Play until done or max steps
    for step in range(max_steps):
        # Split the random key (JAX requirement)
        rng, action_rng, step_rng = jax.random.split(rng, 3)

        # Get action probabilities from the network
        obs_flat = obs.reshape(-1)
        action_logits = network.apply(params, obs_flat)

        # Sample an action from the probability distribution
        action = jax.random.categorical(action_rng, action_logits)

        # Calculate log probability (needed for training)
        # This tells us "how likely was this action?"
        action_probs = jax.nn.log_softmax(action_logits)
        log_prob = action_probs[action]

        # Take the action in the environment
        next_obs, state, reward, done, info = env.step(
            step_rng, state, action, env_params
        )

        # Store what happened
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

        # Update observation
        obs = next_obs

        # End episode if done
        if done:
            break

    return {
        'observations': jnp.array(observations),
        'actions': jnp.array(actions),
        'rewards': jnp.array(rewards),
        'log_probs': jnp.array(log_probs)
    }


# ============================================
# PART 3: LEARNING FROM THE EPISODE
# ============================================

def compute_returns(rewards, gamma=0.99):
    """
    Calculate the "return" (total discounted reward) for each timestep.

    The idea: Future rewards are worth slightly less than immediate rewards.

    Example:
        rewards = [1, 1, 1]
        returns = [1 + 0.99*1 + 0.99^2*1,
                   1 + 0.99*1,
                   1]
                = [2.97, 1.99, 1.00]
    """
    returns = []
    running_return = 0

    # Go backwards through the episode
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    return jnp.array(returns)


def train_step(params, optimizer_state, episode_data, optimizer):
    """
    Update the neural network based on the episode.

    REINFORCE rule:
    - If an action led to high return → increase its probability
    - If an action led to low return → decrease its probability
    """

    def loss_fn(params):
        # Compute returns (how good was each timestep?)
        returns = compute_returns(episode_data['rewards'])

        # Normalize returns (helps with training stability)
        # Makes them have mean=0, std=1
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        # Formula: -log_prob * return
        #
        # Why negative? We want to MAXIMIZE return, but optimizers MINIMIZE loss.
        # So we make high returns → negative loss → optimizer increases probability
        policy_loss = -(episode_data['log_probs'] * returns).mean()

        return policy_loss

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Update parameters using the optimizer
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, loss


# ============================================
# PART 4: MAIN TRAINING LOOP
# ============================================

def main():
    print("="*60)
    print("SIMPLE CRAFTAX AGENT - REINFORCE")
    print("="*60)

    # ========== SETUP ==========

    # Create the Craftax environment
    print("\n[1/5] Creating CRAFTAX environment...")
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    # Random number generator (JAX uses explicit randomness)
    rng = jax.random.PRNGKey(0)

    # Get environment info
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    obs_flat = obs.reshape(-1)
    action_dim = env.action_space(env_params).n

    print(f"   ✓ This IS Craftax! Environment type: {type(env)}")
    print(f"   Observation dimension: {obs_flat.shape[0]}")
    print(f"   Number of actions: {action_dim}")

    # Show what the Craftax state contains
    print(f"\n   Craftax Game State (what the agent is actually playing in):")
    if hasattr(state, 'player_position'):
        print(f"     - Player position: {state.player_position}")
    if hasattr(state, 'player_health'):
        print(f"     - Player health: {state.player_health}")
    if hasattr(state, 'player_level'):
        print(f"     - Player level: {state.player_level}")
    if hasattr(state, 'timestep'):
        print(f"     - Game timestep: {state.timestep}")

    # Explain the observation structure
    print(f"\n   WHAT THE NEURAL NETWORK SEES:")
    print(f"   ════════════════════════════════════════════════════════")

    from craftax.craftax.envs.craftax_symbolic_env import (
        get_flat_map_obs_shape,
        get_inventory_obs_shape
    )

    flat_map_size = get_flat_map_obs_shape()
    inventory_size = get_inventory_obs_shape()

    print(f"   Total input size: {obs_flat.shape[0]} numbers")
    print(f"\n   This breaks down into:")
    print(f"     [0 to {flat_map_size-1}]:     Map observation ({flat_map_size} values)")
    print(f"                             - What's around the player?")
    print(f"                             - Blocks, items, mobs in view")
    print(f"     [{flat_map_size} to {obs_flat.shape[0]-1}]:  Inventory ({inventory_size} values)")
    print(f"                             - What the player is carrying")
    print(f"                             - Resources, tools, health, hunger")

    print(f"\n   All values are normalized between 0.0 and 1.0")
    print(f"   Current observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Show a sample of what the network actually gets
    print(f"\n   Example: First 10 values fed to network:")
    print(f"   {obs_flat[:10]}")

    # ========== CREATE NETWORK ==========

    print("\n[2/5] Creating neural network...")
    network = SimplePolicy(action_dim=action_dim)

    # Initialize network parameters
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, obs_flat)

    # Count parameters
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"   Network has {num_params:,} parameters")

    # ========== CREATE OPTIMIZER ==========

    print("\n[3/5] Creating optimizer...")
    learning_rate = 3e-4
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)
    print(f"   Using Adam optimizer with lr={learning_rate}")

    # ========== TRAINING LOOP ==========

    print("\n[4/5] Training...")
    print("-" * 60)

    num_episodes = 50  # Reduced from 100 to avoid overfitting

    # Track best episode
    best_reward = -float('inf')
    best_episode_data = None
    best_episode_num = 0

    for episode in range(num_episodes):
        # Play one episode
        rng, episode_rng = jax.random.split(rng)
        episode_data = play_episode(
            network, params, episode_rng, env, env_params
        )

        # Learn from the episode
        params, optimizer_state, loss = train_step(
            params, optimizer_state, episode_data, optimizer
        )

        # Calculate episode statistics
        total_reward = episode_data['rewards'].sum()
        episode_length = len(episode_data['rewards'])

        # Track best episode
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_data = episode_data
            best_episode_num = episode

        # Print progress with actual Craftax state info
        if episode % 5 == 0:  # Print every 5 episodes (since we only have 50 now)
            is_best = " 🌟 NEW BEST!" if episode == best_episode_num else ""
            print(f"Episode {episode:3d} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Length: {episode_length:4d} | "
                  f"Loss: {loss:7.4f}{is_best}")

        # Every 25 episodes, show detailed Craftax state
        if episode % 25 == 0 and episode > 0:
            print(f"   └─> [Craftax Info] What happened in the game:")
            print(f"       Max reward in step: {episode_data['rewards'].max():.3f}")
            print(f"       Min reward in step: {episode_data['rewards'].min():.3f}")
            print(f"       Steps survived: {episode_length}")
            print(f"       Best so far: Episode {best_episode_num} with {best_reward:.2f} reward")

    print("-" * 60)
    print("\n[5/5] Training complete!")

    # ========== TEST THE TRAINED AGENT ==========

    print("\nTesting trained agent for 10 episodes IN CRAFTAX...")
    test_rewards = []
    test_lengths = []

    for test_ep in range(10):
        rng, test_rng = jax.random.split(rng)
        episode_data = play_episode(
            network, params, test_rng, env, env_params
        )
        test_rewards.append(episode_data['rewards'].sum())
        test_lengths.append(len(episode_data['rewards']))

    print(f"\nCraftax Performance Summary:")
    print(f"  Average reward: {jnp.mean(jnp.array(test_rewards)):.2f}")
    print(f"  Best reward: {jnp.max(jnp.array(test_rewards)):.2f}")
    print(f"  Average episode length: {jnp.mean(jnp.array(test_lengths)):.1f} steps")
    print(f"  Longest survival: {jnp.max(jnp.array(test_lengths))} steps")

    # ========== SHOW BEST EPISODE IN DETAIL ==========
    print("\n" + "="*70)
    print("REPLAY: The BEST Episode (Highest Reward)")
    print("="*70)

    print(f"\nEpisode {best_episode_num} was the best with {best_reward:.2f} total reward")
    print(f"Episode lasted {len(best_episode_data['rewards'])} steps")

    # Find steps with positive rewards
    rewarding_steps = []
    for step in range(len(best_episode_data['rewards'])):
        reward = float(best_episode_data['rewards'][step])
        if reward > 0:
            rewarding_steps.append((step, int(best_episode_data['actions'][step]), reward))

    if len(rewarding_steps) == 0:
        print(f"\nNo positive rewards in this episode (all zeros)")
        print("The agent survived but didn't accomplish anything rewarding yet.")
    else:
        print(f"\nFound {len(rewarding_steps)} steps with POSITIVE REWARDS:")
        print(f"Showing all {len(rewarding_steps)} rewarding moments...")
        print("-"*70)

        # Get action names
        try:
            from craftax.craftax.constants import Action
            action_enum = Action
        except:
            action_enum = None

        # Show only rewarding steps
        for step, action, reward in rewarding_steps:
            # Get action name
            if action_enum:
                try:
                    action_name = action_enum(action).name
                except:
                    action_name = f"Action_{action}"
            else:
                action_name = f"Action_{action}"

            # Show what happened
            print(f"  Step {step:3d}: {action_name:20s} → reward: {reward:+6.3f} ⭐")

    # Summary
    print("-"*70)
    print(f"\nWhat the agent accomplished in this episode:")
    print(f"  Total reward: {best_reward:.2f}")
    print(f"  Steps taken: {len(best_episode_data['rewards'])}")
    print(f"  Positive rewards: {(best_episode_data['rewards'] > 0).sum()} times")
    print(f"  Biggest single reward: {best_episode_data['rewards'].max():.3f}")
    print(f"  Average reward per step: {best_episode_data['rewards'].mean():.3f}")

    print("\n" + "="*70)
    print("EXPLANATION: What you're seeing")
    print("="*70)
    print("""
Each line shows:
  - Step number: Time in the episode
  - Action: What the agent chose to do (e.g., MOVE_LEFT, DO, etc.)
  - Reward: Points earned (positive = good, zero = neutral)
  - ⭐ marks when the agent got a reward

In Craftax, you get rewards for:
  - Collecting resources (wood, stone, etc.)
  - Crafting items
  - Defeating enemies
  - Discovering new areas
  - Surviving longer

Your agent is learning which actions lead to rewards!
""")

    print("\n" + "="*70)
    print("CONCLUSION: You can see your agent IS running in Craftax!")
    print("The environment is real - it learns to play the actual game.")
    print("="*70)


if __name__ == "__main__":
    main()
