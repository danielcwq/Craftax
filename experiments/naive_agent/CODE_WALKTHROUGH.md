# Code Walkthrough: Where Craftax Happens

## The Big Picture

```
main()
  │
  ├─> [1] Create Craftax env (line 197)
  │
  └─> For each episode (line 257):
       │
       ├─> [2] play_episode()
       │    │
       │    ├─> env.reset() - Start new Craftax world
       │    │
       │    └─> Loop 1000 times:
       │         ├─> Agent picks action
       │         └─> env.step() - Run Craftax physics
       │
       └─> [3] train_step() - Update neural network
```

---

## Step-by-Step Code Trace

### STEP 1: Creating Craftax (Line 197)

```python
# In main() function, line 197:
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params
```

**What happens here:**
- Creates a Craftax environment object
- This is a **class instance** that contains all the game logic
- `env_params` contains game configuration (world size, difficulty, etc.)

**Craftax is now loaded in memory, ready to play!**

---

### STEP 2: Starting a New Episode (Line 75 in play_episode)

```python
# Inside play_episode() function, line 75:
def play_episode(network, params, rng, env, env_params, ...):
    # Start the game (this creates a new Craftax world!)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    #            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #            THIS IS CRAFTAX INTERACTION #1
```

**What `env.reset()` does inside Craftax:**
1. Generates a new random world (terrain, trees, enemies, etc.)
2. Places the player at a starting position
3. Initializes player stats (health=9, hunger=9, etc.)
4. Returns:
   - `obs`: What the agent sees (2941 numbers representing the world)
   - `state`: Full game state (player position, health, inventory, world map, etc.)

**This is the ACTUAL Craftax code running:**
```python
# Inside Craftax (craftax/envs/craftax_symbolic_env.py:143)
def reset_env(self, rng, params):
    rng, _rng = jax.random.split(rng)
    state = generate_world(_rng, params, self.static_env_params)  # Creates world!
    return self.get_obs(state), state
```

---

### STEP 3: Agent Observes the World (Line 90)

```python
# Line 90 in play_episode():
for step in range(max_steps):
    # Get action probabilities from the network
    obs_flat = obs.reshape(-1)
    action_logits = network.apply(params, obs_flat)
    #                                     ^^^^^^^^
    #                            Agent looks at Craftax observation
```

**What the agent sees:**
- `obs` is a vector of 2941 numbers (all between 0 and 1)
- Represents: nearby blocks, items, enemies, player inventory, etc.
- This is the **processed view** of the Craftax state

---

### STEP 4: Agent Chooses Action (Line 96)

```python
# Line 96:
action = jax.random.categorical(action_rng, action_logits)
```

**What this means:**
- Agent picks one of 17 possible actions:
  - 0-3: Move (up/down/left/right)
  - 4: Do (mine/attack/craft)
  - 5-16: Other actions (sleep, place block, etc.)

**This is NOT Craftax yet - just the agent deciding what to do**

---

### STEP 5: Taking Action in Craftax (Line 103)

```python
# Line 103 - THIS IS THE MAIN CRAFTAX INTERACTION!
next_obs, state, reward, done, info = env.step(
    step_rng, state, action, env_params
)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# THIS RUNS THE ENTIRE CRAFTAX GAME ENGINE
```

**What `env.step()` does inside Craftax:**

Here's the ACTUAL Craftax code that runs:

```python
# Inside Craftax (craftax/envs/craftax_symbolic_env.py:128)
def step_env(self, rng, state, action, params):
    # THIS IS WHERE THE GAME HAPPENS!
    state, reward = craftax_step(rng, state, action, params, self.static_env_params)

    done = self.is_terminal(state, params)
    info = log_achievements_to_info(state, done)

    return (
        self.get_obs(state),  # New observation
        state,                # Updated game state
        reward,               # Reward for this action
        done,                 # Is episode over?
        info                  # Extra info (achievements, etc.)
    )
```

**And `craftax_step()` does ALL of this:**

```python
# Inside craftax/game_logic.py
def craftax_step(rng, state, action, params, static_params):
    # 1. Move the player
    state = move_player(state, action)

    # 2. Handle "do" action (mine, attack, craft)
    state = handle_do_action(state, action)

    # 3. Update mobs (enemies move and attack)
    state = update_mobs(rng, state)

    # 4. Update player stats (hunger, health, etc.)
    state = update_player_vitals(state)

    # 5. Grow plants
    state = grow_plants(state)

    # 6. Calculate reward for achievements
    reward = calculate_reward(state)

    return state, reward
```

**This is the ENTIRE game simulation!**
- Player moves
- Enemies move
- Combat happens
- Resources are gathered
- Health/hunger changes
- Achievements are tracked

---

### STEP 6: Agent Receives Results (Line 103)

After `env.step()` returns:

```python
next_obs  # What the world looks like now (2941 numbers)
state     # Full Craftax game state (position, health, inventory, etc.)
reward    # Reward earned (e.g., +1 for collecting wood)
done      # True if player died or won
info      # Achievement data
```

**The agent stores this:**

```python
# Line 106-109:
observations.append(obs)
actions.append(action)
rewards.append(reward)
log_probs.append(log_prob)
```

---

### STEP 7: Repeat Until Episode Ends (Line 115)

```python
# Line 112:
obs = next_obs  # Update observation

# Line 115:
if done:
    break  # Episode over (player died or won)
```

**The loop continues:**
- Agent sees new `obs`
- Picks new `action`
- Craftax runs `env.step()`
- Until `done=True` or 1000 steps

---

## Complete Flow for ONE Episode

```
Episode N starts:
┌──────────────────────────────────────────────────────────┐
│ play_episode() called                                     │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ env.reset()             │  ← CRAFTAX CALL #1
         │ Creates new world       │
         └─────────────────────────┘
                       │
                       ▼ Returns obs, state
         ┌─────────────────────────┐
         │ Step 0:                 │
         │   Agent sees obs        │
         │   Picks action 5        │
         └─────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ env.step(action=5)      │  ← CRAFTAX CALL #2
         │ Simulates game:         │
         │  - Player moves         │
         │  - Enemies move         │
         │  - Combat happens       │
         │  - Reward calculated    │
         └─────────────────────────┘
                       │
                       ▼ Returns next_obs, reward, done
         ┌─────────────────────────┐
         │ Step 1:                 │
         │   Agent sees next_obs   │
         │   Picks action 2        │
         └─────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ env.step(action=2)      │  ← CRAFTAX CALL #3
         └─────────────────────────┘
                       │
                       ▼
         ... (repeat 1000 times or until done) ...
                       │
                       ▼
         ┌─────────────────────────┐
         │ Return episode data     │
         │ (all obs, actions,      │
         │  rewards from Craftax)  │
         └─────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ train_step() - Update neural network                     │
│ (Uses the rewards from Craftax to learn)                 │
└──────────────────────────────────────────────────────────┘
```

---

## Where is Craftax? A Map

```
simple_agent.py
├─ Line 16:  Import Craftax
├─ Line 197: CREATE Craftax env
│
├─ play_episode() [Line 57-123]
│  │
│  ├─ Line 75:  env.reset()     ← CRAFTAX RUNNING HERE
│  │            └─> Creates new world (terrain, enemies, player)
│  │
│  └─ Line 103: env.step()      ← CRAFTAX RUNNING HERE (repeatedly)
│               └─> Simulates: movement, combat, crafting, physics
│
└─ train_step() [Line 131-185]
   └─ Uses Craftax's rewards to update neural network
```

---

## Data Flow

```
Craftax → Agent → Craftax → Agent → ...

┌─────────────┐
│  CRAFTAX    │
│  (Game)     │
└──────┬──────┘
       │
       │ obs (2941 numbers)
       │ "Here's what you see"
       ▼
┌─────────────┐
│   AGENT     │
│  (Network)  │
└──────┬──────┘
       │
       │ action (0-16)
       │ "I want to move left"
       ▼
┌─────────────┐
│  CRAFTAX    │
│  Simulates: │
│  - Player   │
│    moves    │
│  - Enemies  │
│    attack   │
│  - Health   │
│    changes  │
└──────┬──────┘
       │
       │ next_obs, reward, done
       │ "You moved, got attacked, -1 health, reward=0"
       ▼
┌─────────────┐
│   AGENT     │
│   Learns    │
└─────────────┘
```

---

## Key Takeaway

**Craftax is called in exactly 2 places:**

1. **`env.reset()`** (line 75) - Start new episode
   - Generates world
   - Places player
   - Returns initial observation

2. **`env.step(action)`** (line 103) - Take action
   - Simulates entire game tick
   - Player moves
   - Enemies act
   - Physics updates
   - Returns: new observation, reward, done flag

Everything else (neural network, training) uses the **data from Craftax** but doesn't interact with it directly.

---

## To Verify This Yourself

Add print statements:

```python
# Line 75, after env.reset():
obs, state = env.reset(reset_rng, env_params)
print(f"[CRAFTAX RESET] Created world, player at {state.player_position}")

# Line 103, after env.step():
next_obs, state, reward, done, info = env.step(...)
print(f"[CRAFTAX STEP] Action taken, reward={reward}, pos={state.player_position}")
```

Run it and you'll see Craftax in action!
