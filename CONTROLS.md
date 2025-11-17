# 🎮 Craftax Controls Reference

Complete command reference for Craftax. All controls are printed at game start, but this document provides an organized reference with context.

## 🚶 Basic Movement

| Key | Action | Description |
|-----|--------|-------------|
| `W` | Move Up | Move player north |
| `A` | Move Left | Move player west |
| `S` | Move Down | Move player south |
| `D` | Move Right | Move player east |
| `Q` | No-Op | Do nothing (skip turn) |

## 🤝 Core Interaction

| Key | Action | Description |
|-----|--------|-------------|
| `SPACE` | Do/Interact | Mine blocks, attack creatures, drink water, eat fruit, open chests |

## ⛏️ Crafting - Pickaxes

**Requirement:** Must be standing adjacent to a **Crafting Table** (place with `T`)

| Key | Item | Materials Required | Special Requirement |
|-----|------|-------------------|---------------------|
| `1` | Wood Pickaxe | 1 Wood | Crafting Table |
| `2` | Stone Pickaxe | 1 Wood + 1 Stone | Crafting Table |
| `3` | Iron Pickaxe | 1 Wood + 1 Stone + 1 Iron + 1 Coal | Crafting Table + Furnace |
| `4` | Diamond Pickaxe | 1 Wood + 3 Diamond | Crafting Table |

## ⚔️ Crafting - Swords

**Requirement:** Must be standing adjacent to a **Crafting Table** (place with `T`)

| Key | Item | Materials Required | Special Requirement |
|-----|------|-------------------|---------------------|
| `5` | Wood Sword | 1 Wood | Crafting Table |
| `6` | Stone Sword | 1 Wood + 1 Stone | Crafting Table |
| `7` | Iron Sword | 1 Wood + 1 Stone + 1 Iron + 1 Coal | Crafting Table + Furnace |
| `8` | Diamond Sword | 1 Wood + 2 Diamond | Crafting Table |

## 🛡️ Crafting - Armor

**Requirement:** Must be standing adjacent to a **Crafting Table** (place with `T`)

| Key | Item | Materials Required | Special Requirement |
|-----|------|-------------------|---------------------|
| `Y` | Iron Armor (1 piece) | 3 Iron + 3 Coal | Crafting Table + Furnace |
| `U` | Diamond Armor (1 piece) | 3 Diamond | Crafting Table |

**Note:** Armor is crafted one piece at a time. There are 4 pieces: Helmet, Chestplate, Leggings, Boots.

## 🎯 Crafting - Other Items

**Requirement:** Must be standing adjacent to a **Crafting Table** (place with `T`)

| Key | Item | Materials Required | Output |
|-----|------|-------------------|--------|
| `O` | Arrows | 1 Wood + 1 Stone | +2 Arrows |
| `[` | Torches | 1 Wood + 1 Coal | +4 Torches |

## 🏗️ Placing Blocks & Items

| Key | Item | Materials Required | Description |
|-----|------|-------------------|-------------|
| `T` | Crafting Table | 2 Wood | Required for all crafting. Place adjacent to you |
| `F` | Furnace | 1 Stone | Required for iron items. Place next to crafting table |
| `R` | Stone Block | 1 Stone | Build walls, fill water, block yourself in for sleeping |
| `P` | Plant/Sapling | 1 Sapling | Plant seeds for farming food |
| `J` | Torch | 1 Torch (from inventory) | Light up dark levels (Floor 2+) |

## ⚡ Combat & Ranged Attacks

| Key | Action | Cost | Description |
|-----|--------|------|-------------|
| `I` | Shoot Arrow | 1 Arrow | Ranged attack. Requires bow (found in first chest) |
| `G` | Cast Fireball | 2 Mana | Fire damage spell. Must learn from book first |
| `H` | Cast Iceball | 2 Mana | Ice damage spell. Must learn from book first |

## ✨ Enchanting

**Requirement:** Must be standing adjacent to an **Enchantment Table** (found on Floors 3 & 4)

| Key | Item | Cost | Gemstone | Description |
|-----|------|------|----------|-------------|
| `K` | Enchant Sword | 9 Mana | 1 Sapphire or Ruby | Adds 50% elemental damage |
| `L` | Enchant Armor | 9 Mana | 1 Sapphire or Ruby | Reduces elemental damage by 20% per piece |
| `;` | Enchant Bow | 9 Mana | 1 Sapphire or Ruby | Adds elemental damage to arrows |

**Enchantment Types:**
- **Sapphire** = Ice Enchantment
- **Ruby** = Fire Enchantment

## 🧪 Potions

| Key | Potion Color | Effect |
|-----|-------------|--------|
| `Z` | Red Potion | Random effect (changes each game) |
| `X` | Green Potion | Random effect (changes each game) |
| `C` | Blue Potion | Random effect (changes each game) |
| `V` | Pink Potion | Random effect (changes each game) |
| `B` | Cyan Potion | Random effect (changes each game) |
| `N` | Yellow Potion | Random effect (changes each game) |

**Note:** Each potion gives either +8 or -3 to Health, Mana, or Energy. Effects are randomly assigned each game - experiment to learn!

## 📚 Learning & Books

| Key | Action | Description |
|-----|--------|-------------|
| `M` | Read Book | Learn a random spell (Fireball or Iceball). First book found on Floor 3 |

## 📊 Level Up Attributes

**Awarded:** 1 experience point per floor descended (first time only)  
**Maximum:** Level 5 per attribute

| Key | Attribute | Benefits |
|-----|-----------|----------|
| `]` | Dexterity | +Max food/water/energy, slower decay, +bow damage |
| `-` | Strength | +Melee damage, +max health |
| `=` | Intelligence | +Max mana, slower mana decay, +spell damage, +enchantment effectiveness |

## 🛌 Rest & Recovery

| Key | Action | Description |
|-----|--------|-------------|
| `TAB` | Sleep | Restores energy. Makes you vulnerable - block yourself in with stone! |
| `E` | Rest | Auto no-op until health/intrinsic changes or attacked. Good for healing |

## 🪜 Floor Navigation

| Key | Action | Description |
|-----|--------|-------------|
| `.` | Descend | Go down ladder to next floor. Must kill 8 creatures first (except Floor 0) |
| `,` | Ascend | Go up ladder to previous floor |

## 📈 Player Intrinsics (Status Bars)

Monitor these 5 core stats:

- **Health** - Die if it reaches 0. Recovers when other intrinsics are non-zero
- **Hunger** - Naturally decreases. Eat cows, snails, bats, or plants
- **Thirst** - Naturally decreases. Drink from lakes, water patches, or fountains
- **Energy** - Naturally decreases. Sleep to restore (TAB key)
- **Mana** - Used for spells/enchantments. Naturally recovers over time

⚠️ **WARNING:** Health decreases if ANY intrinsic (hunger/thirst/energy) reaches 0!

---

## 🎯 Quick Start Guide

1. **Mine trees** (walk into them, press `SPACE`)
2. **Place crafting table** (press `T`)
3. **Craft wood pickaxe** (stand next to table, press `1`)
4. **Mine stone** (with pickaxe equipped)
5. **Craft stone tools** (press `2` for pickaxe, `6` for sword)
6. **Place furnace** (press `F`, requires 1 stone)
7. **Mine coal and iron** (with stone pickaxe)
8. **Craft iron tools** (stand adjacent to both table & furnace, press `3` or `7`)

## 🗺️ Floor Progression

- **Floor 0:** Overworld (grasslands, lakes, mountains)
- **Floor 1:** Dungeon (rooms with fountains and chests)
- **Floor 2:** Gnomish Mines (DARK - needs torches)
- **Floor 3:** Sewers (water, first spellbook, ice enchantment table)
- **Floor 4:** Vaults (fire enchantment table)
- **Floor 5:** Troll Mines (DARK - rich with ores)
- **Floor 6:** Fire Realm (lava islands, no water)
- **Floor 7:** Ice Realm (DARK - no food, requires fire damage)
- **Floor 8:** Graveyard (final boss: Necromancer)

## 💡 Pro Tips

- **Furnace Placement:** Place furnace directly adjacent to crafting table. Stand where you're touching both (diagonal works!)
- **Sleeping Safety:** Always block yourself in with stone (`R` key) before sleeping
- **Dark Levels:** Craft torches (key `[`) with wood + coal. Place with `J`
- **Farming:** Press `SPACE` on grass to get seeds, plant with `P`, return later for food
- **Boss Fight:** Necromancer summons waves. Attack only when he's vulnerable between waves

## 🔧 Technical Notes

- First frame render: ~30 seconds (JAX compilation)
- First action: ~20 seconds (JAX compilation)
- After compilation: Very fast gameplay

---

**Full Action Count:** 43 distinct commands

For gameplay tutorial and strategy guide, see `tutorial.md`

---

**You're ready to play! Run:** `play_craftax`

