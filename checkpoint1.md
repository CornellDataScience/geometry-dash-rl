# GD RL Checkpoint 1 — BC + PPO Pipeline State

## Project Overview

Building an RL agent that plays Geometry Dash. The pipeline:

- **Mod** (Geode/C++) hooks `PlayLayer::postUpdate` to expose game state via POSIX shared memory at `/gdrl_ipc_v3`. Per frame it writes a 608-float observation, a per-frame jump-held label, and accepts agent actions back from Python.
- **Python**: reads obs, preprocesses into a compact 157-dim per-frame vector with learned object-type embeddings, runs a 1.3M-param MLP, sends back `jump`/`no jump`.

### Stack

```
GD (Geometry Dash) running natively
  ↓ Geode mod (mods/TrainingPipeline) writes obs to /gdrl_ipc_v3
  ↓ Python adapter (src/gdrl/env/geode_ipc_v3.py) reads/writes
  ↓ Gym env (src/gdrl/env/privileged_env.py) — reward shaping, reset
  ↓ Preprocessor (src/gdrl/model/obs_preprocess.py) — subsampling, embeddings
  ↓ Model (src/gdrl/model/mlp_agent.py) — GDPolicyMLP
```

### Observation pipeline

Raw mod obs (608 floats):
- `[0..7]` player: x, y, vy, dx, on_ground, is_dead, always_1, mode
- `[8..607]` 100 nearest objects × 6 floats: relX, relY, objType, objID, scaleX, scaleY

Preprocessed (157 floats per frame):
- 7 player features (drop absolute x, drop is_dead, drop always_1; mode → 4-bit one-hot)
- 30 selected objects × 5 floats: relX, relY, **type_id**, scaleX, scaleY
  - Object selection: 15 nearest by |relX| + 15 exponentially-spaced beyond
  - `type_id` is an integer 0-41 mapping to one of 42 object types (yellow orb, gravity portal, ship portal, spike, sawblade, etc.); unknowns → block

The MLP internally expands `type_id` floats into 8-dim learned embeddings via `nn.Embedding(42, 8)`, producing effective 367 features per frame, 1468 over a 4-frame stack.

### Model architecture (`GDPolicyMLP`)

```
input (628 floats: 4 frames × 157)
  ↓ embedding expansion → 1468 effective features
  ↓ Linear(1468, 512) + LayerNorm + ReLU
  ↓ Linear(512, 512) + LayerNorm + ReLU
  ↓ Linear(512, 512) + LayerNorm + ReLU
  ↓ Linear(512, 128) + ReLU  (neck)
  ├── action_head: Linear(128, 1)   → logit (Bernoulli for jump)
  └── value_head:  Linear(128, 1)   → V(s) (PPO critic)
```

Total: ~1.3M parameters. Used unchanged for both BC training and PPO.

---

## Data Collection

### Recording infrastructure

- `record_human.py` connects to the mod's ring buffer, captures every frame at full game rate (60-240 fps depending on TPS), writes to NPZ shards.
- Dead-frame deduplication: only the first death frame is recorded, then all subsequent dead frames are skipped until respawn.
- Auto-stops on death or `level_done`.
- One session per macro run, organized as `artifacts/recordings/<level_name>/<timestamp>/shard_*.npz`.

### What we collected

Initially used **Eclipse macros** (deterministic perfect plays, including coin paths) to generate clean data.

**Total dataset (12 levels):**
| Level | Sessions | Frames |
|-------|----------|--------|
| back_on_track | 1 | 9,845 |
| base_after_base | 1 | 10,105 |
| cant_let_go | 1 | 9,713 |
| clutterfunk | 1 | 11,547 |
| cycles | 1 | 9,545 |
| dry_out | 1 | 9,803 |
| jumper | 1 | 10,421 |
| polargeist | 1 | 10,858 |
| stereo_madness | 1 | 10,338 |
| theory_of_everything | 3 | 5,330 |
| time_machine | 1 | 11,529 |
| xstep | 1 | 9,845 |
| **TOTAL** | **15** | **118,879** |

**Critical observation**: only **one macro per level** for most levels. Zero behavioral diversity per level.

### Object filtering

The mod filters out coins (`Collectible`, `UserCoin`, `SecretCoin`) and decorations from the obs. Even though Eclipse macros take coin-grabbing paths, the agent never "sees" coins — so it can't be biased toward coin-seeking. Object IDs are taken from the GD modding community's standard table.

---

## Behavioral Cloning (BC)

### Setup

`src/gdrl/train/imitation.py`:

- Loss: `BCEWithLogitsLoss(pos_weight=neg_count/pos_count)` to handle class imbalance (jump is ~30% of frames in our data)
- Optimizer: Adam, lr=3e-4
- Per-frame normalization computed via Welford's algorithm; saved as `<model>.norm.npz`
- Frame stacking with episode-boundary protection (replicate first frame at episode start)
- Train/val split: either random per-frame fraction OR hold out an entire level

### Two BC models trained

| Checkpoint | Train data | Val data | Best val_loss |
|-----------|-----------|----------|---------------|
| `bc_model.pt` | 11 levels (no stereo_madness) | stereo_madness (held out, temporal order) | 0.74, evt_f1=0.50 |
| `bc_stereo_only.pt` | stereo_madness only | random 10% of stereo_madness frames | 0.10, evt_f1=0.98 |

### Evaluation metrics

Built a custom event-level metric (`src/gdrl/eval/offline_metrics.py`):
- Extract contiguous "jump events" (runs of action=1) from human and model predictions
- Match events with ±15 frame tolerance
- Report event-level precision/recall/F1

This is meant to forgive timing differences (jumping at frame 277 vs 280 should both count as "got the obstacle"). Per-frame accuracy is misleading because the loss penalizes any deviation.

### What went wrong with BC

Despite `bc_stereo_only.pt` achieving val_loss=0.10 and evt_f1=0.98 on validation, **it dies at the first spike during live play**. Even on the same level it memorized.

**Three compounding failure modes:**

#### 1. One recording per level = zero behavioral diversity

The macro pressed jump at exactly frame 280 every time. Model learns "jump exactly when distance = 40, on_ground=1, vy=0" — a razor-thin condition. Any inference deviation (subpixel timing, slight obs differences) and the trigger fails.

Compare to having 5 macros where the jump happens at frame 277, 280, 283 — the model would learn "jump when distance is 30-50" which is a robust condition.

#### 2. Distribution shift between training and inference (THE classic BC failure)

In training: by x=460, the player is **already in the air rising** because the macro pressed jump 5 frames earlier. So `(x=460, on_ground=1, vy=0)` essentially **never occurs** in training data.

At inference: the model takes a slightly off action, ends up at x=460 still on ground — a state it has never seen — and gets garbage predictions. Errors compound; the player can't recover.

Confirmed via offline sanity check: model is 99.7% accurate replaying the recorded shard, but the recorded states never match what live play produces because by the time live reaches x=460, the rollout has diverged from the macro's path.

#### 3. Frame-level supervision is the wrong granularity

BCE loss penalizes the model for jumping at frame 275 (when the macro jumped at frame 280) — even though either frame would clear the obstacle. This forces the model toward overfit point predictions instead of learning timing windows.

`evt_f1` measures the right thing for this game, but the loss function doesn't optimize for it.

### What we tried for BC

- ✅ Held-out level for true generalization eval (`--val-level`) — confirmed brittleness
- ✅ Type embeddings vs flat one-hot — embeddings strictly better (smaller, more generalization)
- ✅ Sub-categorization of obstacles (yellow orb vs grav orb etc.) — necessary; can't collapse them all
- ✅ Frame stack of 4 — sufficient for short-term context
- ❌ Multiple macros per level — never collected; would have helped most
- ❌ Frame-tolerance loss (e.g., correct if jump within window) — not implemented
- ❌ Dropout / weight decay regularization — not added; model still overfits hard

---

## PPO Live Training

### Setup

`src/gdrl/train/ppo.py` (teammate's implementation, kept over my SB3 version):

- Pure PyTorch (no SB3 dependency)
- Uses GDPolicyMLP directly — its single logit naturally pairs with `Bernoulli` distribution for binary action
- Standard PPO with clipped objective + GAE advantages
- TensorBoard logging
- Saves checkpoints in same format as BC, so `live_eval.py` works on PPO checkpoints

### Env wrapper

`PreprocessedStackedEnv` (now `_StackedEnv` in ppo.py) wraps `GDPrivilegedEnv`:
- Maintains 4-frame stack of raw obs
- Applies `ObsPreprocessor` per frame
- On reset: replicates the first frame across all stack slots (matches training)

### Reward function (current)

```
reward = progress * 0.1 - 0.01    (per step)
       - 10 if dead
       + 100 if level_done
```

Where `progress = current_x - prev_x` per frame.

### How action sampling works (key concept)

- **During PPO training**: actions sampled from `Bernoulli(logits=logit)`. The neural net is deterministic; the action is stochastic. With `logit = +5`, P(jump) = 0.99 — almost always jumps. With `logit = 0`, coin flip. This is **required for exploration**.
- **During live_eval**: pure greedy. `action = 1 if logit > 0 else 0`. No randomness — pure deterministic execution of the learned policy.

The entropy coefficient (`--ent-coef`) explicitly rewards stochasticity in training to prevent premature collapse. Higher = more exploration. Lower = more deterministic but risks getting stuck.

### Bug we found and fixed

Discovered while debugging "first-frame-of-new-episode jump" anomalies:

**Mod globals (`g_prevX`, `g_actionWasPressed`) weren't reset on `resetLevel()`**. Only on `PlayLayer::init` (which runs only when entering a level fresh). This caused:
1. First frame after death: `dx = 0 - death_x = -460` (huge garbage value)
2. Button state mismatch — Python's intended action got dropped if the previous death had button held

Fixed in two places:
- Mod (when rebuilt): explicitly reset both globals on send_reset and on death-detected
- Python workaround in `privileged_env.reset()`: burn 2 frames with action=0 after reset to flush stale globals

### What we tried with PPO

#### v1: vanilla PPO with `bc_model.pt` warm-start

Used a custom SB3 ActorCriticPolicy that wrapped GDPolicyMLP. Discarded after teammate provided cleaner pure-PyTorch implementation.

#### v2: conservative hyperparameters

```
--lr 1e-4 --ent-coef 0.05 --clip-range 0.1 --n-steps 4096 --n-epochs 10
```

**Result**: regressed faster than v1 ironically. Higher entropy = more chaotic deaths = bigger swings in advantages = more drift.

#### v3 (the "best so far" run): used `bc_stereo_only.pt` as anchor

```
--bc-checkpoint bc_stereo_only.pt --lr 5e-5 --ent-coef 0.01 --clip-range 0.1
--n-steps 4096 --n-epochs 4
```

Ran for 160 updates. Pattern observed:
- Early: ep_rew climbs 35 → 76 over a few rollouts
- Then a really lucky run gets past 3 platforms after 3 spikes
- Update happens → **regresses to dying at first 2 spikes for many runs**
- Slowly recovers back to first-spike behavior
- Cycle repeats

#### v4-v5: aggressive reward shaping

Tried per-episode `new_max_bonus` (0.5 per unit of new x), bigger death penalty (-50). 

**Result**: collapsed to "spam jump" because the bonus reset per episode just amplified progress reward 6×. With low entropy, policy went nearly deterministic on the wrong behavior. Reverted.

#### v6-v7: BC anchor (KL divergence penalty)

Added KL term to PPO loss:
```
total_loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy + kl_coef * KL(current || BC)
```

The intent: prevent policy from drifting away from BC. Even if PPO wants to push P(jump) far away, the KL pulls it back.

Tried `--kl-coef 0.1` then `--kl-coef 1.0`. **Both still suffered catastrophic forgetting.** With kl=0.1, the KL term contributed ~0.018 to loss while PG was ~0.05 — KL got out-weighted. Even at 1.0, the policy still drifted in subtle ways the KL didn't catch.

#### v9 (current target): back to v3 hyperparameters, no KL anchor

The KL anchor wasn't helping enough to justify the constraint it imposed. Reverting to v3-style:

```
--bc-checkpoint bc_stereo_only.pt --lr 5e-5 --ent-coef 0.01 --clip-range 0.1
--n-steps 4096 --n-epochs 4 --total-steps 400000
```

Goal: let it run to ~110 updates and observe the lucky→regress→recover pattern. If net positive over many cycles, it's working slowly. If net negative or stuck oscillating, deeper changes needed.

---

## Why PPO Keeps Failing — Three Mechanisms

### 1. Lucky-run signal is diluted in rollouts

Each rollout has 4096 steps ≈ 10-13 episodes. Suppose 1 lucky episode reaches x=900 (reward +80) and 12 fail at x=460 (reward +36 each).

PPO's gradient is computed per-state, not per-episode. Most states (e.g., "level start") appear in all 13 episodes, with 12 of them ending in early death. The lucky episode's positive signal is averaged against 12 mediocre ones. Hard to learn from rare events.

### 2. Value function inflation causes regression

After a lucky run, the value function `V(s)` learns "high return at level start" — its estimate goes up.

In the next rollout, normal episodes (which were previously good baseline behavior) now have `actual_return < V(s)` → negative advantage → policy gradient pushes **away** from those baseline behaviors.

This is the fundamental credit-assignment issue: even when failed runs had **identical correct early jumps to the lucky run**, they get punished because their *eventual* outcome was below the inflated expectations. There's no way to separate "you did the early jumps right" from "you died at obstacle 3."

### 3. Catastrophic forgetting via shared representations

Every weight in the network contributes to every prediction. Updates to handle "platform clearing" inadvertently change the same hidden units used for "first spike timing."

The model has no isolated "spike-1 jump neuron" — it's all distributed. When PPO nudges weights to improve one behavior, it disturbs all others. Stereo madness has spikes, platforms, and ship sections — three very different regimes sharing the same network parameters.

### 4. Stochastic sampling adds noise

Even a "correct" policy with `logit = +1.5` has only 0.82 probability of jumping per frame. To hold jump for 5 consecutive frames: `0.82^5 = 0.37`. **63% failure rate purely from sampling**, even with a model that has learned the right thing.

This is why training rollouts often look terrible relative to the model's actual learned skill. Live_eval (deterministic) sometimes performs much better than recent rollout rewards suggest.

---

## What This Means / Honest Assessment

### What works

- **Pipeline mechanics**: the mod, IPC, env, preprocessor, model, training scripts all function correctly. We've verified through multiple sanity checks (offline replay = 99.7% accuracy on training data).
- **BC trains the model**: it can fit data perfectly and reach high val metrics on validation.
- **PPO updates the model**: TensorBoard shows the policy and value function learning.

### What doesn't work

- **BC generalizes poorly**: brittleness from one-recording-per-level. Live play fails at first obstacle.
- **PPO can't extract enough signal from rare lucky episodes**: regression after every lucky breakthrough. Live_eval after 200K steps doesn't show clear sustained improvement.

### Why this is fundamentally hard

Geometry Dash + one macro per level + on-policy RL is at the edge of what's feasible. The combination of:
- Precise timing requirements (5-frame margin for spike clears)
- Sparse positive examples (lucky runs are rare)
- Shared network architecture (gradient interference)
- Single live game instance (no parallelism, ~60fps cap)

…means each rollout is expensive, each update is noisy, and each lucky episode produces only a small positive nudge.

### Realistic paths forward

In rough order of expected impact:

1. **Multiple macros per level** (3-5+ per level with timing variation) — addresses BC brittleness directly. The biggest single lever.
2. **DAgger** — collect rollouts from current model, label what the macro would do at *those* states, retrain BC iteratively. Directly addresses the distribution-shift problem.
3. **TPS bypass mod** — let the game run at 4-8× speed. Makes 5-10× more training feasible, moving PPO from "hours of game time" to "tens of minutes."
4. **Off-policy RL** (SAC, DQN) — replay buffer lets you learn many times from each rare success. Different framework but more sample-efficient.
5. **Better reward shaping** — persistent across-episode max-x novelty bonus (correctly implemented this time).

What we're currently betting on: **PPO with conservative settings + patience**, ideally combined with #3 (TPS bypass) so we can run long enough to overcome the variance.

---

## Files Touched

```
mods/TrainingPipeline/src/main.cpp     V3 ring buffer mod, action injection, reset bug fixes
src/gdrl/env/geode_ipc_v3.py           Python adapter for V3 SHM
src/gdrl/env/privileged_env.py         Gym env, reward shaping, post-reset frame burn
src/gdrl/data/record_human.py          Recorder (death-stop, level-stop, dedup)
src/gdrl/data/obs_dataset.py           HumanPlayDataset, level-based train/val split
src/gdrl/model/obs_preprocess.py       42-type ID mapping, density-aware subsampling
src/gdrl/model/mlp_agent.py            GDPolicyMLP with embedding expansion
src/gdrl/train/imitation.py            BC trainer, --val-level, --pretrained, evt_f1
src/gdrl/train/ppo.py                  PPO trainer with optional --kl-coef BC anchor
src/gdrl/eval/offline_metrics.py       Jump-event matching with temporal tolerance
src/gdrl/eval/sanity_check.py          Model on recorded data (used to debug live_eval)
src/gdrl/eval/live_eval.py             Live in-game evaluation, --verbose
tests/test_obs_preprocess.py           Preprocessor unit tests
tests/test_mlp_agent.py                Model architecture tests
tests/test_offline_metrics.py          Event-matching tests
```

## Current Status

- BC: working, but not generalizing well. `bc_stereo_only.pt` and `bc_all_levels.pt` available.
- PPO: implemented and trains stably, but doesn't reliably break through obstacles. Best observed behavior: brief glimpses of getting past 3 platforms before regressing.
- Awaiting decision: continue PPO tuning vs pivot to DAgger or off-policy RL or TPS bypass.
