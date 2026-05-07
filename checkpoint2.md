# GD RL Checkpoint 2 — Snippet-PPO + Frontier + SIL

This checkpoint documents the iteration on `src/gdrl/train/ppo.py` from the
sawtooth-collapsing PPO of [checkpoint1.md](checkpoint1.md) toward a snippet-based PPO
with progress-based exploration and self-imitation learning.

---

## Starting state and core problem

Per checkpoint1.md, vanilla PPO suffered three compounding failure modes:

1. **Lucky-run signal dilution** — 1 good episode in a 13-episode rollout gets averaged against 12 mediocre ones.
2. **Value inflation** — after a lucky run, `V(s)` rises, so next rollout's average behavior gets negative advantage and the policy is pushed away from baseline behavior.
3. **Catastrophic forgetting** — shared backbone weights mean updates for one part of the level disturb others.

This entire iteration was an attempt to design around these mechanisms.

---

## v1: Snippet-PPO baseline

### Why
Expert critique said GD has short-range causal structure (unlike chess) — surviving = winning, dying = the only thing that's bad. So credit assignment should be **short-horizon and local**, not full-episode.

### How
Rewrote [src/gdrl/train/ppo.py](src/gdrl/train/ppo.py) end-to-end:

- **Snippet definition**: slice each episode into K=20 non-overlapping frames. Death-spanning snippets kept and padded; partial trailing non-died snippets dropped.
- **Cross-rollout `SnippetBuffer`**: capacity 4096, FIFO eviction, max-age=4 to bound staleness. Stratified sampling with a 10% death-snippet floor.
- **n-step truncated return** with bootstrap from `V(s_{t+K})` at collection time:
  ```
  G_t = sum_{i=t..t+K-1} gamma^(i-t) r_i + gamma^K · V(s_{t+K})
  ```
- **PPO update**: 4 policy+value epochs followed by 1 value-only epoch (PPG-lite, targeting failure mode #2).
- **Reward**: per-frame `progress * 0.1`, death-frame `-3.0`, level-complete bonus `+10` (later removed).
- **Kept BC warm-start + KL anchor** annealing from 0.3 → 0.05.

### Downsides
- Bootstrap reintroduces value inflation — only at snippet granularity instead of episode granularity.
- Dense progress reward is redundant (GD scrolls at near-constant speed).
- gamma=0.99 over K=20 is inconsistent with "short-horizon" framing.

### Outcome
Same sawtooth pattern as before. Brief improvement, then collapse. Confirmed snippet structure alone isn't enough.

---

## v1.1: Binary reward, lower gamma, higher death floor

### Why
Critic feedback flagged three reward-shape defects:
1. `progress * 0.1` is essentially a survival counter — adds noise, no information beyond the death event.
2. gamma=0.99 over K=20 still weights frame-19 reward at 0.83× from frame 0. Defeats the "short-horizon" premise.
3. Death floor of 10% under-samples the only meaningful learning event class.

### How
- Stripped per-frame progress reward; alive frames now have `r=0`.
- Added `+survival_bonus` (default 1.0) at the **last valid frame** of each non-died snippet (in `_episode_to_snippets`, override the local rewards copy before computing returns).
- `--gamma 0.99 → 0.9`. With K=20 this gives `gamma^19 = 0.135`.
- `--death-floor-frac 0.1 → 0.3`.
- Removed `--progress-scale`; added `--survival-bonus`.

### Downsides
- Returns become a closed-form function of (a) did snippet survive, (b) death frame index. Cleaner signal but very sparse.
- The last frame's survival bonus is a single point of credit — early frames in K=20 still get gamma^19 ≈ 0.13 of the signal.

### Outcome
Tested briefly — same regression pattern (3-update-then-collapse) on first run. Pushed to v1.2.

---

## v1.2 part A: K bumped to "5 seconds" (then 3 seconds)

### Why
User reported the model "kept forgetting" early-game behavior, hypothesized longer snippets would preserve more context per gradient update.

### How
- Empirically measured tick rate from recordings (`stereo_madness` ≈ 10,380 frames per session at 91-second song length → ~115 fps, *not* 60). Mod hooks `PlayLayer::postUpdate` per visual frame on macOS ProMotion.
- Set `--snippet-len 360` (3 sec at 120 fps).
- Bumped `--gamma 0.9 → 0.99` (with K=360, 0.9 makes early-frame returns negligible: `0.9^359 ≈ 3e-14`).
- Reduced buffer to 256 and batch-snippets to 8 to bound memory.

### Downsides (became apparent after testing)
With K=360 spanning multiple obstacles, **deaths at obstacle N+1 contaminate the gradient on the successful jump at obstacle N within the same snippet**. The user observed: agent clears spike 1 sometimes, then dies at spike 2; "after the update, P(jump) at spike 1 collapses." Long snippets actively harm credit attribution across obstacles.

### Outcome
Walked back to K=60.

---

## v1.2 part B: K reduced to 60

### Why
After the cross-obstacle contamination diagnosis, settled on a length that covers approximately one obstacle's decision window (~0.5s at 120fps).

### How
`--snippet-len 60`. Other params unchanged.

### Outcome
Better credit isolation, but still didn't solve the underlying regression. Necessary but not sufficient.

---

## v1.2 part C: Drop the bootstrap entirely

### Why
This was the conceptual fix. Walking through what bootstrap actually does:

The value head `V` is a single function over states, shared across all positions in the level. When the agent has a lucky run, V learns positive samples at late-level states. The bootstrap `gamma^K · V(s_{t+K})` then propagates that increase **backward** into the return computation of earlier snippets, which becomes the regression target for V at those earlier states, cascading inflation all the way to V(s_0).

Setting `bootstrap = 0` severs the channel. Each snippet's return becomes a closed-form function of (a) did this 60-frame window survive, (b) where the death was within the window. Position-symmetric: clearing the 5th obstacle yields the same return as clearing the 1st.

### How
Single point change in `_episode_to_snippets`:
```python
# was:
if died_within: bootstrap = 0.0
elif end < L:   bootstrap = ep_values[end]
else:           bootstrap = final_value

# now:
bootstrap = 0.0
```
Removed `final_value` parameter and the `final_v` computation at rollout boundaries.

Verified empirically with synthetic test: two episodes that differ only in death position produce **identical first-snippet returns**. Setting `ep_values=10` (simulating an inflated value head) does not affect any snippet's returns under bootstrap=0.

### Downsides
- Lose cross-snippet value flow. In standard RL this would matter (you can't grade an early move without knowing the eventual outcome). In our setting each snippet contains its own complete outcome (alive/dead at terminal), so we don't need it.
- Sample efficiency decreases for true long-horizon learning. But the 60-frame snippet is itself the horizon we care about.

### Outcome
Eliminated the documented value-inflation cascade. Necessary fix. Still didn't fully stabilize training because PPO has other failure modes (per below).

---

## v1.3: Value-head warmup + bigger batches + frontier warmup

### Why
After v1.2, the user reported: "first rollout, 3/20 episodes clear spike 1; after one update, 0/20 clear, the agent never even attempts to jump." Catastrophic post-first-update collapse, despite bootstrap fix.

Root cause: the **value head is randomly initialized** (BC trains the action head only). On the first PPO update, V(s) is noise → advantages = G - V are noise → per-minibatch normalization amplifies noise → policy shifts dramatically in random directions. Combined with a small batch size (8 snippets × 60 frames = 480 frames per minibatch), one bad update destroys the BC initialization.

### How
Three independent additions:

**1. Value-head warmup** (`--value-warmup-updates 3`): for the first N updates, run `_update` with `n_epochs=0` (no policy gradient at all), only value-only epochs. Total epochs during warmup = `n_epochs + value_epochs` to match normal compute. Verified via unit test that the action_head's weights are byte-identical before and after a warmup update.

**2. Bigger minibatches** (`--batch-snippets 8 → 32`): 32 snippets × K=60 = 1920 frames per minibatch. Closer to standard PPO's 1k–2k range.

**3. Frontier warmup** (`--frontier-warmup-updates 5`): don't apply progress-based exploration's greedy gating until update > 5. Until then, pure stochastic. Prevents the frontier from locking in a bad early policy.

### Downsides
- 3 updates of training time spent on value-only fitting.
- Frontier warmup is a band-aid for the frontier's own failure mode (see below).
- Doesn't address the structural noise PPO introduces on every update, only the worst case at update 1.

### Outcome
First-update catastrophe was tamed (V-warmup did its job, `vf_loss` converged to ~0.05 by update 4). But the policy still degraded over later updates — the failure mode just shifted from "update 1 destroys BC" to "updates 5+ slowly degrade the policy."

---

## Progress-based (frontier) exploration

### Why
User observation, well-stated: stochastic sampling at confident states wastes gradient. If the policy already knows to clear spike 1 reliably, sampling action=0 there sometimes just produces deaths that contaminate the gradient with bad signal. Exploration should be **state-conditional** — only stochastic at the frontier of explored progress.

### How
- Track `recent_max_x` in a `deque(maxlen=--frontier-history)` (default 20 rollouts).
- Each rollout: `frontier_x = --frontier-frac × max(recent_max_x)` (default frac=0.8).
- During action sampling in `_collect_rollout`:
  ```python
  if cur_x < frontier_x:
      action = int(logit > 0.0)              # greedy
  else:
      action = Bernoulli(logits=logit).sample()
  ```
- `log_prob` is **always** computed under the stochastic policy regardless of which branch we took, so PPO's importance ratio still works.
- Surfaced `frontier/x`, `frontier/frac_greedy`, `frontier/recent_max_x`, `rollout/rollout_max_x` to TB.
- `_StackedEnv.reset()` and `step()` augmented to expose `x`, `progress`, `is_dead` in their info dicts.

### Downsides
This is the part that bit us. When the agent has a single lucky rollout (e.g., x=1437) but then regresses, the deque keeps that high value for `--frontier-history` rollouts. So:
- `frontier_x` stays at `0.8 × 1437 = 1149`
- Agent now dies at x=507 every rollout, never reaches frontier
- `frac_greedy = 100%` — every action deterministic
- Stuck in closed loop: greedy → deterministic death → no new data → no new high-progress trajectory → frontier doesn't shrink

The fix would be to make the frontier **conditional on whether the agent is currently reaching it** (e.g., shrink frontier if the rolling median of `rollout_max_x` falls well below it), but this hasn't been implemented.

### Outcome
Helps in principle, hurts in practice unless training is already on a working trajectory. Currently recommended to disable (`--frontier-frac 0`) until SIL is reliably ratcheting forward.

---

## Self-imitation learning (SIL)

### Why
After all of the above, the policy still oscillated between brief progress and collapse. The fundamental issue: **PPO's gradient is inherently destabilizing in sparse-reward regimes**. Every fix so far is "make PPO less harmful." None preserve and reinforce what works.

The user's intuition: if the agent succeeds at clearing some obstacles, it should remember that. PPO doesn't do this — it only updates relative to V, which is itself trained from the policy's noisy data.

SIL (Oh, Guo, Lee, Lewis, Singh, ICML 2018) is a stable counterweight: maintain a buffer of best trajectories, train an auxiliary loss that is **positive-only** — pulls policy toward stored actions when realized return exceeds V, never pushes away from anything.

### How
**`BestSnippetBuffer`** (new class):
- Capacity 256 by default.
- Snippets ranked by `parent_episode_max_x` (the max x reached in the episode the snippet came from), descending.
- No FIFO, no age eviction. Adding new snippets sorts and trims to top-N. A snippet only leaves the buffer when a strictly-better snippet displaces it.

**Snippet dataclass** extended with `parent_max_x: float = 0.0`. `_episode_to_snippets` and `_collect_rollout` thread the per-episode max-x through.

**`_sil_update`** function (runs after PPO + value-only epochs each main update, **skipped during V-warmup** since SIL is a policy update):
```python
adv = (R − V).clamp(min=0)                         # positive-only
adv_for_actor = adv.detach()
actor_loss = -log π(a|s) · adv_for_actor           # cannot push away
critic_loss = 0.5 · adv²                           # pulls V up only
sil_loss = sil_coef · (actor_loss + vf_coef · critic_loss)
sil_loss.backward()
optimizer.step()
```

Defaults: `--sil-buffer-size 256`, `--sil-epochs 1`, `--sil-batch-snippets 16`, `--sil-coef 0.1`.

Logging: `sil/buffer_size`, `sil/buffer_min_max_x`, `sil/buffer_max_max_x`, `sil/actor_loss`, `sil/critic_loss`, `sil/pos_frac`.

Verified via unit tests:
- All-negative-advantage SIL minibatch produces zero action_head gradient.
- All-positive-advantage SIL minibatch produces nonzero gradient with `pos_frac=100%`.
- Buffer correctly admits high-rank snippets and rejects low-rank ones.

### Downsides
- **Stale behavior.** Stored actions are from older policies. SIL ignores this (no importance ratio); biased but accepted in the original SIL paper.
- **Gradient magnitude vs PPO.** With `sil_coef=0.1` and 1 SIL epoch vs 4 PPO epochs, SIL contributes ~5–10× less total gradient per update than PPO. Currently being outweighed.
- **Buffer can become stale.** If the agent never reaches new high-x territory, the buffer holds the same snippets forever. SIL keeps re-imitating an old success that the current policy can't reproduce.
- **Diversity collapse.** Top-N by parent_max_x can fill with snippets all from the same lucky rollout — high concentration on one specific trajectory. No stratification by level position.

### Outcome
SIL works mechanically — buffer captures lucky trajectories (`sil/buffer_max_max_x` ratcheted from 958 → 1437), `pos_frac` is healthy at ~50%. But the gradient is being outweighed by PPO. With current default coefficients SIL alone hasn't broken the regression pattern.

Recommended next experiment: bump `--sil-coef 0.1 → 0.5`, `--sil-epochs 1 → 3`, and disable the frontier (`--frontier-frac 0`) to roughly equalize SIL's per-update gradient against PPO and prevent the policy from being locked into deterministic regression.

---

## Sparse reward design (final form)

The reward function evolved across versions and ended up as:

```
r_t (during step)              = 0                      while alive
r_t (set in _collect_rollout)  = -death_penalty         on death frame (default 1.0)
r_t (set in _episode_to_snippets) = +survival_bonus     at last valid frame
                                                         of non-died snippet
                                                         (default 1.0)
no level-complete bonus.
bootstrap = 0 always.
gamma = 0.99 (default after K=60 was settled).
```

For a survived snippet: `G_t = gamma^(K-1-t) × survival_bonus`. Closed form, position-independent.

For a died-at-frame-d snippet: `G_t = -gamma^(d-t) × death_penalty` for valid frames `[0, d]`, zero on padded frames `[d+1, K-1]`.

---

## Files modified

```
src/gdrl/train/ppo.py    Full rewrite. Now contains:
                         - _StackedEnv (frame stacking + info-dict augmentation)
                         - Snippet dataclass (with parent_max_x for SIL ranking)
                         - SnippetBuffer (FIFO + age eviction + death-floor sampling)
                         - BestSnippetBuffer (top-N by parent_max_x for SIL)
                         - _episode_to_snippets (slicing + reward shaping + closed-form returns)
                         - _collect_rollout (frontier-gated action sampling)
                         - _flatten_batch (snippet → frame-flat tensor for losses)
                         - _update (PPO + KL anchor + value-only epoch + V-warmup gate)
                         - _sil_update (positive-only auxiliary loss)
                         - _kl_coef_at (linear annealing)
                         - main (CLI, BC warm-start, training loop)

src/gdrl/env/privileged_env.py   UNCHANGED.
src/gdrl/model/mlp_agent.py      UNCHANGED.
src/gdrl/model/obs_preprocess.py UNCHANGED (already has PROCESSED_PLAYER_DIM=8 from "Fix BC" commit).
```

---

## Honest assessment

What works mechanically:
- Snippet structure is sound; tested with synthetic rollouts; closed-form return math verified.
- Bootstrap removal eliminated value-inflation cascade as a failure mode.
- Value-head warmup tamed the catastrophic-first-update issue.
- SIL buffer correctly captures and ranks the best trajectories.

What still doesn't work:
- After ~10 updates, the agent regresses from the brief peaks (x=958, x=1437) to a stable plateau around x=500–600.
- Frontier locks in regressed policy when `recent_max_x` doesn't fall fast enough.
- SIL's gradient is currently too weak to fight PPO's drift.
- Entropy hovers around 0.05 (very low) → policy is highly deterministic per-state, so even small mistakes are reproduced every rollout.

Suspected structural issue, not yet addressed:
- Stochastic Bernoulli sampling at every frame produces lots of "wasted" exploration in regions the policy has already mastered, generating dying samples that contaminate the gradient. The frontier was an attempt to fix this but introduced its own pathologies.
- Without parallelism or game-speed bypass, gradient updates per second are bottlenecked by single-game-instance rollout speed (~110 fps).

Realistic paths forward, in order of expected impact:
1. **Make frontier adaptive** — shrink when median rollout_max_x falls below frontier; reverts to stochastic when policy regresses.
2. **Strengthen SIL** — `sil_coef=0.5`, `sil_epochs=3`, OR bias SIL buffer to also include the most recent decent rollouts (not just all-time best) to keep the buffer fresh.
3. **Action repeat / sticky actions** — make each "decision" cover N frames; reduce per-frame sampling variance.
4. **More BC data (DAgger)** — the original biggest-lever fix from checkpoint1.md is still the biggest-lever fix.
5. **TPS bypass** — let game run at 4–8× speed for more training iterations per wall-clock hour.
