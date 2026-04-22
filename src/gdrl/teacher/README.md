# teacher/

Training pipeline for the privileged teacher policy. The teacher reads `obs[608]` directly from game memory via the Geode mod — it has full knowledge of player state and nearby obstacles at every frame. It is trained in three phases: BC warmup, DAgger iterations, and PPO fine-tuning.

---

## Files

| File | Purpose |
| --- | --- |
| `model.py` | `TeacherPolicy` — shared MLP architecture used across all three phases |
| `bc_train.py` | Phase 0: behavioral cloning warmup on human recordings |
| `dagger_loop.py` | Phase 1: orchestrates the full DAgger loop end to end |
| `train_ppo.py` | Phase 2: PPO fine-tuning after DAgger converges |

The DAgger data collection scripts live in `src/gdrl/data/`:

| File | Purpose |
| --- | --- |
| `data/record_human.py` | Record a human playing the level to `.npz` shards |
| `data/dagger_rollout.py` | Roll out the current policy in the game and record visited states |
| `data/dagger_align.py` | Label rollout states with human actions via X-position matching |
| `data/dagger_dataset.py` | Aggregate human recordings + all DAgger labeled files into one dataset |
| `teacher/dagger_loop.py` | Orchestrator that runs one full DAgger iteration end to end |

---

## The model

`TeacherPolicy` is a 3-layer MLP (~689k parameters):

```
input:  4 stacked obs frames  →  4 × 608 = 2432 floats  (oldest → newest)
hidden: 256 → 256
output: 2 logits  →  [no-jump, jump]
```

Frame stacking gives the model temporal context — it can see velocity and acceleration implicitly from how position changes across frames, not just a static snapshot.

---

## Phase 0 — Behavioral cloning warmup

### What it is

Pure supervised learning on human gameplay. The model is trained to predict the human's action (jump / no-jump) at each frame. No game interaction — it's standard classification on recorded data.

The goal is not a good policy — it's a non-random starting point. A random policy dies on the first obstacle every time, making DAgger's early rollouts useless. Even a mediocre BC policy that survives a few seconds gives DAgger meaningful failure states to learn from.

### Data collection

Open Geometry Dash, enter the level, then run:

```bash
python -m gdrl.data.record_human --out artifacts/recordings/ --shard-size 10000
```

Play normally — die, retry, repeat. The recorder runs in the background capturing every frame at 240fps. Record at least 30–60 minutes of gameplay across many attempts so there is human label coverage at every X position in the level.

Each run creates a new timestamped session directory so sessions never overwrite each other:

```
artifacts/recordings/
    20260408_143022/
        shard_00000.npz
        shard_00001.npz
        ...
    20260408_160511/
        shard_00000.npz
        ...
```

### Training

```bash
python -m gdrl.teacher.bc_train \
    --data artifacts/recordings/ \
    --out artifacts/bc_warmup.pt \
    --epochs 10 \
    --batch-size 256
```

Saves the best checkpoint (lowest validation loss) to `artifacts/bc_warmup.pt`. This checkpoint is the starting policy for DAgger iteration 0.

---

## Phase 1 — DAgger iterations

### What DAgger is

Behavioral cloning only trains on states the human visited. When the policy makes a mistake during deployment it ends up in a state it has never been trained on, makes another mistake, and errors compound. This is called **distribution shift**.

DAgger (Dataset Aggregation) fixes this by repeatedly asking: *what states does the current policy actually visit?* and collecting expert labels specifically for those states. Over several iterations the policy's visited states and the training data converge — there is nowhere the policy can end up that it hasn't been trained on.

Each iteration has three steps:

### Step 1 — Roll out the current policy

Run the policy in the live game and record every state it visits, including its failure states:

```bash
python -m gdrl.data.dagger_rollout \
    --policy artifacts/bc_warmup.pt \
    --episodes 50 \
    --out artifacts/rollouts/iter1.npz
```

On iteration 2+, point `--policy` at the previous iteration's checkpoint:

```bash
python -m gdrl.data.dagger_rollout \
    --policy artifacts/dagger_iter1.pt \
    --episodes 50 \
    --out artifacts/rollouts/iter2.npz
```

This runs unattended — no human required.

### Step 2 — Collect human labels for this iteration

Play the level once more while the recorder runs. This gives fresh human labels across the level's X positions:

```bash
python -m gdrl.data.record_human --out artifacts/recordings/ --shard-size 10000
```

5–10 minutes of play is enough per iteration. You just need human coverage at the X positions the policy reached in Step 1.

### Step 3 — Align and aggregate

Match the policy's visited states to human labels by X position:

```bash
python -m gdrl.data.dagger_align \
    --rollout artifacts/rollouts/iter1.npz \
    --human-data artifacts/recordings/ \
    --out artifacts/dagger_labeled/iter1.npz
```

For each frame the policy visited, the aligner finds the nearest human frame at the same X position and uses the human's action there as the correct label. The observations come from the policy's trajectory; the labels always come from the human.

Then aggregate all past data into one growing dataset and retrain:

```bash
python -m gdrl.data.dagger_dataset \
    --human-data artifacts/recordings/ \
    --labeled-dir artifacts/dagger_labeled/ \
    --out artifacts/dagger_dataset_iter1/
```

`dagger_dataset.py` combines the original human recordings with every `.npz` file in `--labeled-dir` (all past DAgger iterations). It writes the result as shards in `--out` in the same format `HumanPlayDataset` reads, so `bc_train.py` works unchanged:

```bash
python -m gdrl.teacher.bc_train \
    --data artifacts/dagger_dataset_iter1/ \
    --out artifacts/dagger_iter1.pt \
    --epochs 10
```

The dataset only grows — data from earlier iterations is never discarded. This is what makes DAgger stable.

### Full iteration loop

The orchestrator runs all three steps automatically, pausing at step 2 each iteration to prompt you to play the level:

```bash
python -m gdrl.teacher.dagger_loop \
    --policy artifacts/bc_warmup.pt \
    --recordings artifacts/recordings/ \
    --out-dir artifacts/ \
    --iterations 8
```

To resume from a specific iteration after an interruption:

```bash
python -m gdrl.teacher.dagger_loop \
    --policy artifacts/dagger_iter3.pt \
    --recordings artifacts/recordings/ \
    --out-dir artifacts/ \
    --iterations 8 \
    --start-iter 4
```

Each iteration produces:
```
artifacts/
    rollouts/iterN.npz            ← policy rollout
    dagger_labeled/iterN.npz      ← aligned labels
    dagger_dataset_iterN/         ← aggregated training shards
    dagger_iterN.pt               ← retrained checkpoint
```

### How many iterations

Typically 5–10. Stop when the policy consistently clears the level or when the rollout coverage stops changing meaningfully between iterations.

---

## Phase 2 — PPO fine-tuning

### What it is

After DAgger the policy can imitate a human but is fundamentally capped at human performance — it was trained to copy. PPO (Proximal Policy Optimization) switches from imitation to reward optimization. It plays the game live and reinforces whatever leads to more reward, discovering strategies the human never used.

Starting from the DAgger checkpoint means PPO immediately gets reward signal from its first episodes (the policy already survives for a meaningful distance) instead of exploring randomly for thousands of episodes.

The reward function in `GDPrivilegedEnv`:
- `+0.1 × x progress per frame` — reward for moving right
- `-0.01` per step — pressure to be fast
- `-10` on death
- `+100` on level completion

### Training

```bash
python -m gdrl.teacher.train_ppo \
    --checkpoint artifacts/dagger_iter8.pt \
    --timesteps 500000 \
    --out artifacts/teacher_final
```

The DAgger weights are transferred directly into the PPO actor network so training starts from the imitation-learned baseline. Saves the final teacher to `artifacts/teacher_final.zip`.

---

## Full pipeline summary

```
1. Record human gameplay
   python -m gdrl.data.record_human --out artifacts/recordings/

2. BC warmup
   python -m gdrl.teacher.bc_train --data artifacts/recordings/ --out artifacts/bc_warmup.pt

3. DAgger loop (5-10 iterations)
   python -m gdrl.teacher.dagger_loop --policy artifacts/bc_warmup.pt ...

4. PPO fine-tuning
   python -m gdrl.teacher.train_ppo --checkpoint artifacts/dagger_iterN.pt
```
