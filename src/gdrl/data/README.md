# data/

Data collection and dataset utilities for the teacher training pipeline.

## Files

| File | Purpose |
| --- | --- |
| `record_human.py` | Record human gameplay to `.npz` shards via the V3 ring buffer |
| `obs_dataset.py` | `HumanPlayDataset` — loads shards into a PyTorch dataset with frame stacking |
| `dagger_rollout.py` | Roll out a policy in the live game and record every visited state |
| `dagger_align.py` | Label rollout states with human actions via X-position matching |
| `dagger_dataset.py` | Aggregate human recordings + all DAgger labeled files into one dataset |
| `collector.py` | `write_shard()` helper for student distillation data (teacher phase) |
| `collector_cli.py` | CLI for collecting student distillation data — runs teacher, captures frames |

---

## Human recording

Records a human playing the level. Drains the V3 ring buffer at 240fps and writes compressed `.npz` shards to a timestamped session directory.

```bash
python -m gdrl.data.record_human --out artifacts/recordings/ --shard-size 10000
```

Each run creates a new subdirectory so sessions never overwrite each other:

```
artifacts/recordings/
    20260408_143022/
        shard_00000.npz   ← frames 0–9,999
        shard_00001.npz   ← frames 10,000–19,999
    20260408_160511/
        shard_00000.npz
        ...
```

Each shard stores per-frame: `obs (N,608)`, `actions`, `ticks`, `episode_ids`, `is_dead`, `level_done`.

---

## DAgger data pipeline

These three scripts form the data side of each DAgger iteration. See [teacher/README.md](../teacher/README.md) for the full training context.

### 1. Roll out the current policy

Run the policy in the live game for N episodes and record every state it visits:

```bash
python -m gdrl.data.dagger_rollout \
    --policy artifacts/bc_warmup.pt \
    --episodes 50 \
    --out artifacts/rollouts/iter1.npz
```

Output `.npz`: `obs (N,608)`, `x_pos (N,)`, `policy_actions (N,)`, `episode_ids (N,)`.

Uses a rolling 4-frame history buffer so the stacked input to the policy matches what it was trained on.

### 2. Align rollout states to human labels

For each frame the policy visited, find the nearest human frame at the same X position and use the human's action as the correct label:

```bash
python -m gdrl.data.dagger_align \
    --rollout artifacts/rollouts/iter1.npz \
    --human-data artifacts/recordings/ \
    --out artifacts/dagger_labeled/iter1.npz
```

Output `.npz`: `obs (N,608)`, `actions (N,)`, `x_pos (N,)`, `episode_ids (N,)`.

The observations come from the policy's trajectory (including failure states). The labels always come from the human expert. This is what makes DAgger different from plain behavioral cloning.

### 3. Aggregate and write training shards

Combine the original human recordings with all DAgger-labeled files from every past iteration into one dataset:

```bash
python -m gdrl.data.dagger_dataset \
    --human-data artifacts/recordings/ \
    --labeled-dir artifacts/dagger_labeled/ \
    --out artifacts/dagger_dataset_iter1/
```

`--labeled-dir` should contain all `.npz` files produced by `dagger_align.py` across all past iterations — the aggregator loads every file it finds there. Old data is never discarded.

Output is a directory of `shard_*.npz` files in the same format `HumanPlayDataset` reads, so `bc_train.py` can be pointed directly at it.

---

## Dataset loading

`HumanPlayDataset` in `obs_dataset.py` loads any shard directory (human recordings or aggregated DAgger dataset) into a PyTorch Dataset:

```python
from gdrl.data.obs_dataset import HumanPlayDataset, train_val_split

# load with default 4-frame stacking
dataset = HumanPlayDataset("artifacts/dagger_dataset_iter1/")

# or get a train/val split
train_ds, val_ds = train_val_split("artifacts/dagger_dataset_iter1/", val_fraction=0.1)
```

Each item is `(stacked_obs, action)` where `stacked_obs` has shape `(4 * 608,) = (2432,)`.

---

## Student distillation data

`collector.py` and `collector_cli.py` are for the student training phase — they run the trained teacher in the game, capture screen pixels, and record the teacher's action probabilities. These are used after the teacher is fully trained. See `gdrl/student/` for the student training pipeline.
