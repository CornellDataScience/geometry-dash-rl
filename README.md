# geometry-dash-rl

CV-based Rainbow DQN with mode-gated MoE for Geometry Dash, over a Geode SHM bridge that publishes both privileged state and 128×128 grayscale frames.

Pipeline:
- Geode mod (`mods/TrainingPipeline`) writes ring-buffer state + frame mirror + accepts action/reset/load-level commands via `/gdrl_ipc_v4` shared memory.
- Python (`src/gdrl/...`) trains a `ModeGatedRainbow` (DeeperDQN backbone + 4 dueling heads gated by gamemode) with PER + n-step + NoisyNet + privileged-state aux supervision.
- Curriculum across `stereo_madness`, `back_on_track`, `polargeist`. Online checkpoint capture supports mid-level random spawn.

Old PPO/BC/DAgger code lives in `archive/` (not maintained).

## Quickstart

```bash
make setup            # venv + deps
make mod              # build & install Geode mod
# launch GD with the mod loaded, then:
make smoke            # dump 30 frames to /tmp/gdrl_smoke_*.png to verify capture
make train-sm         # single-level Stereo Madness baseline
make train            # full curriculum
make eval             # greedy eval of latest checkpoint
```

TB logs in `artifacts/dqn_cv/logs/`.

## Requirements

- Python `>=3.10`
- Python deps from `requirements.txt` (`torch`, `gymnasium`, `tensorboard`, etc.)
- Geometry Dash `2.2081` and Geode SDK

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
mkdir -p artifacts
```

Or:

```bash
make setup
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Python RL pipeline — see [src/README.md](src/README.md) |
| `mods/` | Geode mods for Geometry Dash — see [mods/README.md](mods/README.md) |
| `artifacts/` | Generated outputs: trained models and NPZ datasets (not committed) |

## Geode CLI and SDK setup (macOS)

### 1. Install the Geode CLI

```bash
curl -L https://github.com/geode-sdk/cli/releases/download/v3.7.4/geode-cli-v3.7.4-mac.zip -o geode-cli.zip
unzip geode-cli.zip
sudo mv geode /usr/local/bin/geode
geode --version
```

### 2. Configure the CLI with your Geometry Dash path

```bash
geode config setup
```

When prompted for the Geometry Dash path, use:

```
$HOME/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app
```

### 3. Install the Geode SDK

```bash
geode sdk install
```

The SDK is installed to `/Users/Shared/Geode/sdk` (shared across all users). Add it to your `~/.zshrc`:

```bash
echo 'export GEODE_SDK="/Users/Shared/Geode/sdk"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Install Geode binaries into Geometry Dash

```bash
geode sdk install-binaries
```

### 5. Set the mods output path

```bash
echo 'export GEODE_MODS_PATH="$HOME/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app/Contents/geode/mods"' >> ~/.zshrc
source ~/.zshrc
```

## Running mods

See [mods/README.md](mods/README.md) for how to create, build, and run each mod.