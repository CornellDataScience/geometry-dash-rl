# geometry-dash-rl

Teacher-student RL pipeline for Geometry Dash using a Geode shared-memory bridge.

## Requirements

- Python `>=3.10`
- Python deps from `requirements.txt` (`torch`, `stable-baselines3`, `gymnasium`, etc.)
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
/Users/<your-username>/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app
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
echo 'export GEODE_MODS_PATH="/Users/<your-username>/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app/Contents/geode/mods"' >> ~/.zshrc
source ~/.zshrc
```

## Running with live Geode telemetry

1. Build and install the Telemetry mod — see [mods/README.md](mods/README.md)

2. Launch Geometry Dash and enter a level

3. Monitor live telemetry:

```bash
python -m gdrl.env.live_monitor --print-every 25
```

4. If a stale segment remains after a crash:

```bash
python -m gdrl.env.geode_shm_cleanup --shm-name gdrl_ipc
```

## Troubleshooting

- **Timeout on `geode_wait`**: make sure a level is running and the mod is enabled in Geode
- **Version mismatch**: compare `EXPECTED_VERSION` in `src/gdrl/env/geode_ipc.py` with `IPC_VERSION` in `mods/GDRLBridge/src/main.cpp`
- **Stale segment**: run `python -m gdrl.env.geode_shm_cleanup --shm-name gdrl_ipc`
