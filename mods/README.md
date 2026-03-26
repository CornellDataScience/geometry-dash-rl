# mods/

Geode mods for Geometry Dash. Each subfolder is an independent Geode mod project with its own `CMakeLists.txt`, `mod.json`, and `src/`.

## Folders

| Folder | Purpose |
| --- | --- |
| `Telemetry/` | Player-only telemetry to shared memory (`/gdrl_ipc`) every frame |
| `TelemetryObstacles/` | Player telemetry + nearby obstacle scanning to shared memory |
| `ModTemplate/` | Clean starting point for new mods — copy this, update `mod.json`, write in `src/main.cpp` |

## Building Telemetry

```bash
cmake -S mods/Telemetry -B mods/Telemetry/build
cmake --build mods/Telemetry/build -j
cp mods/Telemetry/build/gdrl.telemetry.geode \
  "/Users/<your-username>/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app/Contents/geode/mods/"
```

## Building TelemetryObstacles

```bash
cmake -S mods/TelemetryObstacles -B mods/TelemetryObstacles/build
cmake --build mods/TelemetryObstacles/build -j
cp mods/TelemetryObstacles/build/gdrl.telemetry-obstacles.geode \
  "/Users/<your-username>/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app/Contents/geode/mods/"
```

Then launch Geometry Dash, open the Geode menu (bottom-right corner), go to **Mods**, and enable the mod. Restart GD if prompted. Only enable one of Telemetry or TelemetryObstacles at a time since they share the same shared memory segment.

## Creating a new mod

1. Copy `ModTemplate/` to a new folder
2. Update `mod.json`: set a unique `id` (e.g. `yourname.modname`), `name`, and `developer`
3. Write hook logic in `src/main.cpp`
4. Build and install:

```bash
cmake -S mods/YourMod -B mods/YourMod/build
cmake --build mods/YourMod/build -j
cp mods/YourMod/build/yourname.modname.geode \
  "/Users/<your-username>/Library/Application Support/Steam/steamapps/common/Geometry Dash/Geometry Dash.app/Contents/geode/mods/"
```

5. Launch Geometry Dash, open the Geode menu (bottom-right corner), go to **Mods**, and enable your mod. Restart GD if prompted.

## Running TelemetryObstacles Mod
# player state only (default)
python -m gdrl.env.live_monitor --print-every 25

# with obstacles
python -m gdrl.env.live_monitor --print-every 25 --show-objects

# show only first k nearest obstacles
python -m gdrl.env.live_monitor --print-every 25 --show-objects --num-objects k