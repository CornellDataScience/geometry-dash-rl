"""
Step 5 test: Object extraction viewer.

Prints a live table of nearby objects while you play manually.
Useful for verifying object categorization and discovering unknown IDs.

Usage:
    python object_viewer.py
"""

import time
from shm_reader import open_shm, read_state, wait_for_game, write_action

CATEGORIES = {
    0: 'deco', 1: 'hazard', 2: 'orb', 3: 'pad',
    4: 'mode_portal', 5: 'speed_portal', 6: 'solid'
}

print("Object viewer — shows nearby objects during gameplay")
print("Start playing a level in GD. Press Ctrl+C to stop.\n")

buf = open_shm()

try:
    while True:
        wait_for_game(buf)
        state = read_state(buf)

        # Clear screen and print header
        print("\033[2J\033[H", end='')
        print(
            f"Player: X={state['playerX']:.1f} Y={state['playerY']:.1f} "
            f"vY={state['velocityY']:.2f} dead={state['isDead']} "
            f"pct={state['levelPercent']:.1f}%"
        )
        print(f"Objects found: {state['numObjectsFound']}")
        print(f"{'#':>3} {'relX':>8} {'relY':>8} {'category':>14} {'hazard':>7} {'interact':>9} {'size':>8}")
        print('-' * 62)

        for i in range(state['numObjectsFound']):
            obj = state['objects'][i]
            relX, relY, cat, hazard, interact, size = obj
            cat_name = CATEGORIES.get(int(cat), f'?({int(cat)})')
            print(
                f"{i:3d} {relX:8.1f} {relY:8.1f} {cat_name:>14} "
                f"{'Y' if hazard > 0.5 else 'N':>7} "
                f"{'Y' if interact > 0.5 else 'N':>9} "
                f"{size:8.1f}"
            )

        # Respond immediately (no action) to keep game running at full speed
        write_action(buf, command=0, action=0)

except KeyboardInterrupt:
    print("\nDone.")
