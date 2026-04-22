from gdrl.env.geode_ipc import GeodeSharedMemoryAdapter
from gdrl.env.privileged_env import GDPrivilegedEnv

env = GDPrivilegedEnv(GeodeSharedMemoryAdapter())
obs, _ = env.reset()
print(f"reset: x={obs[0]:.1f} y={obs[1]:.1f} dead={int(obs[5])}")

total_r = 0.0
steps = 0
last_x = obs[0]
for i in range(20000):
    # alternate jump every 30 frames so we can see action propagation
    action = 1 if (i // 30) % 2 == 0 else 0
    obs, r, term, trunc, _ = env.step(action)
    total_r += r
    steps += 1
    if i % 30 == 0 or term or trunc:
        dx = obs[0] - last_x
        last_x = obs[0]
        print(f"step={i:5d} act={action} x={obs[0]:7.1f} dx={dx:+6.1f} r={r:+6.2f} dead={int(obs[5])}")
    if term or trunc:
        print(f"DONE term={term} trunc={trunc} final_r={r:+.2f}")
        break

print(f"steps={steps} total_reward={total_r:.2f}")
