from gdrl.env.geode_ipc import GeodeSharedMemoryAdapter
ipc = GeodeSharedMemoryAdapter(); ipc.verify_version()
print("press space in game...")
for _ in range(2000):
    if ipc.wait_next_tick(timeout_s=1.0) and ipc.read_player_input():
        print(f"JUMP detected tick={ipc.read_tick()}")
ipc.close()