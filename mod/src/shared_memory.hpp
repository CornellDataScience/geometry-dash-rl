#pragma once
#include <cstdint>
#include <cstring>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define SHM_NAME "/gd_rl_shm"

#pragma pack(push, 1)
struct RLSharedMemory {
    // Control (offset 0)
    std::atomic<int32_t> turn_flag;  // 0 = game's turn, 1 = agent's turn
    int32_t command;                  // 0 = step, 1 = reset, 2 = shutdown
    int32_t action;                   // 0 = no click, 1 = click

    // Player state (offset 12)
    float playerX;
    float playerY;
    float velocityY;
    float rotation;
    int32_t gameMode;        // 0=cube, 1=ship, 2=ball, 3=ufo
    int32_t isOnGround;      // 0 or 1
    int32_t isDead;           // 0 or 1
    float speedMultiplier;
    int32_t gravityFlipped;  // 0 or 1
    float levelPercent;      // 0-100

    // Nearby objects (offset 52): 25 slots x 6 floats = 600 bytes
    // Per slot: [relativeX, relativeY, typeCategory, isHazard, isInteractable, sizeClass]
    float objects[25][6];

    // Metadata (offset 652)
    int32_t numObjectsFound;
    float reward;
    int32_t done;            // 1 if episode ended
    int32_t attemptNumber;
    float prevPlayerX;       // internal bookkeeping for reward delta
};
#pragma pack(pop)
// Total size: 672 bytes

static_assert(sizeof(RLSharedMemory) == 672, "Struct size mismatch — Python offsets will be wrong");

inline RLSharedMemory* createSharedMemory() {
    // Clean up any stale mapping
    shm_unlink(SHM_NAME);

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) return nullptr;

    if (ftruncate(fd, sizeof(RLSharedMemory)) != 0) {
        close(fd);
        return nullptr;
    }

    void* ptr = mmap(nullptr, sizeof(RLSharedMemory),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) return nullptr;

    auto* shm = static_cast<RLSharedMemory*>(ptr);
    std::memset(shm, 0, sizeof(RLSharedMemory));
    return shm;
}

inline void destroySharedMemory(RLSharedMemory* shm) {
    if (shm) munmap(shm, sizeof(RLSharedMemory));
    shm_unlink(SHM_NAME);
}
