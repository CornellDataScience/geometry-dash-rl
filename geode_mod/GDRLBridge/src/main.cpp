#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>

using namespace geode::prelude;

namespace {
#pragma pack(push, 1)
struct IPCBufferV0 {
    uint32_t version;
    uint32_t tick;
    float obs[108];
    uint8_t action_in;
    uint8_t reserved[3];
};
#pragma pack(pop)

static_assert(sizeof(IPCBufferV0) == 444, "IPCBufferV0 size mismatch");

constexpr const char* SHM_NAME = "/gdrl_ipc"; // POSIX name
constexpr uint32_t IPC_VERSION = 1;

int g_fd = -1;
IPCBufferV0* g_ipc = nullptr;
float g_prevX = 0.f;

void ensure_ipc() {
    if (g_ipc) return;

    g_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_fd < 0) {
        log::error("GDRL shm_open failed");
        return;
    }

    if (ftruncate(g_fd, sizeof(IPCBufferV0)) != 0) {
        log::error("GDRL ftruncate failed");
        close(g_fd);
        g_fd = -1;
        return;
    }

    void* ptr = mmap(nullptr, sizeof(IPCBufferV0), PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, 0);
    if (ptr == MAP_FAILED) {
        log::error("GDRL mmap failed");
        close(g_fd);
        g_fd = -1;
        return;
    }

    g_ipc = reinterpret_cast<IPCBufferV0*>(ptr);
    std::memset(g_ipc, 0, sizeof(IPCBufferV0));
    g_ipc->version = IPC_VERSION;
    g_ipc->tick = 0;
    log::info("GDRL shm initialized at {}", SHM_NAME);
}

int mode_id_from_player(PlayerObject* p) {
    if (!p) return 0;
    if (p->m_isShip) return 1;
    if (p->m_isBall) return 2;
    if (p->m_isBird) return 3; // UFO
    return 0; // cube fallback
}
} // namespace

$on_mod(Loaded) {
    ensure_ipc();
    log::info("GDRL mod loaded + ensure_ipc called");
}

class $modify(GDRLPlayLayer, PlayLayer) {
    void update(float dt) {
        PlayLayer::update(dt);
        ensure_ipc();
        if (!g_ipc || !m_player1) return;

        auto* p = m_player1;

        float xNow = p->getPositionX();
        g_ipc->obs[0] = xNow;
        g_ipc->obs[1] = p->getPositionY();
        g_ipc->obs[2] = static_cast<float>(p->m_yVelocity);
        g_ipc->obs[3] = xNow - g_prevX; // approx x velocity per frame
        g_prevX = xNow;
        g_ipc->obs[4] = p->m_isOnGround ? 1.0f : 0.0f;
        g_ipc->obs[5] = p->m_isDead ? 1.0f : 0.0f;
        g_ipc->obs[6] = 1.0f; // TODO: wire real speed multiplier
        g_ipc->obs[7] = static_cast<float>(mode_id_from_player(p));

        g_ipc->tick += 1;

        static int dbg = 0;
        if (++dbg % 300 == 0) {
            log::info("GDRL update alive tick={} x={} y={} mode={}", g_ipc->tick, g_ipc->obs[0], g_ipc->obs[1], static_cast<int>(g_ipc->obs[7]));
        }

        // action_in read is present for next step (actual input injection pending)
        [[maybe_unused]] bool wantPress = g_ipc->action_in != 0;
    }
};
