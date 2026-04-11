#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

using namespace geode::prelude;

namespace {
#pragma pack(push, 1)
struct IPCBufferV3 {
    uint32_t version;      // =3
    uint32_t tick;
    uint16_t obs_dim;      // 8 (player floats only)
    float obs[608];
    uint8_t action_in;     // 0=idle, 1=jump (written by Python)
    uint8_t ctrl_flags;    // bit0=full_reset, bit1=checkpoint_reset (written by Python)
    uint8_t player_input;  // 1 if human pressed jump this frame
    uint8_t level_done;    // 1 if level complete
    float reset_target_x;  // ignored in Telemetry, accepted for protocol compatibility
};
#pragma pack(pop)

static_assert(sizeof(IPCBufferV3) == 4+4+2 + 608*4 + 1+1+1+1+4, "IPCBufferV3 size mismatch");

constexpr const char* SHM_NAME = "/gdrl_ipc";
constexpr uint32_t IPC_VERSION = 3;
constexpr uint8_t CTRL_RESET_FULL = 0x01;
constexpr uint8_t CTRL_RESET_CHECKPOINT = 0x02;

int g_fd = -1;
IPCBufferV3* g_ipc = nullptr;
float g_prevX = 0.f;
bool g_humanJumpedThisFrame = false;
bool g_actionWasPressed = false;

void ensure_ipc() {
    if (g_ipc) return;

    shm_unlink(SHM_NAME);

    g_fd = shm_open(SHM_NAME, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (g_fd < 0) {
        log::error("GDRL shm_open failed errno={} msg={}", errno, std::strerror(errno));
        return;
    }

    if (ftruncate(g_fd, sizeof(IPCBufferV3)) != 0) {
        log::error("GDRL ftruncate failed errno={} msg={}", errno, std::strerror(errno));
        close(g_fd);
        g_fd = -1;
        return;
    }

    void* ptr = mmap(nullptr, sizeof(IPCBufferV3), PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, 0);
    if (ptr == MAP_FAILED) {
        log::error("GDRL mmap failed errno={} msg={}", errno, std::strerror(errno));
        close(g_fd);
        g_fd = -1;
        return;
    }

    g_ipc = reinterpret_cast<IPCBufferV3*>(ptr);
    std::memset(g_ipc, 0, sizeof(IPCBufferV3));
    g_ipc->version = IPC_VERSION;
    g_ipc->tick = 0;
    log::info("GDRL shm initialized at {}", SHM_NAME);
}

void clear_runtime_flags() {
    g_actionWasPressed = false;
    g_humanJumpedThisFrame = false;
    g_prevX = 0.f;
    if (g_ipc) {
        g_ipc->level_done = 0;
        g_ipc->ctrl_flags = 0;
        g_ipc->player_input = 0;
        g_ipc->action_in = 0;
        g_ipc->reset_target_x = 0.f;
    }
}

void release_injected_jump(PlayerObject* player) {
    if (player && g_actionWasPressed) {
        player->releaseButton(PlayerButton::Jump);
    }
    g_actionWasPressed = false;
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
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        bool ok = PlayLayer::init(level, useReplay, dontCreateObjects);
        ensure_ipc();
        clear_runtime_flags();
        if (g_ipc) {
            g_ipc->tick = 0;
        }
        log::info(
            "GDRL PlayLayer::init called ok={} level_ptr={} replay={} no_objs={}",
            ok,
            static_cast<void*>(level),
            useReplay,
            dontCreateObjects
        );
        return ok;
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        ensure_ipc();
        if (!g_ipc) return;

        // Increment tick on every hooked PlayLayer update so Python can verify
        // that the frame hook is alive even if player pointer access fails.
        g_ipc->tick += 1;

        static int postDbg = 0;
        postDbg += 1;
        if (postDbg <= 5 || postDbg % 300 == 0) {
            log::info(
                "GDRL postUpdate hook tick={} has_player={}",
                g_ipc->tick,
                m_player1 != nullptr
            );
        }

        static int noPlayerDbg = 0;
        if (!m_player1) {
            if (++noPlayerDbg <= 5 || noPlayerDbg % 300 == 0) {
                log::warn("GDRL PlayLayer update active but m_player1 is null; tick={}", g_ipc->tick);
            }
            return;
        }

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
        g_ipc->obs_dim = 8;
        g_ipc->player_input = g_humanJumpedThisFrame ? 1 : 0;
        g_humanJumpedThisFrame = false;

        bool wantPress = g_ipc->action_in != 0;
        if (wantPress && !g_actionWasPressed) {
            p->pushButton(PlayerButton::Jump);
            g_actionWasPressed = true;
        } else if (!wantPress && g_actionWasPressed) {
            p->releaseButton(PlayerButton::Jump);
            g_actionWasPressed = false;
        }
        g_ipc->action_in = 0;

        uint8_t resetFlags = g_ipc->ctrl_flags & (CTRL_RESET_FULL | CTRL_RESET_CHECKPOINT);
        if (resetFlags != 0) {
            g_ipc->ctrl_flags &= ~(CTRL_RESET_FULL | CTRL_RESET_CHECKPOINT);
            g_ipc->reset_target_x = 0.f;
            geode::Loader::get()->queueInMainThread([]() {
                auto* pl = PlayLayer::get();
                if (!pl) return;
                release_injected_jump(pl->m_player1);
                if (g_ipc) {
                    g_ipc->level_done = 0;
                    g_ipc->action_in = 0;
                    g_ipc->player_input = 0;
                }
                pl->resetLevel();
                g_prevX = 0.f;
            });
        }

        static int dbg = 0;
        if (++dbg % 300 == 0) {
            log::info("GDRL update alive tick={} x={} y={} mode={}", g_ipc->tick, g_ipc->obs[0], g_ipc->obs[1], static_cast<int>(g_ipc->obs[7]));
        }

    }

    void levelComplete() {
        ensure_ipc();
        if (g_ipc) {
            g_ipc->level_done = 1;
            log::info("GDRL levelComplete detected at tick={}", g_ipc->tick);
        }
        PlayLayer::levelComplete();
    }
};

class $modify(GDRLPlayerObject, PlayerObject) {
    bool pushButton(PlayerButton button) {
        bool result = PlayerObject::pushButton(button);
        if (button == PlayerButton::Jump) {
            g_humanJumpedThisFrame = true;
        }
        return result;
    }
};
