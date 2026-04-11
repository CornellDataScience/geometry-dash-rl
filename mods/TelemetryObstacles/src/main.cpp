#include <Geode/Geode.hpp>
#include <Geode/binding/CheckpointObject.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace geode::prelude;

namespace {
#pragma pack(push, 1)
struct IPCBufferV3 {
    uint32_t version;      // =3
    uint32_t tick;
    uint16_t obs_dim;      // 8 + N*6, written by mod each frame
    float obs[608];        // 8 player + 100*6 obstacles
    uint8_t action_in;     // 0=idle, 1=jump (written by Python)
    uint8_t ctrl_flags;    // bit0=full_reset, bit1=checkpoint_reset (written by Python)
    uint8_t player_input;  // 1 if human pressed jump this frame (written by mod)
    uint8_t level_done;    // 1 if level complete (written by mod)
    float reset_target_x;  // requested checkpoint frontier x (written by Python)
};
#pragma pack(pop)

static_assert(sizeof(IPCBufferV3) == 4+4+2 + 608*4 + 1+1+1+1+4, "IPCBufferV3 size mismatch");

constexpr const char* SHM_NAME = "/gdrl_ipc";
constexpr uint32_t IPC_VERSION = 3;
constexpr uint8_t CTRL_RESET_FULL = 0x01;
constexpr uint8_t CTRL_RESET_CHECKPOINT = 0x02;
constexpr float CHECKPOINT_CAPTURE_SPACING = 75.f;
constexpr float MIN_CHECKPOINT_CAPTURE_X = 75.f;
constexpr size_t MAX_CACHED_CHECKPOINTS = 48;

int g_fd = -1;
IPCBufferV3* g_ipc = nullptr;
float g_prevX = 0.f;
bool g_humanJumpedThisFrame = false;  // set by pushButton hook, read+cleared in postUpdate
bool g_actionWasPressed = false;      // edge detection for action injection
float g_nextCheckpointCaptureX = MIN_CHECKPOINT_CAPTURE_X;

struct CachedCheckpoint {
    float x;
    CheckpointObject* checkpoint;
};

std::vector<CachedCheckpoint> g_cachedCheckpoints;

void ensure_ipc() {
    if (g_ipc) return;

    // The mod owns the shared-memory lifetime. Recreate it on load so
    // stale segments from prior runs can't wedge ftruncate / mmap.
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

constexpr int MAX_NEARBY_OBJECTS = 100;
constexpr int FLOATS_PER_OBJECT = 6;  // relX, relY, objType, objID, scaleX, scaleY
constexpr int OBJ_OBS_START = 8;      // obs[8..607]
constexpr float SCAN_BEHIND = 100.f;
constexpr float SCAN_AHEAD = 2000.f;

void clear_cached_checkpoints() {
    for (auto& slot : g_cachedCheckpoints) {
        if (slot.checkpoint) {
            slot.checkpoint->release();
            slot.checkpoint = nullptr;
        }
    }
    g_cachedCheckpoints.clear();
    g_nextCheckpointCaptureX = MIN_CHECKPOINT_CAPTURE_X;
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

void store_checkpoint(PlayLayer* layer, float playerX) {
    if (!layer || !layer->m_player1 || layer->m_player1->m_isDead) return;

    auto* checkpoint = layer->createCheckpoint();
    if (!checkpoint) {
        static int createFailCount = 0;
        if (++createFailCount <= 5 || createFailCount % 100 == 0) {
            log::warn("GDRL createCheckpoint returned null at x={}", playerX);
        }
        return;
    }

    if (!g_cachedCheckpoints.empty() && std::abs(g_cachedCheckpoints.back().x - playerX) < 5.f) {
        return;
    }

    checkpoint->retain();
    g_cachedCheckpoints.push_back({playerX, checkpoint});
    while (g_cachedCheckpoints.size() > MAX_CACHED_CHECKPOINTS) {
        auto& slot = g_cachedCheckpoints.front();
        if (slot.checkpoint) {
            slot.checkpoint->release();
        }
        g_cachedCheckpoints.erase(g_cachedCheckpoints.begin());
    }

    log::info(
        "GDRL stored checkpoint count={} x={}",
        static_cast<int>(g_cachedCheckpoints.size()),
        playerX
    );
}

void maybe_store_checkpoint(PlayLayer* layer, float playerX) {
    if (playerX < MIN_CHECKPOINT_CAPTURE_X) return;
    if (playerX + 1e-3f < g_nextCheckpointCaptureX) return;

    store_checkpoint(layer, playerX);

    float nextBoundary = (std::floor(playerX / CHECKPOINT_CAPTURE_SPACING) + 1.f) * CHECKPOINT_CAPTURE_SPACING;
    if (nextBoundary <= playerX + 0.5f) {
        nextBoundary = playerX + CHECKPOINT_CAPTURE_SPACING;
    }
    g_nextCheckpointCaptureX = std::max(MIN_CHECKPOINT_CAPTURE_X, nextBoundary);
}

CachedCheckpoint* best_checkpoint_for_target(float targetX) {
    CachedCheckpoint* best = nullptr;
    for (auto& slot : g_cachedCheckpoints) {
        if (slot.x > targetX + 1.f) continue;
        if (!best || slot.x > best->x) {
            best = &slot;
        }
    }
    return best;
}

void queue_reset_request(uint8_t ctrlFlags, float targetX) {
    bool wantsCheckpoint = (ctrlFlags & CTRL_RESET_CHECKPOINT) != 0;
    geode::Loader::get()->queueInMainThread([wantsCheckpoint, targetX]() {
        auto* pl = PlayLayer::get();
        if (!pl) return;

        release_injected_jump(pl->m_player1);
        if (g_ipc) {
            g_ipc->level_done = 0;
            g_ipc->action_in = 0;
            g_ipc->player_input = 0;
        }

        if (wantsCheckpoint) {
            if (auto* slot = best_checkpoint_for_target(targetX)) {
                pl->loadFromCheckpoint(slot->checkpoint);
                g_prevX = slot->x;
                log::info(
                    "GDRL loaded cached checkpoint targetX={} actualX={} cacheCount={}",
                    targetX,
                    slot->x,
                    static_cast<int>(g_cachedCheckpoints.size())
                );
                return;
            }

            log::warn(
                "GDRL no cached checkpoint at/before targetX={} cacheCount={} - falling back to resetLevel",
                targetX,
                static_cast<int>(g_cachedCheckpoints.size())
            );
        }

        pl->resetLevel();
        g_prevX = 0.f;
    });
}

bool is_gameplay_object(GameObjectType type) {
    switch (type) {
        case GameObjectType::Decoration:
        case GameObjectType::Collectible:
        case GameObjectType::UserCoin:
        case GameObjectType::SecretCoin:
        case GameObjectType::EnterEffectObject:
        case GameObjectType::Modifier:
        case GameObjectType::Special:
        case GameObjectType::Breakable:
        case GameObjectType::CollisionObject:
            return false;
        default:
            return true;
    }
}

struct NearbyObj {
    float relX;
    float relY;
    float objType;
    float objID;
    float scaleX;
    float scaleY;
    float absDist;
};

void scan_nearby_objects(GJBaseGameLayer* layer, float playerX, float playerY, float* obs) {
    std::memset(&obs[OBJ_OBS_START], 0, MAX_NEARBY_OBJECTS * FLOATS_PER_OBJECT * sizeof(float));

    auto* objects = layer->m_objects;
    if (!objects) return;

    std::vector<NearbyObj> found;
    found.reserve(64);

    float minX = playerX - SCAN_BEHIND;
    float maxX = playerX + SCAN_AHEAD;

    for (int i = 0; i < objects->count(); i++) {
        auto* obj = static_cast<GameObject*>(objects->objectAtIndex(i));
        if (!obj) continue;

        float ox = obj->getPositionX();
        if (ox < minX || ox > maxX) continue;

        auto type = obj->m_objectType;
        if (!is_gameplay_object(type)) continue;

        float relX = ox - playerX;
        float relY = obj->getPositionY() - playerY;
        found.push_back({
            relX,
            relY,
            static_cast<float>(type),
            static_cast<float>(obj->m_objectID),
            obj->getScaleX(),
            obj->getScaleY(),
            std::abs(relX)
        });
    }

    std::sort(found.begin(), found.end(), [](const NearbyObj& a, const NearbyObj& b) {
        return a.absDist < b.absDist;
    });

    int count = std::min((int)found.size(), MAX_NEARBY_OBJECTS);
    for (int i = 0; i < count; i++) {
        int base = OBJ_OBS_START + i * FLOATS_PER_OBJECT;
        obs[base + 0] = found[i].relX;
        obs[base + 1] = found[i].relY;
        obs[base + 2] = found[i].objType;
        obs[base + 3] = found[i].objID;
        obs[base + 4] = found[i].scaleX;
        obs[base + 5] = found[i].scaleY;
    }
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
        clear_cached_checkpoints();
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

        // scan nearby gameplay objects into obs[8..607]
        scan_nearby_objects(this, xNow, p->getPositionY(), g_ipc->obs);
        maybe_store_checkpoint(this, xNow);

        // write obs_dim (player floats + actual obstacle count * floats per obj)
        int objCount = 0;
        for (int i = 0; i < MAX_NEARBY_OBJECTS; i++) {
            if (g_ipc->obs[OBJ_OBS_START + i * FLOATS_PER_OBJECT] != 0.f ||
                g_ipc->obs[OBJ_OBS_START + i * FLOATS_PER_OBJECT + 1] != 0.f)
                objCount++;
            else break; // sorted by distance, first zero means no more
        }
        g_ipc->obs_dim = static_cast<uint16_t>(OBJ_OBS_START + objCount * FLOATS_PER_OBJECT);

        // human input recording: write whether human pressed jump, then clear flag
        g_ipc->player_input = g_humanJumpedThisFrame ? 1 : 0;
        g_humanJumpedThisFrame = false;

        // action injection from Python agent
        bool wantPress = g_ipc->action_in != 0;
        if (wantPress && !g_actionWasPressed) {
            p->pushButton(PlayerButton::Jump);
            g_actionWasPressed = true;
        } else if (!wantPress && g_actionWasPressed) {
            p->releaseButton(PlayerButton::Jump);
            g_actionWasPressed = false;
        }
        g_ipc->action_in = 0; // acknowledge

        // level reset handling (deferred to next safe point)
        uint8_t resetFlags = g_ipc->ctrl_flags & (CTRL_RESET_FULL | CTRL_RESET_CHECKPOINT);
        if (resetFlags != 0) {
            float targetX = g_ipc->reset_target_x;
            g_ipc->ctrl_flags &= ~(CTRL_RESET_FULL | CTRL_RESET_CHECKPOINT);
            g_ipc->reset_target_x = 0.f;
            queue_reset_request(resetFlags, targetX);
        }

        static int dbg = 0;
        if (++dbg % 300 == 0) {
            log::info("GDRL update alive tick={} x={} y={} mode={} nearbyObjs={}",
                g_ipc->tick, g_ipc->obs[0], g_ipc->obs[1],
                static_cast<int>(g_ipc->obs[7]), objCount);
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
