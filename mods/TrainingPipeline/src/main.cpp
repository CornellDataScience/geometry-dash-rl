#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace geode::prelude;

namespace {

constexpr int OBS_DIM = 608;
constexpr int MAX_NEARBY_OBJECTS = 100;
constexpr int FLOATS_PER_OBJECT = 6;  // relX, relY, objType, objID, scaleX, scaleY
constexpr int OBJ_OBS_START = 8;
constexpr float SCAN_BEHIND = 100.f;
constexpr float SCAN_AHEAD = 2000.f;
constexpr int RING_CAPACITY = 512;    // ~2 sec at 240fps

#pragma pack(push, 1)
struct FrameSlot {
    uint32_t tick;
    float    obs[OBS_DIM];
    uint8_t  player_input;   // 1 if human jumped this frame
    uint8_t  level_done;
    uint8_t  is_dead;
    uint8_t  pad0;
    uint32_t episode_id;     // increments on reset/death/level_complete
    uint8_t  pad1[4];
};

struct IPCBufferV3 {
    // --- header ---
    uint32_t version;        // =3
    uint32_t tick;           // latest tick (mirrors latest ring write)
    uint16_t obs_dim;        // 8 + N*6
    uint16_t ring_capacity;  // RING_CAPACITY (so Python can verify)
    uint32_t write_index;    // monotonic frame counter
    uint32_t frames_dropped; // reserved
    uint32_t episode_id;     // current episode

    // --- latest-frame mirror (for live monitor / inference / lockstep) ---
    float    obs[OBS_DIM];
    uint8_t  action_in;      // 0=idle, 1=jump (Python writes)
    uint8_t  ctrl_flags;     // bit0=reset_request
    uint8_t  player_input;   // 1 if human pressed jump this frame
    uint8_t  level_done;     // 1 if level complete
    uint8_t  reserved[8];

    // --- ring buffer ---
    FrameSlot frames[RING_CAPACITY];
};
#pragma pack(pop)

constexpr const char* SHM_NAME = "/gdrl_ipc_v3";
constexpr uint32_t IPC_VERSION = 3;

int g_fd = -1;
IPCBufferV3* g_ipc = nullptr;
float g_prevX = 0.f;
bool g_actionWasPressed = false;
uint32_t g_episodeId = 0;
bool g_wasDead = false;
bool g_wasLevelDone = false;

void ensure_ipc() {
    if (g_ipc) return;

    g_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_fd < 0) {
        log::error("GDRL-TP shm_open failed");
        return;
    }

    if (ftruncate(g_fd, sizeof(IPCBufferV3)) != 0) {
        log::error("GDRL-TP ftruncate failed");
        close(g_fd);
        g_fd = -1;
        return;
    }

    void* ptr = mmap(nullptr, sizeof(IPCBufferV3), PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, 0);
    if (ptr == MAP_FAILED) {
        log::error("GDRL-TP mmap failed");
        close(g_fd);
        g_fd = -1;
        return;
    }

    g_ipc = reinterpret_cast<IPCBufferV3*>(ptr);
    std::memset(g_ipc, 0, sizeof(IPCBufferV3));
    g_ipc->version = IPC_VERSION;
    g_ipc->tick = 0;
    g_ipc->obs_dim = OBS_DIM;
    g_ipc->ring_capacity = RING_CAPACITY;
    g_ipc->write_index = 0;
    g_ipc->episode_id = 0;
    log::info("GDRL-TP shm initialized at {} size={} ring_cap={}",
        SHM_NAME, (int)sizeof(IPCBufferV3), RING_CAPACITY);
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
    if (p->m_isBird) return 3;
    return 0;
}

} // namespace

$on_mod(Loaded) {
    ensure_ipc();
    log::info("GDRL-TP mod loaded");
}

class $modify(GDRLTPPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        bool ok = PlayLayer::init(level, useReplay, dontCreateObjects);
        ensure_ipc();
        if (g_ipc) {
            g_ipc->tick = 0;
            g_ipc->level_done = 0;
            g_ipc->ctrl_flags = 0;
            g_ipc->player_input = 0;
            g_ipc->action_in = 0;
            g_episodeId += 1;
            g_ipc->episode_id = g_episodeId;
            g_actionWasPressed = false;
            g_wasDead = false;
            g_wasLevelDone = false;
        }
        log::info("GDRL-TP PlayLayer::init ok={} episode={}", ok, g_episodeId);
        return ok;
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        ensure_ipc();
        if (!g_ipc) return;

        g_ipc->tick += 1;

        if (!m_player1) return;
        auto* p = m_player1;

        float xNow = p->getPositionX();
        g_ipc->obs[0] = xNow;
        g_ipc->obs[1] = p->getPositionY();
        g_ipc->obs[2] = static_cast<float>(p->m_yVelocity);
        g_ipc->obs[3] = xNow - g_prevX;
        g_prevX = xNow;
        g_ipc->obs[4] = p->m_isOnGround ? 1.0f : 0.0f;
        g_ipc->obs[5] = p->m_isDead ? 1.0f : 0.0f;
        g_ipc->obs[6] = 1.0f;
        g_ipc->obs[7] = static_cast<float>(mode_id_from_player(p));

        scan_nearby_objects(this, xNow, p->getPositionY(), g_ipc->obs);

        int objCount = 0;
        for (int i = 0; i < MAX_NEARBY_OBJECTS; i++) {
            if (g_ipc->obs[OBJ_OBS_START + i * FLOATS_PER_OBJECT] != 0.f ||
                g_ipc->obs[OBJ_OBS_START + i * FLOATS_PER_OBJECT + 1] != 0.f)
                objCount++;
            else break;
        }
        g_ipc->obs_dim = static_cast<uint16_t>(OBJ_OBS_START + objCount * FLOATS_PER_OBJECT);

        // sample whether jump is currently HELD (works for all modes: tap-to-jump
        // in cube, hold-to-thrust in ship/ufo, etc.)
        bool jumpHeld = false;
        auto it = p->m_holdingButtons.find(static_cast<int>(PlayerButton::Jump));
        if (it != p->m_holdingButtons.end()) jumpHeld = it->second;
        g_ipc->player_input = jumpHeld ? 1 : 0;

        // detect death/level_done transitions for episode boundaries
        bool isDead = p->m_isDead;
        bool levelDone = g_ipc->level_done != 0;
        bool episodeEnded = (isDead && !g_wasDead) || (levelDone && !g_wasLevelDone);

        // --- write a ring slot for this frame ---
        uint32_t idx = g_ipc->write_index;
        FrameSlot& slot = g_ipc->frames[idx % RING_CAPACITY];
        slot.tick = g_ipc->tick;
        std::memcpy(slot.obs, g_ipc->obs, sizeof(slot.obs));
        slot.player_input = g_ipc->player_input;
        slot.level_done = g_ipc->level_done;
        slot.is_dead = isDead ? 1 : 0;
        slot.episode_id = g_ipc->episode_id;
        // publish: bump write_index AFTER slot is fully written.
        // On ARM64 a naturally-aligned u32 store is atomic. We use a
        // compiler barrier to prevent reordering the slot writes past
        // the index update.
        std::atomic_thread_fence(std::memory_order_release);
        g_ipc->write_index = idx + 1;

        if (episodeEnded) {
            g_episodeId += 1;
            g_ipc->episode_id = g_episodeId;
        }
        g_wasDead = isDead;
        g_wasLevelDone = levelDone;

        // action injection
        bool wantPress = g_ipc->action_in != 0;
        if (wantPress && !g_actionWasPressed) {
            p->pushButton(PlayerButton::Jump);
            g_actionWasPressed = true;
        } else if (!wantPress && g_actionWasPressed) {
            p->releaseButton(PlayerButton::Jump);
            g_actionWasPressed = false;
        }
        g_ipc->action_in = 0;

        // reset handling
        if (g_ipc->ctrl_flags & 0x01) {
            g_ipc->ctrl_flags &= ~0x01;
            geode::Loader::get()->queueInMainThread([]() {
                auto* pl = PlayLayer::get();
                if (pl) pl->resetLevel();
            });
        }

        static int dbg = 0;
        if (++dbg % 600 == 0) {
            log::info("GDRL-TP alive tick={} write_idx={} ep={} objs={}",
                g_ipc->tick, g_ipc->write_index, g_episodeId, objCount);
        }
    }

    void levelComplete() {
        ensure_ipc();
        if (g_ipc) {
            g_ipc->level_done = 1;
            log::info("GDRL-TP levelComplete tick={}", g_ipc->tick);
        }
        PlayLayer::levelComplete();
    }
};

