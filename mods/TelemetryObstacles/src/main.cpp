#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

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

constexpr int MAX_NEARBY_OBJECTS = 20;
constexpr int FLOATS_PER_OBJECT = 5;
constexpr int OBJ_OBS_START = 8; // obs[8..107]
constexpr float SCAN_BEHIND = 100.f;
constexpr float SCAN_AHEAD = 800.f;

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
        if (g_ipc) {
            g_ipc->tick = 0;
            g_ipc->reserved[0] = 0; // clear level-complete flag for new run
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

        // scan nearby gameplay objects into obs[8..107]
        scan_nearby_objects(this, xNow, p->getPositionY(), g_ipc->obs);

        static int dbg = 0;
        if (++dbg % 300 == 0) {
            int objCount = 0;
            for (int i = 0; i < MAX_NEARBY_OBJECTS; i++) {
                if (g_ipc->obs[OBJ_OBS_START + i * FLOATS_PER_OBJECT] != 0.f ||
                    g_ipc->obs[OBJ_OBS_START + i * FLOATS_PER_OBJECT + 1] != 0.f)
                    objCount++;
            }
            log::info("GDRL update alive tick={} x={} y={} mode={} nearbyObjs={}",
                g_ipc->tick, g_ipc->obs[0], g_ipc->obs[1],
                static_cast<int>(g_ipc->obs[7]), objCount);
        }

        [[maybe_unused]] bool wantPress = g_ipc->action_in != 0;
    }

    void levelComplete() {
        ensure_ipc();
        if (g_ipc) {
            g_ipc->reserved[0] = 1; // level-complete flag
            log::info("GDRL levelComplete detected at tick={}", g_ipc->tick);
        }
        PlayLayer::levelComplete();
    }
};
