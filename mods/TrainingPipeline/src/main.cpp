#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/AppDelegate.hpp>

#include <OpenGL/gl.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>

using namespace geode::prelude;

namespace {

// --- Observation layout (state) ---
constexpr int   OBS_DIM = 608;
constexpr int   MAX_NEARBY_OBJECTS = 100;
constexpr int   FLOATS_PER_OBJECT = 6;  // relX, relY, objType, objID, scaleX, scaleY
constexpr int   OBJ_OBS_START = 8;
constexpr float SCAN_BEHIND = 100.f;
constexpr float SCAN_AHEAD = 2000.f;

// --- Ring buffer (state-only) ---
constexpr int   RING_CAPACITY = 512;    // ~2 sec at 240fps

// --- Frame capture ---
constexpr int   FRAME_W = 128;
constexpr int   FRAME_H = 128;
constexpr int   FRAME_CHANNELS = 1;     // grayscale
constexpr int   FRAME_BYTES = FRAME_W * FRAME_H * FRAME_CHANNELS;
constexpr int   FRAME_CAPTURE_EVERY = 4; // every 4th tick (60fps frames at 240fps physics)

// --- Mid-level spawn checkpoints (online capture) ---
constexpr int   CHECKPOINT_BUCKET_PCT = 5;       // bucket size in percent
constexpr int   CHECKPOINT_BUCKETS_PER_LEVEL = 100 / CHECKPOINT_BUCKET_PCT; // 20

#pragma pack(push, 1)
struct FrameSlot {
    uint32_t tick;
    float    obs[OBS_DIM];
    uint8_t  player_input;   // 1 if jump held
    uint8_t  level_done;
    uint8_t  is_dead;
    uint8_t  pad0;
    uint32_t episode_id;     // increments on reset/death/level_complete
    uint8_t  pad1[4];
};

struct IPCBufferV4 {
    // --- header (40 bytes) ---
    uint32_t version;            // =4
    uint32_t tick;               // latest tick
    uint16_t obs_dim;            // 8 + N*6
    uint16_t ring_capacity;      // RING_CAPACITY
    uint32_t write_index;        // monotonic frame counter
    uint32_t frames_dropped;     // reserved
    uint32_t episode_id;         // current episode id
    uint16_t frame_w;            // FRAME_W
    uint16_t frame_h;            // FRAME_H
    uint16_t frame_channels;     // FRAME_CHANNELS
    uint16_t reserved_hdr;       // padding
    float    level_length;       // in world units (mod's m_levelLength); 0 if unknown
    uint32_t current_level_id;   // GJGameLevel id of currently loaded level

    // --- latest-frame mirror (state) ---
    float    obs[OBS_DIM];       // 2432 bytes
    uint8_t  action_in;          // 0=idle, 1=jump (Python writes)
    uint8_t  ctrl_flags;         // bit0=reset_request, bit1=load_level_request
    uint8_t  player_input;       // jump held this frame
    uint8_t  level_done;         // level complete flag
    uint32_t requested_level_id; // Python writes for ctrl bit1
    uint8_t  reset_target_pct;   // Python writes alongside ctrl bit0; 0=full reset
    uint8_t  reserved_mirror[15];// padding so mirror_frame_seq is 4-byte aligned

    // --- frame mirror (latest captured frame; updated every FRAME_CAPTURE_EVERY ticks) ---
    uint32_t mirror_frame_seq;   // bumps after pixels published; lock-free read pattern
    uint8_t  mirror_frame[FRAME_BYTES];  // FRAME_W*FRAME_H grayscale, upright

    // --- ring buffer (state-only; pixels intentionally omitted to save memory) ---
    FrameSlot frames[RING_CAPACITY];
};
#pragma pack(pop)

constexpr const char* SHM_NAME = "/gdrl_ipc_v4";
constexpr uint32_t IPC_VERSION = 4;

int g_fd = -1;
IPCBufferV4* g_ipc = nullptr;

// per-frame state
float    g_prevX = 0.f;
bool     g_actionWasPressed = false;
uint32_t g_episodeId = 0;
bool     g_wasDead = false;
bool     g_wasLevelDone = false;
int      g_lastBucketCaptured = -1;     // last 5%-bucket we created a checkpoint for in this run
int      g_lastBucketDeath = -1;        // bucket reached on the most recent death (for adaptive logging)
int      g_frameTick = 0;               // monotonic tick for FRAME_CAPTURE_EVERY gating
std::vector<uint8_t> g_glScratch;       // reusable buffer for glReadPixels

// online checkpoint store: key = (levelID << 8) | bucket, value = retained CheckpointObject*
std::unordered_map<uint64_t, CheckpointObject*> g_checkpoints;

uint64_t cp_key(int level_id, int bucket) {
    return (static_cast<uint64_t>(level_id) << 8) | static_cast<uint64_t>(bucket & 0xff);
}

void ensure_ipc() {
    if (g_ipc) return;

    g_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_fd < 0) {
        log::error("GDRL-TP shm_open failed");
        return;
    }
    if (ftruncate(g_fd, sizeof(IPCBufferV4)) != 0) {
        log::error("GDRL-TP ftruncate failed");
        close(g_fd);
        g_fd = -1;
        return;
    }
    void* ptr = mmap(nullptr, sizeof(IPCBufferV4), PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, 0);
    if (ptr == MAP_FAILED) {
        log::error("GDRL-TP mmap failed");
        close(g_fd);
        g_fd = -1;
        return;
    }

    g_ipc = reinterpret_cast<IPCBufferV4*>(ptr);
    std::memset(g_ipc, 0, sizeof(IPCBufferV4));
    g_ipc->version = IPC_VERSION;
    g_ipc->obs_dim = OBS_DIM;
    g_ipc->ring_capacity = RING_CAPACITY;
    g_ipc->frame_w = FRAME_W;
    g_ipc->frame_h = FRAME_H;
    g_ipc->frame_channels = FRAME_CHANNELS;
    log::info("GDRL-TP shm initialized at {} size={} ring_cap={} frame={}x{}x{}",
        SHM_NAME, (int)sizeof(IPCBufferV4), RING_CAPACITY, FRAME_W, FRAME_H, FRAME_CHANNELS);
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
            relX, relY,
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

// Capture the current GL framebuffer, downsample (nearest stride) +
// luma-convert to 128x128 grayscale into the IPC mirror_frame slot.
// glReadPixels causes a GPU pipeline stall (~1-3ms on M-series), so this
// is gated to every FRAME_CAPTURE_EVERY ticks by the caller.
void capture_frame_into_mirror() {
    if (!g_ipc) return;

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    int W = vp[2], H = vp[3];
    if (W <= 0 || H <= 0) return;

    size_t need = static_cast<size_t>(W) * H * 4;
    if (g_glScratch.size() < need) g_glScratch.resize(need);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, g_glScratch.data());

    uint8_t* dst = g_ipc->mirror_frame;
    // GL is bottom-up, we want upright. sy index from top.
    for (int y = 0; y < FRAME_H; y++) {
        int sy = (H - 1) - (y * H / FRAME_H);
        if (sy < 0) sy = 0;
        for (int x = 0; x < FRAME_W; x++) {
            int sx = x * W / FRAME_W;
            const uint8_t* p = &g_glScratch[(static_cast<size_t>(sy) * W + sx) * 4];
            // Rec.601 luma; integer math.
            dst[y * FRAME_W + x] =
                static_cast<uint8_t>((p[0] * 299 + p[1] * 587 + p[2] * 114) / 1000);
        }
    }

    std::atomic_thread_fence(std::memory_order_release);
    g_ipc->mirror_frame_seq += 1;
}

void release_all_checkpoints() {
    for (auto& kv : g_checkpoints) {
        if (kv.second) {
            kv.second->release();
        }
    }
    g_checkpoints.clear();
}

} // namespace

$on_mod(Loaded) {
    ensure_ipc();
    log::info("GDRL-TP mod loaded (v{})", IPC_VERSION);
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
            g_ipc->reset_target_pct = 0;
            g_ipc->level_length = (level ? this->m_levelLength : 0.f);
            g_ipc->current_level_id = (level ? static_cast<uint32_t>(level->m_levelID.value()) : 0u);
            g_episodeId += 1;
            g_ipc->episode_id = g_episodeId;
            g_actionWasPressed = false;
            g_wasDead = false;
            g_wasLevelDone = false;
            g_prevX = 0.f;
            g_lastBucketCaptured = -1;
        }
        log::info("GDRL-TP PlayLayer::init ok={} episode={} level_id={} length={}",
            ok, g_episodeId, g_ipc ? g_ipc->current_level_id : 0u,
            g_ipc ? g_ipc->level_length : 0.f);
        return ok;
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        ensure_ipc();
        if (!g_ipc) return;

        g_ipc->tick += 1;
        g_frameTick += 1;

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

        bool jumpHeld = false;
        auto it = p->m_holdingButtons.find(static_cast<int>(PlayerButton::Jump));
        if (it != p->m_holdingButtons.end()) jumpHeld = it->second;
        g_ipc->player_input = jumpHeld ? 1 : 0;

        bool isDead = p->m_isDead;
        bool levelDone = g_ipc->level_done != 0;
        bool episodeEnded = (isDead && !g_wasDead) || (levelDone && !g_wasLevelDone);

        // online checkpoint capture: every new 5%-bucket crossed in a healthy frame
        if (!isDead && this->m_levelLength > 0.f) {
            float pct = (xNow / this->m_levelLength) * 100.0f;
            int bucket = static_cast<int>(pct) / CHECKPOINT_BUCKET_PCT;
            if (bucket > g_lastBucketCaptured && bucket > 0 && bucket < CHECKPOINT_BUCKETS_PER_LEVEL) {
                CheckpointObject* cp = this->createCheckpoint();
                if (cp) {
                    cp->retain();
                    uint64_t k = cp_key(g_ipc->current_level_id, bucket);
                    auto pr = g_checkpoints.find(k);
                    if (pr != g_checkpoints.end() && pr->second) {
                        pr->second->release();
                    }
                    g_checkpoints[k] = cp;
                    g_lastBucketCaptured = bucket;
                }
            }
        }

        // ring slot
        uint32_t idx = g_ipc->write_index;
        FrameSlot& slot = g_ipc->frames[idx % RING_CAPACITY];
        slot.tick = g_ipc->tick;
        std::memcpy(slot.obs, g_ipc->obs, sizeof(slot.obs));
        slot.player_input = g_ipc->player_input;
        slot.level_done = g_ipc->level_done;
        slot.is_dead = isDead ? 1 : 0;
        slot.episode_id = g_ipc->episode_id;
        std::atomic_thread_fence(std::memory_order_release);
        g_ipc->write_index = idx + 1;

        if (episodeEnded) {
            if (isDead && !g_wasDead) {
                g_lastBucketDeath = static_cast<int>(
                    (xNow / std::max(this->m_levelLength, 1.f)) * 100.0f
                ) / CHECKPOINT_BUCKET_PCT;
            }
            g_episodeId += 1;
            g_ipc->episode_id = g_episodeId;
        }
        g_wasDead = isDead;
        g_wasLevelDone = levelDone;

        // frame capture (gated)
        if ((g_frameTick % FRAME_CAPTURE_EVERY) == 0) {
            capture_frame_into_mirror();
        }

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

        // reset request (with optional mid-level spawn)
        if (g_ipc->ctrl_flags & 0x01) {
            g_ipc->ctrl_flags &= ~0x01;
            uint8_t targetPct = g_ipc->reset_target_pct;
            g_ipc->reset_target_pct = 0;
            uint32_t lvlId = g_ipc->current_level_id;

            g_prevX = 0.f;
            g_actionWasPressed = false;
            g_lastBucketCaptured = -1;

            geode::Loader::get()->queueInMainThread([targetPct, lvlId]() {
                auto* pl = PlayLayer::get();
                if (!pl) return;
                CheckpointObject* cp = nullptr;
                if (targetPct > 0) {
                    int bucket = std::min(
                        targetPct / CHECKPOINT_BUCKET_PCT,
                        CHECKPOINT_BUCKETS_PER_LEVEL - 1);
                    for (int b = bucket; b >= 1; b--) {
                        auto it = g_checkpoints.find(cp_key(lvlId, b));
                        if (it != g_checkpoints.end() && it->second) {
                            cp = it->second;
                            break;
                        }
                    }
                }
                pl->resetLevel();
                if (cp) {
                    pl->loadFromCheckpoint(cp);
                }
            });
        }

        // load-level request
        if (g_ipc->ctrl_flags & 0x02) {
            g_ipc->ctrl_flags &= ~0x02;
            uint32_t reqLvl = g_ipc->requested_level_id;
            geode::Loader::get()->queueInMainThread([reqLvl]() {
                auto* glm = GameLevelManager::sharedState();
                if (!glm) {
                    log::error("GDRL-TP load_level: GameLevelManager null");
                    return;
                }
                auto* level = glm->getMainLevel(static_cast<int>(reqLvl), false);
                if (!level) {
                    log::error("GDRL-TP load_level: level id={} not found", reqLvl);
                    return;
                }
                auto* scene = PlayLayer::scene(level, false, false);
                if (!scene) return;
                CCDirector::sharedDirector()->replaceScene(
                    cocos2d::CCTransitionFade::create(0.0f, scene));
            });
        }

        // death-respawn cleanup
        if (isDead && !g_wasDead) {
            g_prevX = 0.f;
            g_actionWasPressed = false;
            g_lastBucketCaptured = -1;
        }

        static int dbg = 0;
        if (++dbg % 600 == 0) {
            log::info("GDRL-TP alive tick={} write_idx={} ep={} objs={} cp_count={}",
                g_ipc->tick, g_ipc->write_index, g_episodeId, objCount,
                (int)g_checkpoints.size());
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

class $modify(GDRLTPAppDelegate, AppDelegate) {
    void applicationWillResignActive() {
        // keep training even when window loses focus
    }
    void applicationDidEnterBackground() {
        // keep training; do NOT call default which would pause and break IPC pacing
    }
};
