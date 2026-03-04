#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include "shared_memory.hpp"
#include "state_extractor.hpp"
#include <thread>
#include <chrono>

using namespace geode::prelude;

static RLSharedMemory* g_shm = nullptr;
static bool g_rlActive = false;
static bool g_resetPending = false;
static bool g_wasClicking = false;

// Timeout duration for agent response
static constexpr auto AGENT_TIMEOUT = std::chrono::seconds(5);
// Spin-wait sleep interval (100 microseconds)
static constexpr auto SPIN_SLEEP = std::chrono::microseconds(100);

// Spin-wait until turn_flag becomes 0 (agent responded).
// Returns false on timeout.
static bool waitForAgent() {
    auto start = std::chrono::steady_clock::now();
    while (g_shm->turn_flag.load(std::memory_order_acquire) != 0) {
        std::this_thread::sleep_for(SPIN_SLEEP);
        if (std::chrono::steady_clock::now() - start > AGENT_TIMEOUT) {
            return false;
        }
    }
    return true;
}

$on_mod(Loaded) {
    g_shm = createSharedMemory();
    if (g_shm) {
        log::info("GD-RL: Shared memory created at {}", SHM_NAME);
        g_rlActive = true;
    } else {
        log::error("GD-RL: Failed to create shared memory!");
    }
}

// Note: Geode v5 has no "Unloaded" event. SHM is cleaned up on process exit.
// If needed, cleanup could be added via a destructor or atexit handler.

class $modify(RLPlayLayer, PlayLayer) {
    void update(float dt) {
        // Pass through if RL not active
        if (!g_shm || !g_rlActive) {
            PlayLayer::update(dt);
            return;
        }

        // --- Handle deferred reset ---
        // resetLevel() is deferred to avoid reentrancy issues when called
        // from inside the update hook.
        if (g_resetPending) {
            g_resetPending = false;
            this->resetLevel();
            g_shm->prevPlayerX = 0.0f;
            g_shm->attemptNumber++;
            g_shm->done = 0;
            g_shm->isDead = 0;
            g_wasClicking = false;

            // Extract post-reset state and signal agent
            extractPlayerState(this, g_shm);
            extractNearbyObjects(this, g_shm);
            g_shm->reward = 0.0f;
            g_shm->turn_flag.store(1, std::memory_order_release);

            if (!waitForAgent()) {
                log::warn("GD-RL: Agent timeout after reset, disabling RL");
                g_rlActive = false;
            }
            // Don't run physics on reset frame
            return;
        }

        // --- Step 1: Run original physics (uses last frame's action) ---
        PlayLayer::update(dt);

        // --- Step 2: Extract state ---
        extractPlayerState(this, g_shm);
        extractNearbyObjects(this, g_shm);
        computeReward(g_shm);

        // --- Step 3: Signal agent ---
        g_shm->turn_flag.store(1, std::memory_order_release);

        // --- Step 4: Wait for agent response ---
        if (!waitForAgent()) {
            log::warn("GD-RL: Agent timeout, disabling RL");
            g_rlActive = false;
            return;
        }

        // --- Step 5: Handle command ---
        if (g_shm->command == 1) {
            // Reset requested — defer to next frame
            g_resetPending = true;
            return;
        }
        if (g_shm->command == 2) {
            log::info("GD-RL: Shutdown command received");
            g_rlActive = false;
            return;
        }

        // --- Step 6: Apply action ---
        // Only call pushButton/releaseButton on state transitions to avoid
        // spamming the input system every frame.
        int action = g_shm->action;
        if (action == 1 && !g_wasClicking) {
            this->m_player1->pushButton(PlayerButton::Jump);
            g_wasClicking = true;
        } else if (action == 0 && g_wasClicking) {
            this->m_player1->releaseButton(PlayerButton::Jump);
            g_wasClicking = false;
        }
    }
};
