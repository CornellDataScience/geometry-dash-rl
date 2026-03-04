#pragma once
#include "shared_memory.hpp"
#include "object_ids.hpp"
#include <Geode/Geode.hpp>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace geode::prelude;

struct NearbyObject {
    float relX, relY, category, isHazard, isInteractable, sizeClass;
    float distSq; // for sorting only, not written to SHM
};

inline void extractPlayerState(PlayLayer* pl, RLSharedMemory* shm) {
    auto* player = pl->m_player1;
    auto pos = player->getPosition();

    shm->playerX = pos.x;
    shm->playerY = pos.y;
    shm->velocityY = static_cast<float>(player->m_yVelocity);
    shm->rotation = player->getRotation();

    // Game mode detection
    if (player->m_isShip)       shm->gameMode = 1;
    else if (player->m_isBall)  shm->gameMode = 2;
    else if (player->m_isBird)  shm->gameMode = 3; // UFO
    else                        shm->gameMode = 0;  // cube (default)

    shm->isOnGround    = player->m_isOnGround ? 1 : 0;
    shm->isDead        = player->m_isDead ? 1 : 0;
    shm->speedMultiplier = player->m_playerSpeed;
    shm->gravityFlipped = player->m_isUpsideDown ? 1 : 0;
    shm->levelPercent  = static_cast<float>(pl->getCurrentPercentInt());
}

inline void extractNearbyObjects(PlayLayer* pl, RLSharedMemory* shm) {
    float playerX = shm->playerX;
    float playerY = shm->playerY;

    std::vector<NearbyObject> found;
    found.reserve(64);

    // m_objects is a CCArray* on GJBaseGameLayer containing all level objects.
    // If this doesn't compile, check generated headers — may be named differently.
    auto* objects = pl->m_objects;
    if (!objects) {
        shm->numObjectsFound = 0;
        std::memset(shm->objects, 0, sizeof(shm->objects));
        return;
    }

    // Scan window: 100 behind to 800 ahead of player
    float minX = playerX - 100.0f;
    float maxX = playerX + 800.0f;

    // Geode v5 removed CCARRAY_FOREACH — use CCArrayExt range-for instead
    for (auto* obj : CCArrayExt<GameObject*>(objects)) {
        if (!obj) continue;

        auto objPos = obj->getPosition();
        if (objPos.x < minX || objPos.x > maxX) continue;

        int objID = obj->m_objectID;

        // Solid detection: check if object has solid-type collision.
        // This may need adjustment after inspecting generated headers.
        // For now, we don't detect solids via ID — only via the isSolid param.
        bool isSolid = false;
        // TODO: check obj->m_objectType or similar property for solid detection
        // after verifying headers. For now, solid blocks are not categorized.

        int cat = categorizeObject(objID, isSolid);
        if (cat == 0) continue; // decoration — skip

        float relX = objPos.x - playerX;
        float relY = objPos.y - playerY;

        auto contentSize = obj->getContentSize();
        float maxDim = std::max(
            contentSize.width * std::abs(obj->getScaleX()),
            contentSize.height * std::abs(obj->getScaleY())
        );

        NearbyObject no;
        no.relX = relX;
        no.relY = relY;
        no.category = static_cast<float>(cat);
        no.isHazard = (cat == 1) ? 1.0f : 0.0f;
        no.isInteractable = (cat >= 2 && cat <= 5) ? 1.0f : 0.0f;
        no.sizeClass = maxDim;
        no.distSq = relX * relX + relY * relY;
        found.push_back(no);
    }

    // Sort by distance, take closest 25
    std::sort(found.begin(), found.end(),
              [](const NearbyObject& a, const NearbyObject& b) {
                  return a.distSq < b.distSq;
              });

    int count = std::min(static_cast<int>(found.size()), 25);
    shm->numObjectsFound = count;

    for (int i = 0; i < count; i++) {
        shm->objects[i][0] = found[i].relX;
        shm->objects[i][1] = found[i].relY;
        shm->objects[i][2] = found[i].category;
        shm->objects[i][3] = found[i].isHazard;
        shm->objects[i][4] = found[i].isInteractable;
        shm->objects[i][5] = found[i].sizeClass;
    }
    // Zero-pad remaining slots
    for (int i = count; i < 25; i++) {
        std::memset(shm->objects[i], 0, 6 * sizeof(float));
    }
}

inline void computeReward(RLSharedMemory* shm) {
    float currentX = shm->playerX;

    if (shm->isDead) {
        shm->reward = -10.0f;
        shm->done = 1;
    } else if (shm->levelPercent >= 100.0f) {
        shm->reward = 100.0f;
        shm->done = 1;
    } else {
        shm->reward = (currentX - shm->prevPlayerX) * 0.1f;
        shm->done = 0;
    }

    shm->prevPlayerX = currentX;
}
