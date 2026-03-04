#pragma once
#include <unordered_set>

// Object ID → category mapping.
// IDs are community-reverse-engineered for GD 2.2081.
// Start with known common IDs; expand by enabling debug logging
// and playing through levels to discover unknown IDs.
//
// Categories:
//   0 = decoration/unknown (filtered out, not sent to agent)
//   1 = hazard (spikes, sawblades)
//   2 = jump orb (mid-air click-activated)
//   3 = jump pad (ground contact-activated)
//   4 = mode portal (cube/ship/ball/ufo/wave/robot/spider)
//   5 = speed portal (0.5x/1x/2x/3x/4x)
//   6 = solid block (platforms, walls) — detected via object properties, not ID

// Category 1: Hazards
inline const std::unordered_set<int> HAZARD_IDS = {
    // Basic spikes
    8, 39, 103, 392,
    // Spike variants
    9, 61, 243, 244,
    // Directional spikes
    363, 364, 365, 366, 367, 368,
    // Sawblades
    88, 89, 98, 397, 399,
    // 1.9/2.0 spikes
    135, 136, 137, 138, 139,
    // 2.1+ spike variants
    176, 177, 178, 179, 180, 181, 182, 183,
};

// Category 2: Jump orbs (click mid-air to activate)
inline const std::unordered_set<int> JUMP_ORB_IDS = {
    36,   // yellow orb
    84,   // pink orb
    141,  // red orb
    1022, // blue orb
    1330, // green orb
    1594, // black orb
    1704, // spider orb
    1751, // dash orb (2.2)
};

// Category 3: Jump pads (activate on ground contact)
inline const std::unordered_set<int> JUMP_PAD_IDS = {
    35,   // yellow pad
    67,   // pink pad
    140,  // red pad
    1332, // blue pad
    1333, // spider pad
};

// Category 4: Mode portals
inline const std::unordered_set<int> MODE_PORTAL_IDS = {
    12,   // cube portal
    13,   // ship portal
    47,   // ball portal
    111,  // UFO portal
    660,  // wave portal
    745,  // robot portal
    1331, // spider portal
};

// Category 5: Speed portals
inline const std::unordered_set<int> SPEED_PORTAL_IDS = {
    200,  // 0.5x (slow)
    201,  // 1x (normal)
    202,  // 2x (fast)
    203,  // 3x (faster)
    1334, // 4x (fastest)
};

// Returns: 0=decoration, 1=hazard, 2=orb, 3=pad, 4=mode portal, 5=speed portal, 6=solid
inline int categorizeObject(int objectID, bool isSolid) {
    if (HAZARD_IDS.count(objectID)) return 1;
    if (JUMP_ORB_IDS.count(objectID)) return 2;
    if (JUMP_PAD_IDS.count(objectID)) return 3;
    if (MODE_PORTAL_IDS.count(objectID)) return 4;
    if (SPEED_PORTAL_IDS.count(objectID)) return 5;
    if (isSolid) return 6;
    return 0; // decoration — skip
}
