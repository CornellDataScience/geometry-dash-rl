// Skeleton only: Geode implementation placeholder
// Goal: write privileged obs into shared memory + consume action byte

#include <cstdint>
#include <cstring>

// TODO when wiring Geode SDK:
// #include <Geode/Geode.hpp>
// #include <Geode/modify/PlayLayer.hpp>

namespace gdrl {

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

class Bridge {
public:
    bool init() {
        // TODO: create/open platform shared memory named "gdrl_ipc"
        // Windows: CreateFileMapping/MapViewOfFile
        // macOS/Linux: shm_open + ftruncate + mmap
        return true;
    }

    void onFrame(/*PlayLayer* pl*/) {
        if (!buf_) return;

        // TODO: read from Geode player/object state
        // buf_->obs[0] = player x
        // buf_->obs[1] = player y
        // ... fill obs fields per protocol doc

        buf_->tick += 1;

        // TODO: consume action byte
        // uint8_t action = buf_->action_in;
        // apply input press/release in GD
    }

private:
    IPCBufferV0* buf_ = nullptr;
};

} // namespace gdrl
