#pragma once

#include <filesystem>
#include <cstdint>

namespace flexon::online {

/**
 * Public online runtime API.
 *
 * The runtime will preload all executable segment/resource variants before
 * inference starts. Per-inference scheduling should therefore contain no
 * model/session construction.
 */
class FlexOnRuntime {
public:
    FlexOnRuntime() = default;

    /**
     * Load offline artifacts and prepare the runtime execution resources.
     */
    void load(const std::filesystem::path& artifact_directory);

    /**
     * Execute one inference period.
     *
     * The concrete tensor API will be introduced with the first executor
     * module so that ownership and device-memory semantics are explicit.
     */
    void run(std::uint32_t iterations);
};

}  // namespace flexon::online
