#pragma once

#include <filesystem>

namespace flexon::api {

/**
 * High-level facade for users of the FlexOn framework.
 *
 * Offline profiling/segmentation and online execution remain separate
 * internally while this facade exposes a small public surface.
 */
class FlexOn {
public:
    FlexOn() = default;

    /**
     * Execute the offline compilation/profiling pipeline.
     */
    void profile(const std::filesystem::path& model_path,
                 const std::filesystem::path& artifact_directory);

    /**
     * Run end-to-end inference using previously generated artifacts.
     */
    void infer(const std::filesystem::path& artifact_directory);
};

}  // namespace flexon::api
