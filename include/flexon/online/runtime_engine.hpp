#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <memory>
#include <vector>

namespace flexon::online {

struct FlexOnRuntimeImpl;

enum class RuntimeResource : std::uint8_t {
    CPU,
    CUDA,
    Auto
};

struct RunOptions {
    std::uint32_t iterations{1};
    std::uint32_t level{0};
    RuntimeResource resource{RuntimeResource::CPU};
    // Optional per-segment execution plan. When non-empty, its size must
    // match the selected level segment count.
    std::vector<RuntimeResource> resource_plan;
    // Enable FlexOn's AD/MI segmentation-level selection across inference
    // periods. The supplied level is used as the initial level.
    bool adaptive_level{false};
    // Optional scheduler configuration. An empty path uses
    // config/scheduler.yaml when present, otherwise paper defaults.
    std::filesystem::path scheduler_config_path;
};

/**
 * Online runtime for executing a frozen offline artifact.
 *
 * Milestone 1/2/3/4 runtime for executing a frozen offline artifact.
 * All supported segment/resource session variants are prepared by load().
 * Segment execution uses I/O binding and explicit CPU/CUDA tensor placement;
 * cross-resource boundary copies are measured separately from compute time.
 * Segment-boundary buffers are reused by the Milestone-3 activation arena
 * when tensor shape, type, resource, and lifetime permit.
 * Milestone 4 adds the paper's AD/MI level selection and degradation-aware
 * per-segment resource selection when RuntimeResource::Auto is requested.
 */
class FlexOnRuntime {
public:
    FlexOnRuntime();
    ~FlexOnRuntime();

    FlexOnRuntime(FlexOnRuntime&&) noexcept;
    FlexOnRuntime& operator=(FlexOnRuntime&&) noexcept;

    FlexOnRuntime(const FlexOnRuntime&) = delete;
    FlexOnRuntime& operator=(const FlexOnRuntime&) = delete;

    /** Load an offline artifact and prepare executable segment variants. */
    void load(const std::filesystem::path& artifact_directory);

    /**
     * Execute a complete segmentation level.
     *
     * The first segment receives generated deterministic inputs based on its
     * ONNX input metadata. Later segments consume tensors produced by earlier
     * segments in the same level.
     */
    void run(const RunOptions& options);

    bool loaded() const noexcept;
    std::uint32_t level_count() const;
    std::string model_name() const;

private:
    std::unique_ptr<FlexOnRuntimeImpl> impl_;
};

}  // namespace flexon::online
