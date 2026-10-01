#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <memory>

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
};

/**
 * Online runtime for executing a frozen offline artifact.
 *
 * Milestone 1 focuses on correctness of segmented execution.
 * All supported segment/resource session variants are prepared by load(),
 * while run() only executes already-created sessions and propagates tensors
 * between segment boundaries. I/O binding is used for explicit input/output
 * placement, while adaptive scheduling is introduced in later milestones.
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
