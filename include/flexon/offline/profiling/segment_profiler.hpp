#pragma once

#include <flexon/core/types.hpp>
#include <flexon/offline/config/offline_config.hpp>

#include <onnx/onnx_pb.h>

#include <filesystem>

namespace flexon::offline::profiling {

class SegmentProfiler {
public:
    SegmentProfiler(const config::OfflineConfig& config,
                    const std::filesystem::path& work_directory);

    core::SegmentProfile profile(
        const onnx::ModelProto& source,
        const core::GraphInfo& graph,
        const core::SegmentInfo& segment);

    /**
     * Probe and profile one source-graph operator on every enabled resource.
     *
     * Each enabled resource receives an explicit status: supported,
     * unsupported, or profiling-failed. Unsupported resources carry an
     * infinite cost rather than being omitted from the profile.
     */
    core::OperatorProfile profile_operator(
        const onnx::ModelProto& source,
        const core::GraphInfo& graph,
        std::uint32_t operator_index);

private:
    config::OfflineConfig config_;
    std::filesystem::path work_directory_;
};

}  // namespace flexon::offline::profiling
