#pragma once

#include <vector>

#include <flexon/core/types.hpp>
#include <flexon/offline/config/offline_config.hpp>
#include <flexon/offline/profiling/segment_profiler.hpp>
#include <onnx/onnx_pb.h>


namespace flexon::offline::segmentation {

struct SegmentationResult {
    std::vector<std::vector<core::SegmentProfile>> levels;
    std::vector<core::LevelStatistics> level_statistics;
};

class MultiLevelSegmenter {
public:
    static SegmentationResult build(
        const onnx::ModelProto& model,
        const core::GraphInfo& graph,
        const std::vector<core::SegmentInfo>& initial,
        const core::OperatorProfileMap& operator_profiles,
        profiling::SegmentProfiler& profiler,
        const config::OfflineConfig& config);
};

}  // namespace flexon::offline::segmentation
