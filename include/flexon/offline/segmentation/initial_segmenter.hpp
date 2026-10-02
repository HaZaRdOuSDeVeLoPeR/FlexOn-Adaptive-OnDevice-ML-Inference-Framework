#pragma once

#include <flexon/core/types.hpp>
#include <flexon/offline/config/offline_config.hpp>

#include <onnx/onnx_pb.h>

#include <vector>

namespace flexon::offline::segmentation {

class InitialSegmenter {
public:
    // Build the level-0 partition from measured resource capability.
    // Operators supported by exactly one enabled resource become explicit
    // resource-capability boundaries; operators supported by both resources
    // remain in the surrounding segment.
    static std::vector<core::SegmentInfo> create(
        const core::GraphInfo& graph,
        const config::OfflineConfig& config,
        const core::OperatorProfileMap& operator_profiles);
};

}  // namespace flexon::offline::segmentation
