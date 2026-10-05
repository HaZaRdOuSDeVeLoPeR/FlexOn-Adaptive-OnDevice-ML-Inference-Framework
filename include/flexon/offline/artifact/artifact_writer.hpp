#pragma once

#include <filesystem>

#include <flexon/core/types.hpp>
#include <flexon/offline/config/offline_config.hpp>
#include <flexon/offline/segmentation/multi_level_segmenter.hpp>
#include <onnx/onnx_pb.h>


namespace flexon::offline::artifact {

class ArtifactWriter {
public:
    static void write(
        const std::filesystem::path& directory,
        const std::filesystem::path& source_model_path,
        const onnx::ModelProto& model,
        const core::GraphInfo& graph,
        const core::OperatorProfileMap& operator_profiles,
        const segmentation::SegmentationResult& result,
        const config::OfflineConfig& config);
};

}  // namespace flexon::offline::artifact
