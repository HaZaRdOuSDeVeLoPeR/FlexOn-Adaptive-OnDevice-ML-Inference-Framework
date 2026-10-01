#pragma once

#include <flexon/core/types.hpp>

#include <onnx/onnx_pb.h>

#include <filesystem>

namespace flexon::offline::model {

class SegmentModelGenerator {
public:
    static onnx::ModelProto build(
        const onnx::ModelProto& source,
        const core::GraphInfo& graph,
        const core::SegmentInfo& segment);

    static void save(const onnx::ModelProto& model,
                     const std::filesystem::path& path);
};

}  // namespace flexon::offline::model
