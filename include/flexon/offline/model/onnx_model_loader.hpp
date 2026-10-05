#pragma once

#include <filesystem>
#include <memory>

#include <onnx/onnx_pb.h>

namespace flexon::offline::model {

/**
 * Loads the source ONNX model used by offline analysis.
 *
 * The loader deliberately returns the ONNX ModelProto rather than an
 * ONNX Runtime object. The original ONNX graph is the authoritative
 * representation for FlexOn's offline graph analysis.
 */
class OnnxModelLoader {
public:
    static std::unique_ptr<onnx::ModelProto> load(
        const std::filesystem::path& model_path);
};

}  // namespace flexon::offline::model