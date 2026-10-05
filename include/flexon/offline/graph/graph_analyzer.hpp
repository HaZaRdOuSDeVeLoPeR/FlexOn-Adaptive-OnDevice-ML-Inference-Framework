#pragma once

#include <onnx/onnx_pb.h>
#include <flexon/core/types.hpp>

namespace flexon::offline::graph {

/**
 * Converts the source ONNX graph into FlexOn's data-only GraphInfo.
 *
 * The analyzer is intentionally structural:
 *
 *   ONNX ModelProto
 *          ↓
 *      GraphInfo
 *
 * It does not perform:
 *   - execution-provider assignment
 *   - shape inference
 *   - segmentation
 *   - performance measurement
 *   - scheduling
 */
class GraphAnalyzer {
public:
    static core::GraphInfo analyze(const onnx::ModelProto& model);
};

}  // namespace flexon::offline::graph