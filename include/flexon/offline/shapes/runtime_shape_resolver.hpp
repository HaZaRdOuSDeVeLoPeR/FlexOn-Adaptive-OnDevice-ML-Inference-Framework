#pragma once

#include <onnx/onnx_pb.h>
#include <flexon/core/types.hpp>

namespace flexon::offline::shapes {

/**
 * Resolves tensor shapes that static ONNX shape inference could not provide.
 *
 * The resolver is model-agnostic. It exposes unresolved intermediate tensors
 * as temporary graph outputs, executes the resolved source model with the
 * configured concrete graph-input shapes, and records the runtime shapes
 * observed for those tensors.
 */
class RuntimeShapeResolver {
public:
    static void resolve(const onnx::ModelProto& model,
                        core::GraphInfo& graph);
};

}  // namespace flexon::offline::shapes
