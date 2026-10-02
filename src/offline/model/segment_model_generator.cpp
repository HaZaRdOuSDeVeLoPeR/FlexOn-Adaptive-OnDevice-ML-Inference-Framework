#include <flexon/offline/model/segment_model_generator.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace flexon::offline::model {

namespace {

const onnx::ValueInfoProto* find_value(
    const onnx::GraphProto& graph, const std::string& name) {

    for (const auto& value : graph.input()) {
        if (value.name() == name) return &value;
    }
    for (const auto& value : graph.value_info()) {
        if (value.name() == name) return &value;
    }
    for (const auto& value : graph.output()) {
        if (value.name() == name) return &value;
    }
    return nullptr;
}

void add_value(
    onnx::GraphProto* graph,
    const onnx::GraphProto& source_graph,
    const core::GraphInfo& info,
    const std::string& name) {

    const auto tensor = info.tensors.find(name);
    if (tensor != info.tensors.end()) {
        if (!tensor->second.shape_known) {
            throw std::runtime_error(
                "Tensor '" + name +
                "' has no concrete shape for a generated segment input");
        }

        auto* value = graph->add_input();
        value->set_name(name);
        auto* type = value->mutable_type()->mutable_tensor_type();
        type->set_elem_type(
            tensor->second.data_type == 0
                ? onnx::TensorProto_DataType_FLOAT
                : tensor->second.data_type);

        auto* shape = type->mutable_shape();
        for (const auto dim : tensor->second.shape) {
            if (dim <= 0) {
                throw std::runtime_error(
                    "Tensor '" + name +
                    "' has a non-positive dimension for a generated segment input");
            }
            shape->add_dim()->set_dim_value(dim);
        }
        return;
    }

    if (const auto* value = find_value(source_graph, name)) {
        // A source value without GraphInfo metadata is not sufficient for a
        // generated executable segment. Do not silently manufacture a shape.
        if (!value->has_type() || !value->type().has_tensor_type() ||
            !value->type().tensor_type().has_shape()) {
            throw std::runtime_error(
                "Tensor '" + name +
                "' has no concrete shape for a generated segment input");
        }

        *graph->add_input() = *value;
        return;
    }

    throw std::runtime_error(
        "Missing metadata for generated segment input tensor '" + name + "'");
}

void add_output(
    onnx::GraphProto* graph,
    const onnx::GraphProto& source_graph,
    const core::GraphInfo& info,
    const std::string& name) {

    const auto tensor = info.tensors.find(name);

    if (tensor != info.tensors.end()) {
        auto* value = graph->add_output();
        value->set_name(name);
        auto* type = value->mutable_type()->mutable_tensor_type();
        type->set_elem_type(
            tensor->second.data_type == 0
                ? onnx::TensorProto_DataType_FLOAT
                : tensor->second.data_type);

        // Output shapes are useful when known, but they are not required for
        // ORT to execute a valid generated segment. ORT can infer the runtime
        // output shape from the graph itself.
        if (tensor->second.shape_known) {
            auto* shape = type->mutable_shape();
            for (const auto dim : tensor->second.shape) {
                if (dim <= 0) {
                    throw std::runtime_error(
                        "Tensor '" + name +
                        "' has a non-positive dimension for a generated segment output");
                }
                shape->add_dim()->set_dim_value(dim);
            }
        }
        return;
    }

    if (const auto* value = find_value(source_graph, name)) {
        *graph->add_output() = *value;
        return;
    }

    auto* value = graph->add_output();
    value->set_name(name);
    auto* type = value->mutable_type()->mutable_tensor_type();
    type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
}

}  // namespace

onnx::ModelProto SegmentModelGenerator::build(
    const onnx::ModelProto& source,
    const core::GraphInfo& graph,
    const core::SegmentInfo& segment) {

    if (!source.has_graph()) {
        throw std::invalid_argument("Source ONNX model has no graph");
    }

    const auto& source_graph = source.graph();

    std::unordered_set<std::uint32_t> members(
        segment.operator_indices.begin(),
        segment.operator_indices.end());

    onnx::ModelProto result;
    result.set_ir_version(source.ir_version());
    result.set_producer_name("FlexOn");
    result.set_producer_version("0.1.0");
    result.set_domain("flexon.segment");

    for (const auto& opset : source.opset_import()) {
        *result.add_opset_import() = opset;
    }

    auto* graph_out = result.mutable_graph();
    graph_out->set_name(
        source_graph.name() + "_level_" +
        std::to_string(segment.level) + "_segment_" +
        std::to_string(segment.id));

    // Nodes retain their original order.
    for (const auto index : segment.operator_indices) {
        if (index >= static_cast<std::uint32_t>(source_graph.node_size())) {
            throw std::out_of_range("Segment references invalid operator index");
        }
        *graph_out->add_node() = source_graph.node(
            static_cast<int>(index));
    }

    // Boundary inputs.
    for (const auto& name : segment.input_tensors) {
        if (graph.initializers.end() !=
            std::find(graph.initializers.begin(),
                      graph.initializers.end(), name)) {
            continue;
        }
        add_value(graph_out, source_graph, graph, name);
    }

    // Boundary outputs.
    for (const auto& name : segment.output_tensors) {
        add_output(graph_out, source_graph, graph, name);
    }

    // Weights/constants required by the selected nodes.
    std::unordered_set<std::string> required_inputs;
    for (const auto index : segment.operator_indices) {
        const auto& node = source_graph.node(static_cast<int>(index));
        for (const auto& input : node.input()) {
            required_inputs.insert(input);
        }
    }

    for (const auto& initializer : source_graph.initializer()) {
        if (required_inputs.count(initializer.name()) != 0) {
            *graph_out->add_initializer() = initializer;
        }
    }

    return result;
}

void SegmentModelGenerator::save(
    const onnx::ModelProto& model,
    const std::filesystem::path& path) {

    std::filesystem::create_directories(path.parent_path());

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Failed to create segment model: " + path.string());
    }

    if (!model.SerializeToOstream(&output)) {
        throw std::runtime_error(
            "Failed to serialize segment model: " + path.string());
    }
}

}  // namespace flexon::offline::model
