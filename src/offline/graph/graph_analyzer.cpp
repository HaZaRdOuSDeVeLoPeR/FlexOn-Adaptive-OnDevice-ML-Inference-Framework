#include <flexon/offline/graph/graph_analyzer.hpp>

#include <onnx/shape_inference/implementation.h>

#include <stdexcept>
#include <string>

namespace flexon::offline::graph {

namespace {

std::string node_name(const onnx::NodeProto& node,
                      std::uint32_t graph_index) {
    if (!node.name().empty()) {
        return node.name();
    }

    // ONNX node names are optional. Keep the source graph position
    // visible through a deterministic synthetic name.
    return "node_" + std::to_string(graph_index);
}


void add_tensor_info(const onnx::ValueInfoProto& value,
                     core::GraphInfo& result) {
    if (!value.has_type() || !value.type().has_tensor_type()) {
        return;
    }

    core::TensorInfo info;
    info.name = value.name();
    info.data_type = value.type().tensor_type().elem_type();

    if (value.type().tensor_type().has_shape()) {
        info.shape_known = true;
        for (const auto& dim : value.type().tensor_type().shape().dim()) {
            if (dim.has_dim_value()) {
                info.shape.push_back(dim.dim_value());
            } else {
                info.shape.push_back(-1);
                info.shape_known = false;
            }
        }
    }

    result.tensors[info.name] = std::move(info);
}

void ensure_tensor_info(const std::string& name,
                        core::GraphInfo& result) {
    if (name.empty()) {
        return;
    }

    if (result.tensors.find(name) == result.tensors.end()) {
        core::TensorInfo info;
        info.name = name;
        info.shape_known = false;
        result.tensors.emplace(name, std::move(info));
    }
}

}  // namespace

core::GraphInfo GraphAnalyzer::analyze(const onnx::ModelProto& model) {
    if (!model.has_graph()) {
        throw std::invalid_argument(
            "Cannot analyze ONNX model without a graph");
    }

    onnx::ModelProto inferred_model = model;
    const onnx::ModelProto* source_model;

    // Shape inference is best-effort. A model with custom/unsupported
    // operators can still be structurally analyzed.
    try {
        onnx::shape_inference::InferShapes(inferred_model);
        source_model = &inferred_model;
    } catch (const std::exception&) {
        source_model = &model;
    }

    const auto& graph = source_model->graph();

    core::GraphInfo result;
    result.name = graph.name();

    // ---------------------------------------------------------------------
    // Graph inputs and outputs
    // ---------------------------------------------------------------------

    result.graph_inputs.reserve(
        static_cast<std::size_t>(graph.input_size()));

    for (const auto& input : graph.input()) {
        result.graph_inputs.push_back(input.name());
    }

    result.graph_outputs.reserve(
        static_cast<std::size_t>(graph.output_size()));

    for (const auto& output : graph.output()) {
        result.graph_outputs.push_back(output.name());
    }

    // ---------------------------------------------------------------------
    // Initializers
    // ---------------------------------------------------------------------

    result.initializers.reserve(
        static_cast<std::size_t>(graph.initializer_size()));

    for (const auto& initializer : graph.initializer()) {
        result.initializers.push_back(initializer.name());
    }

    // ---------------------------------------------------------------------
    // Operators and tensor connectivity
    // ---------------------------------------------------------------------

    result.operators.reserve(
        static_cast<std::size_t>(graph.node_size()));

    for (int i = 0; i < graph.node_size(); ++i) {
        const auto& node = graph.node(i);
        const auto graph_index = static_cast<std::uint32_t>(i);

        core::OperatorInfo operator_info;
        operator_info.graph_index = graph_index;
        operator_info.name = node_name(node, graph_index);
        operator_info.op_type = node.op_type();

        operator_info.inputs.reserve(
            static_cast<std::size_t>(node.input_size()));

        for (const auto& input : node.input()) {
            operator_info.inputs.push_back(input);
        }

        operator_info.outputs.reserve(
            static_cast<std::size_t>(node.output_size()));

        for (const auto& output : node.output()) {
            operator_info.outputs.push_back(output);
        }

        result.operators.push_back(std::move(operator_info));

        // Every output tensor produced by this operator gets a producer
        // entry. ONNX should not contain multiple producers for the same
        // tensor in a valid graph.
        for (const auto& output : node.output()) {
            if (output.empty()) {
                // Optional/unused outputs can be represented by an empty
                // string in ONNX. They are not graph tensors.
                continue;
            }

            const auto [it, inserted] =
                result.producer_operator.emplace(output, graph_index);

            if (!inserted) {
                throw std::runtime_error(
                    "Multiple operators produce tensor '" + output + "'");
            }
        }

        // Every input tensor gets a consumer entry.
        for (const auto& input : node.input()) {
            if (input.empty()) {
                continue;
            }

            result.consumer_operators[input].push_back(graph_index);
        }
    }

    for (const auto& input : graph.input()) {
        add_tensor_info(input, result);
    }
    for (const auto& value : graph.value_info()) {
        add_tensor_info(value, result);
    }
    for (const auto& output : graph.output()) {
        add_tensor_info(output, result);
    }

    for (const auto& initializer : graph.initializer()) {
        core::TensorInfo info;
        info.name = initializer.name();
        info.data_type = initializer.data_type();
        info.shape_known = true;
        for (const auto dim : initializer.dims()) {
            info.shape.push_back(dim);
        }
        result.tensors[info.name] = std::move(info);
    }

    for (const auto& node : graph.node()) {
        for (const auto& input : node.input()) {
            ensure_tensor_info(input, result);
        }

        for (const auto& output : node.output()) {
            ensure_tensor_info(output, result);
        }
    }

    return result;
}

}  // namespace flexon::offline::graph