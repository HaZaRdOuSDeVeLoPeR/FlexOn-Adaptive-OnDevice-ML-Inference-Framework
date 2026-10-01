#include <flexon/offline/shapes/runtime_shape_resolver.hpp>
#include <onnx/shape_inference/implementation.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace flexon::offline::shapes {

namespace {

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FlexOnShapeResolver");
    return env;
}

std::size_t element_size(onnx::TensorProto_DataType type) {
    switch (type) {
        case onnx::TensorProto_DataType_FLOAT: return 4;
        case onnx::TensorProto_DataType_UINT8: return 1;
        case onnx::TensorProto_DataType_INT8: return 1;
        case onnx::TensorProto_DataType_UINT16: return 2;
        case onnx::TensorProto_DataType_INT16: return 2;
        case onnx::TensorProto_DataType_INT32: return 4;
        case onnx::TensorProto_DataType_INT64: return 8;
        case onnx::TensorProto_DataType_BOOL: return 1;
        case onnx::TensorProto_DataType_FLOAT16: return 2;
        case onnx::TensorProto_DataType_DOUBLE: return 8;
        case onnx::TensorProto_DataType_UINT32: return 4;
        case onnx::TensorProto_DataType_UINT64: return 8;
        case onnx::TensorProto_DataType_BFLOAT16: return 2;
        case onnx::TensorProto_DataType_COMPLEX64: return 8;
        case onnx::TensorProto_DataType_COMPLEX128: return 16;
        default:
            throw std::runtime_error(
                "Unsupported ONNX input tensor type " +
                std::to_string(static_cast<int>(type)) +
                " during runtime shape probing");
    }
}

std::size_t element_count(const std::vector<std::int64_t>& shape,
                          const std::string& name) {
    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error(
                "Input '" + name + "' has an unresolved/non-positive dimension");
        }
        const auto d = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / d) {
            throw std::runtime_error(
                "Input '" + name + "' element count overflows size_t");
        }
        count *= d;
    }
    return count;
}

std::vector<std::int64_t> concrete_shape(const onnx::ValueInfoProto& input) {
    if (!input.has_type() || !input.type().has_tensor_type() ||
        !input.type().tensor_type().has_shape()) {
        throw std::runtime_error(
            "Model input '" + input.name() + "' has no concrete tensor shape");
    }

    std::vector<std::int64_t> shape;
    for (const auto& dim : input.type().tensor_type().shape().dim()) {
        if (!dim.has_dim_value() || dim.dim_value() <= 0) {
            throw std::runtime_error(
                "Model input '" + input.name() +
                "' still has an unresolved dimension during runtime shape probing");
        }
        shape.push_back(dim.dim_value());
    }
    return shape;
}

Ort::Value make_input(const onnx::ValueInfoProto& input,
                      std::vector<std::vector<std::uint8_t>>& buffers,
                      std::size_t index) {
    const auto type = static_cast<onnx::TensorProto_DataType>(
        input.type().tensor_type().elem_type());
    const auto shape = concrete_shape(input);
    const auto count = element_count(shape, input.name());

    if (type == onnx::TensorProto_DataType_STRING) {
        throw std::runtime_error(
            "Runtime shape probing does not currently support STRING "
                "input tensor '" + input.name() + "'");
    }

    const auto bytes = count * element_size(type);
    buffers[index].assign(bytes, 0U);

    // Use a small deterministic non-zero byte pattern. Shape probing should
    // execute the graph normally while avoiding assumptions about model data.
    for (auto& byte : buffers[index]) {
        byte = 1U;
    }

    auto memory = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(
        memory,
        buffers[index].data(),
        buffers[index].size(),
        shape.data(), shape.size(),
        static_cast<ONNXTensorElementDataType>(type));
}

void append_probe_output(onnx::GraphProto* graph,
                         const core::GraphInfo& info,
                         const std::string& tensor_name) {
    const auto it = info.tensors.find(tensor_name);

    auto* output = graph->add_output();
    output->set_name(tensor_name);

    // If static analysis already knows the type, preserve it.
    // Otherwise leave the type unspecified and let the runtime/inference
    // machinery resolve it.
    if (it != info.tensors.end() && it->second.data_type != 0) {
        output->mutable_type()->mutable_tensor_type()->set_elem_type(
            it->second.data_type);
    }
}

void probe_one(const onnx::ModelProto& source,
               core::GraphInfo& graph,
               const std::string& tensor_name) {
    onnx::ModelProto probe_model = source;
    auto* probe_graph = probe_model.mutable_graph();

    bool already_output = false;
    for (const auto& output : probe_graph->output()) {
        if (output.name() == tensor_name) {
            already_output = true;
            break;
        }
    }
    if (!already_output) {
        append_probe_output(probe_graph, graph, tensor_name);
    }

    // The source model may not contain ValueInfo metadata for every
    // intermediate tensor. After exposing the tensor as a graph output,
    // run ONNX shape/type inference on this temporary probe model so
    // the output can acquire its element type and shape metadata.
    try {
        onnx::shape_inference::InferShapes(probe_model);
    } catch (const std::exception&) {
        // Runtime execution below may still be able to determine the
        // concrete shape. Do not modify the original source model.
    }

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    options.SetIntraOpNumThreads(1);

    const auto serialized = probe_model.SerializeAsString();
    Ort::Session session(ort_env(), serialized.data(), serialized.size(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    const auto input_count = session.GetInputCount();
    std::vector<Ort::Value> inputs;
    std::vector<const char*> names;
    std::vector<std::string> name_storage;
    std::vector<std::vector<std::uint8_t>> buffers(input_count);
    std::vector<std::vector<std::string>> string_buffers(input_count);

    inputs.reserve(input_count);
    names.reserve(input_count);
    name_storage.reserve(input_count);

    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        name_storage.emplace_back(name.get());
        names.push_back(name_storage.back().c_str());

        // Find the corresponding input ValueInfo in the concrete source graph.
        const auto it = std::find_if(
            probe_graph->input().begin(), probe_graph->input().end(),
            [&](const auto& value) {
                return value.name() == name_storage.back();
            });
        if (it == probe_graph->input().end()) {
            throw std::runtime_error(
                "Could not find source metadata for model input '" +
                name_storage.back() + "'");
        }

        inputs.emplace_back(
            make_input(*it, buffers, i));
    }

    const char* output_name = tensor_name.c_str();
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        names.data(), inputs.data(), inputs.size(),
        &output_name, 1);

    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error(
            "Runtime shape probe did not produce a tensor for '" + tensor_name + "'");
    }

    const auto shape_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = shape_info.GetShape();

    auto& tensor = graph.tensors.at(tensor_name);
    tensor.shape = shape;
    tensor.shape_known = true;
    if (tensor.data_type == 0) {
        tensor.data_type = static_cast<int>(shape_info.GetElementType());
    }
}

}  // namespace

void RuntimeShapeResolver::resolve(const onnx::ModelProto& model,
                                   core::GraphInfo& graph) {
    std::vector<std::string> unresolved;
    unresolved.reserve(graph.tensors.size());

    for (const auto& [name, tensor] : graph.tensors) {
        if (!tensor.shape_known &&
            graph.producer_operator.find(name) != graph.producer_operator.end()) {
            unresolved.push_back(name);
        }
    }

    if (unresolved.empty()) {
        return;
    }

    std::sort(unresolved.begin(), unresolved.end());

    for (const auto& tensor_name : unresolved) {
        probe_one(model, graph, tensor_name);
    }
}

}  // namespace flexon::offline::shapes
