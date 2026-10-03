#include <iostream>
#include <random>

#include <flexon/online/execution/helper.hpp>

namespace flexon::online::helper {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

void print_segment_stats(
    const manifest::SegmentManifest& manifest,
    const executor::SegmentExecutionStats& stats,
    const arena::BufferArena& arena) {

    std::cout << "    segment " << manifest.id
              << " [" << resource_name(stats.resource) << "] "
              << stats.elapsed_ms << " ms"
              << " | boundary_copy=" << stats.boundary_copy_ms << " ms"
              << " | arena_allocations=" << arena.allocation_count()
              << " | arena_reuses=" << arena.reuse_count()
              << " | arena_bytes=" << arena.allocated_bytes()
              << " | arena_peak_live_bytes=" << arena.peak_live_bytes() << "\n";
}

const char* resource_name(core::Resource resource) {
    switch (resource) {
        case core::Resource::CPU: return "cpu";
        case core::Resource::CUDA: return "cuda";
        case core::Resource::Auto: return "auto";
    }
    return "unknown";
}

std::vector<std::int64_t> concrete_shape(std::vector<std::int64_t> shape) {
    for (auto& dim : shape) {
        if (dim < 0) {
            throw std::runtime_error(
                "Online milestone requires concrete segment input shapes; "
                "found a dynamic input dimension");
        }
    }
    return shape;
}

std::unique_ptr<Ort::Session> create_session(
    Ort::Env& env,
    const std::filesystem::path& model_path,
    core::Resource resource) {

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);

    if (resource == core::Resource::CUDA) {
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda);
    }

    try {
        return std::make_unique<Ort::Session>(
            env, model_path.c_str(), options);
    } catch (const Ort::Exception& ex) {
        throw std::runtime_error(
            std::string("Failed to create ") + resource_name(resource) +
            " session for segment model " + model_path.string() + ": " +
            ex.what());
    }
}

runtime::PreparedSession* find_session(runtime::SegmentRuntime& segment,
                               core::Resource resource) {
    for (auto& prepared : segment.sessions) {
        if (prepared.resource == resource) return &prepared;
    }
    return nullptr;
}

Ort::Value make_input_value(Ort::AllocatorWithDefaultOptions& allocator,
                            const Ort::TypeInfo& type_info,
                            std::uint32_t seed) {
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    const auto shape = concrete_shape(tensor_info.GetShape());
    const auto count = arena::element_count(shape);
    const auto type = tensor_info.GetElementType();

    auto make_zero_tensor = [&]() {
        return Ort::Value::CreateTensor<bool>(
            allocator, shape.data(), shape.size());
    };

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> float_dist(-1.0F, 1.0F);

    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            auto value = Ort::Value::CreateTensor<float>(
                allocator, shape.data(), shape.size());
            auto* data = static_cast<float*>(value.GetTensorMutableRawData());
            for (std::size_t i = 0; i < count; ++i) data[i] = float_dist(rng);
            return value;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            auto value = Ort::Value::CreateTensor<double>(
                allocator, shape.data(), shape.size());
            auto* data = static_cast<double*>(value.GetTensorMutableRawData());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < count; ++i) data[i] = dist(rng);
            return value;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            auto value = Ort::Value::CreateTensor<std::int64_t>(
                allocator, shape.data(), shape.size());
            auto* data = static_cast<std::int64_t*>(value.GetTensorMutableRawData());
            std::fill(data, data + count, 0);
            return value;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            auto value = Ort::Value::CreateTensor<std::int32_t>(
                allocator, shape.data(), shape.size());
            auto* data = static_cast<std::int32_t*>(value.GetTensorMutableRawData());
            std::fill(data, data + count, 0);
            return value;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            // ONNX Runtime represents BOOL tensors as one byte per element.
            auto value = make_zero_tensor();
            auto* data = static_cast<bool*>(value.GetTensorMutableRawData());
            std::fill(data, data + count, false);
            return value;
        }
        default:
            throw std::runtime_error(
                "Unsupported segment input tensor type in online milestone: " +
                std::to_string(static_cast<int>(type)));
    }
}

Ort::MemoryInfo memory_info_for_resource(core::Resource resource) {
    if (resource == core::Resource::CPU) {
        return Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemTypeDefault);
    }

    if (resource == core::Resource::CUDA) {
        return Ort::MemoryInfo(
            "Cuda",
            OrtAllocatorType::OrtDeviceAllocator,
            0,
            OrtMemTypeDefault);
    }

    throw std::invalid_argument(
        "A concrete CPU/CUDA resource is required for tensor placement");
}

void add_initial_inputs(
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    const std::vector<std::string>& input_names,
    Ort::Session& first_session,
    Ort::AllocatorWithDefaultOptions& allocator) {

    const auto count = first_session.GetInputCount();
    for (std::size_t i = 0; i < count; ++i) {
        auto name = first_session.GetInputNameAllocated(i, allocator);
        const std::string tensor_name(name.get());
        if (tensor_store.find(tensor_name) != tensor_store.end()) continue;
        auto type_info = first_session.GetInputTypeInfo(i);
        runtime::RuntimeTensor tensor;
        tensor.value = helper::make_input_value(
            allocator, type_info, static_cast<std::uint32_t>(i + 1));
        tensor_store.emplace(tensor_name, std::move(tensor));
    }

    (void)input_names;
}

} // namespace::flexon::online::execution