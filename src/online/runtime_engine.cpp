#include <flexon/online/runtime_engine.hpp>

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flexon::online {
namespace {

struct SegmentManifest {
    std::uint32_t id{0};
    std::filesystem::path model_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    bool cpu_supported{false};
    bool cuda_supported{false};
    double cpu_mean_ms{std::numeric_limits<double>::infinity()};
    double cuda_mean_ms{std::numeric_limits<double>::infinity()};
};

struct LevelManifest {
    std::uint32_t level{0};
    std::vector<SegmentManifest> segments;
};

struct ArtifactManifest {
    std::filesystem::path directory;
    std::string model_name;
    std::vector<LevelManifest> levels;
};

const char* resource_name(RuntimeResource resource) {
    switch (resource) {
        case RuntimeResource::CPU: return "cpu";
        case RuntimeResource::CUDA: return "cuda";
        case RuntimeResource::Auto: return "auto";
    }
    return "unknown";
}

bool is_supported_status(const YAML::Node& cost) {
    return cost["status"] && cost["status"].as<std::string>() == "supported";
}

ArtifactManifest load_manifest(const std::filesystem::path& directory) {
    const auto manifest_path = directory / "manifest.yaml";
    if (!std::filesystem::exists(manifest_path)) {
        throw std::runtime_error("Offline artifact manifest not found: " +
                                 manifest_path.string());
    }

    const auto root = YAML::LoadFile(manifest_path.string());
    if (!root["format_version"] || root["format_version"].as<int>() != 2) {
        throw std::runtime_error("Unsupported offline artifact format version");
    }
    if (!root["model"] || !root["model"]["name"] || !root["levels"]) {
        throw std::runtime_error("Offline artifact is missing model or levels");
    }

    ArtifactManifest artifact;
    artifact.directory = directory;
    artifact.model_name = root["model"]["name"].as<std::string>();

    for (const auto& level_node : root["levels"]) {
        LevelManifest level;
        level.level = level_node["level"].as<std::uint32_t>();

        for (const auto& segment_node : level_node["segments"]) {
            SegmentManifest segment;
            segment.id = segment_node["id"].as<std::uint32_t>();
            segment.model_path =
                directory / segment_node["model"].as<std::string>();

            if (!std::filesystem::exists(segment.model_path)) {
                throw std::runtime_error(
                    "Artifact segment model is missing: " +
                    segment.model_path.string());
            }

            if (segment_node["inputs"]) {
                for (const auto& value : segment_node["inputs"]) {
                    segment.input_names.push_back(value.as<std::string>());
                }
            }
            if (segment_node["outputs"]) {
                for (const auto& value : segment_node["outputs"]) {
                    segment.output_names.push_back(value.as<std::string>());
                }
            }

            if (segment_node["costs"]) {
                for (const auto& cost : segment_node["costs"]) {
                    if (!cost["resource"] || !cost["status"]) {
                        continue;
                    }
                    const auto resource = cost["resource"].as<std::string>();
                    if (!is_supported_status(cost)) {
                        continue;
                    }
                    const double mean = cost["mean_ms"]
                        ? cost["mean_ms"].as<double>()
                        : std::numeric_limits<double>::infinity();
                    if (resource == "cpu") {
                        segment.cpu_supported = true;
                        segment.cpu_mean_ms = mean;
                    } else if (resource == "cuda") {
                        segment.cuda_supported = true;
                        segment.cuda_mean_ms = mean;
                    }
                }
            }

            if (!segment.cpu_supported && !segment.cuda_supported) {
                throw std::runtime_error(
                    "Segment has no supported execution resource: level=" +
                    std::to_string(level.level) + " segment=" +
                    std::to_string(segment.id));
            }

            level.segments.push_back(std::move(segment));
        }

        if (level.segments.empty()) {
            throw std::runtime_error(
                "Offline artifact contains an empty level: " +
                std::to_string(level.level));
        }
        artifact.levels.push_back(std::move(level));
    }

    if (artifact.levels.empty()) {
        throw std::runtime_error("Offline artifact contains no levels");
    }

    std::sort(artifact.levels.begin(), artifact.levels.end(),
              [](const auto& a, const auto& b) { return a.level < b.level; });
    return artifact;
}

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FlexOnOnline");
    return env;
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

std::size_t element_count(const std::vector<std::int64_t>& shape) {
    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("Negative tensor dimension");
        }
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(dim)) {
            throw std::runtime_error("Tensor is too large for host allocation");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

Ort::Value make_input_value(Ort::AllocatorWithDefaultOptions& allocator,
                            const Ort::TypeInfo& type_info,
                            std::uint32_t seed) {
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    const auto shape = concrete_shape(tensor_info.GetShape());
    const auto count = element_count(shape);
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

struct PreparedSession {
    RuntimeResource resource{RuntimeResource::CPU};
    std::unique_ptr<Ort::Session> session;
};

struct SegmentRuntime {
    SegmentManifest manifest;
    std::vector<PreparedSession> sessions;
};

struct RuntimeState {
    ArtifactManifest artifact;
    Ort::Env* env{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::vector<SegmentRuntime>> levels;
};

std::unique_ptr<Ort::Session> create_session(
    const std::filesystem::path& model_path,
    RuntimeResource resource) {

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);

    if (resource == RuntimeResource::CUDA) {
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda);
    }

    try {
        return std::make_unique<Ort::Session>(
            ort_env(), model_path.c_str(), options);
    } catch (const Ort::Exception& ex) {
        throw std::runtime_error(
            std::string("Failed to create ") + resource_name(resource) +
            " session for segment model " + model_path.string() + ": " +
            ex.what());
    }
}

PreparedSession* find_session(SegmentRuntime& segment,
                               RuntimeResource resource) {
    for (auto& prepared : segment.sessions) {
        if (prepared.resource == resource) return &prepared;
    }
    return nullptr;
}

RuntimeResource choose_resource(const SegmentRuntime& segment,
                                RuntimeResource requested) {
    if (requested == RuntimeResource::CPU) {
        if (!segment.manifest.cpu_supported) {
            throw std::runtime_error(
                "CPU execution requested for a CPU-unsupported segment " +
                std::to_string(segment.manifest.id));
        }
        return RuntimeResource::CPU;
    }
    if (requested == RuntimeResource::CUDA) {
        if (!segment.manifest.cuda_supported) {
            throw std::runtime_error(
                "CUDA execution requested for a CUDA-unsupported segment " +
                std::to_string(segment.manifest.id));
        }
        return RuntimeResource::CUDA;
    }

    if (segment.manifest.cpu_supported && segment.manifest.cuda_supported) {
        return segment.manifest.cuda_mean_ms < segment.manifest.cpu_mean_ms
                   ? RuntimeResource::CUDA
                   : RuntimeResource::CPU;
    }
    return segment.manifest.cpu_supported ? RuntimeResource::CPU
                                          : RuntimeResource::CUDA;
}

void add_initial_inputs(
    std::unordered_map<std::string, Ort::Value>& tensor_store,
    const std::vector<std::string>& input_names,
    Ort::Session& first_session,
    Ort::AllocatorWithDefaultOptions& allocator) {

    const auto count = first_session.GetInputCount();
    for (std::size_t i = 0; i < count; ++i) {
        auto name = first_session.GetInputNameAllocated(i, allocator);
        const std::string tensor_name(name.get());
        if (tensor_store.find(tensor_name) != tensor_store.end()) continue;
        auto type_info = first_session.GetInputTypeInfo(i);
        tensor_store.emplace(
            tensor_name,
            make_input_value(allocator, type_info, static_cast<std::uint32_t>(i + 1)));
    }

    (void)input_names;
}

void execute_segment(
    SegmentRuntime& segment,
    RuntimeResource requested_resource,
    std::unordered_map<std::string, Ort::Value>& tensor_store,
    Ort::AllocatorWithDefaultOptions& allocator) {

    const auto resource = choose_resource(segment, requested_resource);
    auto* prepared = find_session(segment, resource);
    if (!prepared || !prepared->session) {
        throw std::runtime_error(
            "Prepared session missing for " + std::string(resource_name(resource)) +
            " segment " + std::to_string(segment.manifest.id));
    }

    Ort::Session& session = *prepared->session;
    const auto input_count = session.GetInputCount();
    const auto output_count = session.GetOutputCount();

    std::vector<std::string> input_name_storage;
    input_name_storage.reserve(input_count);

    Ort::IoBinding io_binding(session);

    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_name_storage.emplace_back(name.get());

        const auto it = tensor_store.find(input_name_storage.back());
        if (it == tensor_store.end()) {
            throw std::runtime_error(
                "Missing tensor required by segment " +
                std::to_string(segment.manifest.id) + ": " +
                input_name_storage.back());
        }

        // Bind the existing OrtValue without copying or transferring ownership.
        // ORT performs any required device transfer when the selected EP needs it.
        io_binding.BindInput(input_name_storage.back().c_str(), it->second);
    }

    std::vector<std::string> output_name_storage;
    output_name_storage.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_name_storage.emplace_back(name.get());
    }

    // Let ORT allocate outputs in the selected execution resource. This is
    // the first step toward keeping segment boundaries device-resident and
    // avoiding the implicit CPU output copy of ordinary Session::Run().
    if (resource == RuntimeResource::CUDA) {
        const Ort::MemoryInfo cuda_memory(
            "Cuda",
            OrtDeviceAllocator,
            0,
            OrtMemTypeDefault);
        for (const auto& name : output_name_storage) {
            io_binding.BindOutput(name.c_str(), cuda_memory);
        }
    } else {
        const auto cpu_memory =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        for (const auto& name : output_name_storage) {
            io_binding.BindOutput(name.c_str(), cpu_memory);
        }
    }

    const auto start = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr}, io_binding);
    io_binding.SynchronizeOutputs();
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    auto outputs = io_binding.GetOutputValues(allocator);
    if (outputs.size() != output_name_storage.size()) {
        throw std::runtime_error(
            "Segment " + std::to_string(segment.manifest.id) +
            " returned an unexpected number of outputs");
    }

    // GetOutputValues() materializes Ort::Value handles from the bound
    // outputs. Detach the bound-output references before the returned
    // Ort::Values are destroyed; otherwise the IoBinding and tensor_store
    // can end up releasing the same OrtValue twice.
    io_binding.ClearBoundOutputs();

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        tensor_store.insert_or_assign(
            output_name_storage[i],
            std::move(outputs[i]));
    }

    std::cout << "    segment " << segment.manifest.id
              << " [" << resource_name(resource) << "] "
              << elapsed_ms << " ms\n";
}
}  // namespace

struct FlexOnRuntimeImpl {
    std::unique_ptr<RuntimeState> state;
};

FlexOnRuntime::FlexOnRuntime()
    : impl_(std::make_unique<FlexOnRuntimeImpl>()) {}

FlexOnRuntime::~FlexOnRuntime() = default;
FlexOnRuntime::FlexOnRuntime(FlexOnRuntime&&) noexcept = default;
FlexOnRuntime& FlexOnRuntime::operator=(FlexOnRuntime&&) noexcept = default;

void FlexOnRuntime::load(const std::filesystem::path& artifact_directory) {
    auto artifact = load_manifest(artifact_directory);

    auto state = std::make_unique<RuntimeState>();
    state->env = &ort_env();
    state->artifact = std::move(artifact);
    state->levels.resize(state->artifact.levels.size());

    for (std::size_t level_index = 0;
         level_index < state->artifact.levels.size(); ++level_index) {
        const auto& level = state->artifact.levels[level_index];
        auto& runtime_level = state->levels[level_index];
        runtime_level.reserve(level.segments.size());

        for (const auto& manifest_segment : level.segments) {
            SegmentRuntime runtime_segment;
            runtime_segment.manifest = manifest_segment;

            // Pre-create every resource variant supported by the offline
            // artifact. No session construction occurs in run().
            if (manifest_segment.cpu_supported) {
                runtime_segment.sessions.push_back(
                    PreparedSession{
                        RuntimeResource::CPU,
                        create_session(manifest_segment.model_path,
                                       RuntimeResource::CPU)});
            }
            if (manifest_segment.cuda_supported) {
                runtime_segment.sessions.push_back(
                    PreparedSession{
                        RuntimeResource::CUDA,
                        create_session(manifest_segment.model_path,
                                       RuntimeResource::CUDA)});
            }

            runtime_level.push_back(std::move(runtime_segment));
        }
    }

    impl_->state = std::move(state);

    std::cout << "[online] artifact loaded: " << artifact_directory << '\n'
              << "[online] model: " << impl_->state->artifact.model_name << '\n'
              << "[online] levels: " << impl_->state->levels.size() << '\n';
    for (std::size_t i = 0; i < impl_->state->levels.size(); ++i) {
        std::cout << "[online] level " << i << ": "
                  << impl_->state->levels[i].size() << " segments prepared\n";
    }
}

void FlexOnRuntime::run(const RunOptions& options) {
    if (!impl_->state) {
        throw std::runtime_error("FlexOnRuntime::load() must be called before run()");
    }
    if (options.iterations == 0) {
        throw std::invalid_argument("run iterations must be greater than zero");
    }
    if (options.level >= impl_->state->levels.size()) {
        throw std::out_of_range(
            "Requested level is outside the offline artifact");
    }

    auto& level = impl_->state->levels[options.level];
    if (level.empty()) {
        throw std::runtime_error("Requested level contains no segments");
    }

    // The first segment/session defines the model's external input tensors.
    // These are regenerated for each inference iteration; all intermediate
    // tensors are retained in the tensor store until consumed by later
    // segments, which also handles non-adjacent graph dependencies.
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        std::unordered_map<std::string, Ort::Value> tensor_store;

        auto first_resource = choose_resource(level.front(), options.resource);
        auto* first_prepared = find_session(level.front(), first_resource);
        if (!first_prepared || !first_prepared->session) {
            throw std::runtime_error("First segment session is not prepared");
        }

        add_initial_inputs(
            tensor_store,
            level.front().manifest.input_names,
            *first_prepared->session,
            impl_->state->allocator);

        std::cout << "[online] iteration " << (iteration + 1)
                  << "/" << options.iterations
                  << " level=" << options.level << '\n';

        for (auto& segment : level) {
            execute_segment(
                segment,
                options.resource,
                tensor_store,
                impl_->state->allocator);
        }

            for (const auto& output_name : level.back().manifest.output_names) {
            if (tensor_store.find(output_name) == tensor_store.end()) {
                throw std::runtime_error(
                    "Final level output tensor was not produced: " + output_name);
            }
        }
    }
}

bool FlexOnRuntime::loaded() const noexcept {
    return impl_ && static_cast<bool>(impl_->state);
}

std::uint32_t FlexOnRuntime::level_count() const {
    if (!impl_->state) return 0;
    return static_cast<std::uint32_t>(impl_->state->levels.size());
}

std::string FlexOnRuntime::model_name() const {
    if (!impl_->state) return {};
    return impl_->state->artifact.model_name;
}

}  // namespace flexon::online
