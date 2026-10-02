#include <flexon/online/runtime_engine.hpp>
#include <flexon/online/scheduler.hpp>

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <future>
#include <random>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <mutex>
#include <thread>

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

    Ort::Env env{
        ORT_LOGGING_LEVEL_WARNING,
        "FlexOnOnline"
    };

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::vector<SegmentRuntime>> levels;
};

std::unique_ptr<Ort::Session> create_session(
    Ort::Env& env,
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
            env, model_path.c_str(), options);
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


SchedulerSegmentCosts scheduler_costs(const SegmentManifest& segment) {
    return SchedulerSegmentCosts{
        SchedulerResourceCost{
            segment.cpu_supported,
            segment.cpu_mean_ms},
        SchedulerResourceCost{
            segment.cuda_supported,
            segment.cuda_mean_ms}};
}

RuntimeResource choose_concrete_resource(
    const SegmentRuntime& segment,
    RuntimeResource requested) {
    if (requested == RuntimeResource::Auto) {
        throw std::invalid_argument(
            "Auto resource must be resolved by OnlineScheduler before "
            "segment execution");
    }
    return choose_resource(segment, requested);
}

struct SegmentExecutionStats {
    RuntimeResource resource{RuntimeResource::CPU};
    double elapsed_ms{0.0};
    double boundary_copy_ms{0.0};
};

void check_cuda(cudaError_t status, const char* operation);

struct BufferKey {
    RuntimeResource resource{RuntimeResource::CPU};
    std::size_t bytes{0};
    ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    std::vector<std::int64_t> shape;

    bool operator==(const BufferKey& other) const noexcept {
        return resource == other.resource &&
               bytes == other.bytes &&
               type == other.type &&
               shape == other.shape;
    }
};

struct BufferKeyHash {
    std::size_t operator()(const BufferKey& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key.resource);
        hash ^= key.bytes + static_cast<std::size_t>(0x9e3779b9) +
                (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.type) +
                static_cast<std::size_t>(0x9e3779b9) +
                (hash << 6) + (hash >> 2);
        for (const auto dim : key.shape) {
            const auto value = static_cast<std::size_t>(dim);
            hash ^= value + static_cast<std::size_t>(0x9e3779b9) +
                    (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct BufferBlock {
    RuntimeResource resource{RuntimeResource::CPU};
    void* data{nullptr};
    std::size_t bytes{0};
    BufferKey key;
};

class BufferArena;

struct BufferLease {
    BufferArena* arena{nullptr};
    BufferBlock* block{nullptr};

    ~BufferLease();

    BufferLease(const BufferLease&) = delete;
    BufferLease& operator=(const BufferLease&) = delete;
};

class BufferArena {
public:
    ~BufferArena() {
        // All RuntimeTensor leases are destroyed before the arena because the
        // arena is declared before the per-run tensor store.
        for (auto& block : blocks_) {
            if (block->data == nullptr) continue;
            if (block->resource == RuntimeResource::CUDA) {
                cudaFree(block->data);
            } else {
                std::free(block->data);
            }
        }
    }

    std::shared_ptr<BufferLease> acquire(const BufferKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& free_list = free_blocks_[key];
        if (!free_list.empty()) {
            auto* block = free_list.back();
            free_list.pop_back();
            in_use_.insert(block);
            ++reuse_count_;
            update_peak_live_bytes();
            return std::shared_ptr<BufferLease>(
                new BufferLease{this, block});
        }

        auto block = std::make_unique<BufferBlock>();
        block->resource = key.resource;
        block->bytes = key.bytes;
        block->key = key;

        if (key.bytes != 0) {
            if (key.resource == RuntimeResource::CUDA) {
                check_cuda(cudaMalloc(&block->data, key.bytes), "cudaMalloc");
            } else {
                block->data = std::malloc(key.bytes);
                if (block->data == nullptr) {
                    throw std::bad_alloc();
                }
            }
        }

        auto* raw = block.get();
        blocks_.push_back(std::move(block));
        in_use_.insert(raw);
        ++allocation_count_;
        allocated_bytes_ += key.bytes;
        update_peak_live_bytes();
        return std::shared_ptr<BufferLease>(new BufferLease{this, raw});
    }

    std::size_t allocation_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocation_count_;
    }
    std::size_t reuse_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return reuse_count_;
    }
    std::size_t allocated_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return allocated_bytes_;
    }
    std::size_t peak_live_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_live_bytes_;
    }

    std::size_t live_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return live_bytes_unlocked();
    }

private:
    std::size_t live_bytes_unlocked() const noexcept {
        std::size_t total = 0;
        for (const auto& block : blocks_) {
            if (in_use_.find(block.get()) != in_use_.end()) {
                total += block->bytes;
            }
        }
        return total;
    }

    friend struct BufferLease;

    void release(BufferBlock* block) {
        if (block == nullptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        free_blocks_[block->key].push_back(block);
        in_use_.erase(block);
    }

    void update_peak_live_bytes() noexcept {
        peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_unlocked());
    }

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<BufferBlock>> blocks_;
    std::unordered_map<BufferKey, std::vector<BufferBlock*>, BufferKeyHash>
        free_blocks_;
    std::unordered_set<BufferBlock*> in_use_;
    std::size_t allocation_count_{0};
    std::size_t reuse_count_{0};
    std::size_t allocated_bytes_{0};
    std::size_t peak_live_bytes_{0};
};

BufferLease::~BufferLease() {
    if (arena != nullptr && block != nullptr) {
        arena->release(block);
    }
}

struct RuntimeTensor {
    // Declared before value so destruction is value first, lease second.
    // This guarantees that an externally-backed Ort::Value is gone before
    // its arena allocation becomes reusable.
    std::shared_ptr<BufferLease> lease;
    Ort::Value value{nullptr};
};

BufferKey buffer_key(
    RuntimeResource resource,
    const std::vector<std::int64_t>& shape,
    ONNXTensorElementDataType type) {
    const auto count = element_count(shape);
    const auto checked_bytes = [count](std::size_t element_bytes) {
        if (count != 0 &&
            element_bytes > std::numeric_limits<std::size_t>::max() / count) {
            throw std::runtime_error("Tensor byte size overflows size_t");
        }
        return count * element_bytes;
    };
    std::size_t bytes = 0;
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: bytes = checked_bytes(sizeof(float)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: bytes = checked_bytes(sizeof(double)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: bytes = checked_bytes(sizeof(std::int64_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: bytes = checked_bytes(sizeof(std::int32_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: bytes = checked_bytes(sizeof(std::uint8_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: bytes = checked_bytes(sizeof(std::int8_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: bytes = checked_bytes(sizeof(std::uint16_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: bytes = checked_bytes(sizeof(std::int16_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: bytes = checked_bytes(sizeof(std::uint32_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: bytes = checked_bytes(sizeof(std::uint64_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: bytes = checked_bytes(sizeof(bool)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: bytes = checked_bytes(2); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: bytes = checked_bytes(2); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: bytes = checked_bytes(8); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: bytes = checked_bytes(16); break;
        default:
            throw std::runtime_error(
                "Unsupported tensor type for buffer arena: " +
                std::to_string(static_cast<int>(type)));
    }
    return BufferKey{resource, bytes, type, shape};
}

void add_initial_inputs(
    std::unordered_map<std::string, RuntimeTensor>& tensor_store,
    const std::vector<std::string>& input_names,
    Ort::Session& first_session,
    Ort::AllocatorWithDefaultOptions& allocator) {

    const auto count = first_session.GetInputCount();
    for (std::size_t i = 0; i < count; ++i) {
        auto name = first_session.GetInputNameAllocated(i, allocator);
        const std::string tensor_name(name.get());
        if (tensor_store.find(tensor_name) != tensor_store.end()) continue;
        auto type_info = first_session.GetInputTypeInfo(i);
        RuntimeTensor tensor;
        tensor.value = make_input_value(
            allocator, type_info, static_cast<std::uint32_t>(i + 1));
        tensor_store.emplace(tensor_name, std::move(tensor));
    }

    (void)input_names;
}

RuntimeResource tensor_resource(const Ort::Value& value) {
    const auto memory_info = value.GetTensorMemoryInfo();
    auto name = memory_info.GetAllocatorName();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (name.find("cuda") != std::string::npos) {
        return RuntimeResource::CUDA;
    }
    if (name.find("cpu") != std::string::npos) {
        return RuntimeResource::CPU;
    }

    throw std::runtime_error(
        "Unsupported tensor memory allocator: " + name);
}

Ort::MemoryInfo memory_info_for_resource(RuntimeResource resource) {
    if (resource == RuntimeResource::CPU) {
        return Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemTypeDefault);
    }

    if (resource == RuntimeResource::CUDA) {
        return Ort::MemoryInfo(
            "Cuda",
            OrtAllocatorType::OrtDeviceAllocator,
            0,
            OrtMemTypeDefault);
    }

    throw std::invalid_argument(
        "A concrete CPU/CUDA resource is required for tensor placement");
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

struct SegmentExecutionResult {
    SegmentExecutionStats stats;
    std::chrono::steady_clock::time_point completed_at{};
    std::vector<std::pair<std::string, RuntimeTensor>> outputs;
};

SegmentExecutionResult execute_segment_once(
    SegmentRuntime& segment,
    RuntimeResource requested_resource,
    std::unordered_map<std::string, RuntimeTensor>& tensor_store,
    BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::shared_ptr<std::promise<void>>& inputs_bound = nullptr) {

    const auto resource = choose_concrete_resource(segment, requested_resource);
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
    std::vector<std::string> output_name_storage;
    output_name_storage.reserve(output_count);
    double boundary_copy_ms = 0.0;

    std::vector<std::shared_ptr<BufferLease>> transferred_inputs;
    transferred_inputs.reserve(input_count);
    std::vector<std::shared_ptr<BufferLease>> output_leases;
    output_leases.reserve(output_count);
    Ort::IoBinding io_binding(session);

    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_name_storage.emplace_back(name.get());
        const auto& input_name = input_name_storage.back();
        const auto it = tensor_store.find(input_name);

        if (it == tensor_store.end()) {
            throw std::runtime_error(
                "Missing tensor required by segment " +
                std::to_string(segment.manifest.id) + ": " + input_name);
        }

        const auto source_resource = tensor_resource(it->second.value);
        if (source_resource == resource) {
            io_binding.BindInput(input_name.c_str(), it->second.value);
        } else {
            const auto type_info = it->second.value.GetTensorTypeAndShapeInfo();
            const auto shape = concrete_shape(type_info.GetShape());
            const auto type = type_info.GetElementType();
            auto lease = arena.acquire(buffer_key(resource, shape, type));
            const auto bytes = lease->block->bytes;

            const auto memory_info = memory_info_for_resource(resource);
            Ort::Value transferred = Ort::Value::CreateTensor(
                memory_info,
                lease->block->data,
                bytes,
                shape.data(),
                shape.size(),
                type);

            const auto start = std::chrono::steady_clock::now();
            if (resource == RuntimeResource::CUDA) {
                check_cuda(
                    cudaMemcpy(
                        lease->block->data,
                        it->second.value.GetTensorRawData(),
                        bytes,
                        cudaMemcpyHostToDevice),
                    "cudaMemcpyHostToDevice");
            } else {
                check_cuda(
                    cudaMemcpy(
                        lease->block->data,
                        it->second.value.GetTensorRawData(),
                        bytes,
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpyDeviceToHost");
            }
            const auto end = std::chrono::steady_clock::now();
            boundary_copy_ms += std::chrono::duration<double, std::milli>(
                end - start).count();

            io_binding.BindInput(input_name.c_str(), transferred);
            transferred_inputs.push_back(std::move(lease));
        }
    }

    // Recovery executions may continue after the winning execution returns.
    // Signal once this worker has finished reading the shared tensor store so
    // the caller can safely update tensor liveness/output ownership.
    if (inputs_bound) {
        inputs_bound->set_value();
    }

    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_name_storage.emplace_back(name.get());
    }

    const auto memory_info = memory_info_for_resource(resource);

    for (std::size_t i = 0; i < output_count; ++i) {
        auto type_info = session.GetOutputTypeInfo(i);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const auto raw_shape = tensor_info.GetShape();
        bool concrete = true;
        for (const auto dim : raw_shape) {
            if (dim < 0) {
                concrete = false;
                break;
            }
        }

        if (!concrete) {
            io_binding.BindOutput(output_name_storage[i].c_str(), memory_info);
            output_leases.push_back(nullptr);
            continue;
        }

        const auto shape = concrete_shape(raw_shape);
        const auto type = tensor_info.GetElementType();
        auto lease = arena.acquire(buffer_key(resource, shape, type));
        const auto bytes = lease->block->bytes;
        Ort::Value output = Ort::Value::CreateTensor(
            memory_info,
            lease->block->data,
            bytes,
            shape.data(),
            shape.size(),
            type);
        io_binding.BindOutput(output_name_storage[i].c_str(), output);
        output_leases.push_back(std::move(lease));
    }

    const auto start = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr}, io_binding);
    io_binding.SynchronizeOutputs();
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    auto outputs = io_binding.GetOutputValues();
    if (outputs.size() != output_name_storage.size()) {
        throw std::runtime_error(
            "Segment " + std::to_string(segment.manifest.id) +
            " returned an unexpected number of outputs");
    }

    SegmentExecutionResult result;
    result.stats = SegmentExecutionStats{
        resource,
        elapsed_ms,
        boundary_copy_ms};
    result.completed_at = end;
    result.outputs.reserve(outputs.size());

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        RuntimeTensor tensor;
        tensor.lease = std::move(output_leases[i]);
        tensor.value = std::move(outputs[i]);
        result.outputs.emplace_back(
            output_name_storage[i],
            std::move(tensor));
    }

    io_binding.ClearBoundOutputs();
    io_binding.ClearBoundInputs();

    if (resource == RuntimeResource::CUDA) {
        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize before releasing transferred inputs");
    }

    return result;
}

void commit_segment_outputs(
    SegmentExecutionResult& result,
    std::unordered_map<std::string, RuntimeTensor>& tensor_store) {

    for (auto& output : result.outputs) {
        auto existing = tensor_store.find(output.first);
        if (existing != tensor_store.end()) {
            existing->second.value = Ort::Value{nullptr};
            existing->second.lease.reset();
            existing->second = std::move(output.second);
        } else {
            tensor_store.emplace(
                output.first,
                std::move(output.second));
        }
    }
}

void print_segment_stats(
    const SegmentManifest& manifest,
    const SegmentExecutionStats& stats,
    const BufferArena& arena) {

    std::cout << "    segment " << manifest.id
              << " [" << resource_name(stats.resource) << "] "
              << stats.elapsed_ms << " ms"
              << " | boundary_copy=" << stats.boundary_copy_ms << " ms"
              << " | arena_allocations=" << arena.allocation_count()
              << " | arena_reuses=" << arena.reuse_count()
              << " | arena_bytes=" << arena.allocated_bytes()
              << " | arena_peak_live_bytes=" << arena.peak_live_bytes() << "\n";
}

SegmentExecutionStats execute_segment(
    SegmentRuntime& segment,
    RuntimeResource requested_resource,
    std::unordered_map<std::string, RuntimeTensor>& tensor_store,
    BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator) {

    auto result = execute_segment_once(
        segment,
        requested_resource,
        tensor_store,
        arena,
        allocator);
    const auto stats = result.stats;
    commit_segment_outputs(result, tensor_store);
    print_segment_stats(segment.manifest, stats, arena);
    return stats;
}

SegmentExecutionStats execute_segment_with_recovery(
    SegmentRuntime& segment,
    RuntimeResource primary_resource,
    const SchedulerSegmentCosts& costs,
    OnlineScheduler& scheduler,
    const SchedulerConfig& scheduler_config,
    std::unordered_map<std::string, RuntimeTensor>& tensor_store,
    BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::future<SegmentExecutionResult>>& background_recoveries) {

    const auto primary_ready = std::make_shared<std::promise<void>>();
    auto primary_ready_future = primary_ready->get_future();

    auto launch = [&](RuntimeResource resource,
                      const std::shared_ptr<std::promise<void>>& ready) {
        return std::async(
            std::launch::async,
            [&segment, resource, &tensor_store, &arena, &allocator, ready]() {
                try {
                    return execute_segment_once(
                        segment,
                        resource,
                        tensor_store,
                        arena,
                        allocator,
                        ready);
                } catch (...) {
                    try {
                        ready->set_exception(std::current_exception());
                    } catch (...) {
                    }
                    throw;
                }
            });
    };

    auto primary_future = launch(primary_resource, primary_ready);

    // Wait until the primary worker has finished reading its input tensors.
    // From this point onward a speculative recovery worker can safely continue
    // without racing tensor-store liveness updates in the caller.
    primary_ready_future.get();

    bool recovery_triggered = false;
    RuntimeResource recovery_resource = RuntimeResource::CPU;
    std::future<SegmentExecutionResult> recovery_future;

    const auto segment_start = std::chrono::steady_clock::now();
    while (primary_future.wait_for(std::chrono::milliseconds(0)) !=
           std::future_status::ready) {

        const auto now = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(now - segment_start).count();

        if (!recovery_triggered && scheduler_config.recovery_enabled) {
            const auto alternative =
                scheduler.select_recovery_resource(costs, primary_resource);

            if (scheduler.should_trigger_recovery(
                    elapsed_ms, alternative.score)) {

                // Avoid launching a speculative execution if the primary
                // completed between the polling check and this decision.
                if (primary_future.wait_for(std::chrono::milliseconds(0)) ==
                    std::future_status::ready) {
                    break;
                }

                recovery_triggered = true;
                recovery_resource = alternative.resource;

                std::cout
                    << "[recovery] segment " << segment.manifest.id
                    << " primary=" << resource_name(primary_resource)
                    << " elapsed_ms=" << elapsed_ms
                    << " alternative=" << resource_name(recovery_resource)
                    << " alternative_score=" << alternative.score
                    << " gamma=" << scheduler_config.gamma << '\n';

                const auto recovery_ready =
                    std::make_shared<std::promise<void>>();
                auto recovery_ready_future = recovery_ready->get_future();
                recovery_future = launch(recovery_resource, recovery_ready);
                recovery_ready_future.get();

                // The primary may have completed while the alternative was
                // being prepared. Prefer whichever result is observed first.
                while (recovery_future.wait_for(std::chrono::milliseconds(0)) !=
                           std::future_status::ready &&
                       primary_future.wait_for(std::chrono::milliseconds(0)) !=
                           std::future_status::ready) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                const bool recovery_ready_now =
                    recovery_future.wait_for(std::chrono::milliseconds(0)) ==
                    std::future_status::ready;
                const bool primary_ready_now =
                    primary_future.wait_for(std::chrono::milliseconds(0)) ==
                    std::future_status::ready;

                if (recovery_ready_now && primary_ready_now) {
                    std::optional<SegmentExecutionResult> recovery_result;
                    std::optional<SegmentExecutionResult> primary_result;

                    try {
                        recovery_result.emplace(recovery_future.get());
                    } catch (...) {
                    }
                    try {
                        primary_result.emplace(primary_future.get());
                    } catch (...) {
                    }

                    if (recovery_result && primary_result) {
                        if (recovery_result->completed_at <
                            primary_result->completed_at) {
                            recovery_result->stats.elapsed_ms =
                                std::chrono::duration<double, std::milli>(
                                    recovery_result->completed_at - segment_start).count();
                            commit_segment_outputs(*recovery_result, tensor_store);
                            print_segment_stats(
                                segment.manifest, recovery_result->stats, arena);
                            std::cout << "[recovery] alternative result won\n";
                            return recovery_result->stats;
                        }

                        primary_result->stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                primary_result->completed_at - segment_start).count();
                        commit_segment_outputs(*primary_result, tensor_store);
                        print_segment_stats(
                            segment.manifest, primary_result->stats, arena);
                        std::cout << "[recovery] primary result won\n";
                        return primary_result->stats;
                    }

                    if (recovery_result) {
                        recovery_result->stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                recovery_result->completed_at - segment_start).count();
                        commit_segment_outputs(*recovery_result, tensor_store);
                        print_segment_stats(
                            segment.manifest, recovery_result->stats, arena);
                        std::cout << "[recovery] alternative result won\n";
                        return recovery_result->stats;
                    }

                    if (primary_result) {
                        primary_result->stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                primary_result->completed_at - segment_start).count();
                        commit_segment_outputs(*primary_result, tensor_store);
                        print_segment_stats(
                            segment.manifest, primary_result->stats, arena);
                        std::cout << "[recovery] primary result won\n";
                        return primary_result->stats;
                    }

                    throw std::runtime_error(
                        "Both primary and recovery executions failed");
                }

                if (recovery_ready_now) {
                    try {
                        auto result = recovery_future.get();
                        result.stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                result.completed_at - segment_start).count();
                        background_recoveries.push_back(std::move(primary_future));
                        commit_segment_outputs(result, tensor_store);
                        print_segment_stats(segment.manifest, result.stats, arena);
                        std::cout << "[recovery] alternative result won\n";
                        return result.stats;
                    } catch (...) {
                        // A speculative recovery failure must not invalidate a
                        // successful primary execution.
                    }
                }

                auto result = primary_future.get();
                result.stats.elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        result.completed_at - segment_start).count();
                if (recovery_future.valid()) {
                    background_recoveries.push_back(std::move(recovery_future));
                }
                commit_segment_outputs(result, tensor_store);
                print_segment_stats(segment.manifest, result.stats, arena);
                std::cout << "[recovery] primary result won\n";
                return result.stats;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto result = primary_future.get();
    commit_segment_outputs(result, tensor_store);
    print_segment_stats(segment.manifest, result.stats, arena);
    return result.stats;
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
                        create_session(
                            state->env,
                            manifest_segment.model_path,
                            RuntimeResource::CPU)});
            }
            if (manifest_segment.cuda_supported) {
                runtime_segment.sessions.push_back(
                    PreparedSession{
                        RuntimeResource::CUDA,
                        create_session(
                            state->env,
                            manifest_segment.model_path,
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
            "Requested initial level is outside the offline artifact");
    }

    if (!options.resource_plan.empty()) {
        const auto& initial_level = impl_->state->levels[options.level];
        if (options.resource_plan.size() != initial_level.size()) {
            throw std::invalid_argument(
                "resource plan length must match the selected initial level "
                "segment count");
        }
        if (options.resource == RuntimeResource::Auto) {
            throw std::invalid_argument(
                "--resource-plan cannot be combined with --resource auto");
        }
        for (const auto resource : options.resource_plan) {
            if (resource == RuntimeResource::Auto) {
                throw std::invalid_argument(
                    "resource_plan entries must be concrete cpu/cuda resources; "
                    "use --resource auto for FlexOn dynamic resource selection");
            }
        }
    }

    SchedulerConfig scheduler_config;
    if (!options.scheduler_config_path.empty()) {
        scheduler_config =
            OnlineScheduler::load_config(options.scheduler_config_path);
    } else {
        const std::filesystem::path default_config =
            "config/scheduler.yaml";
        scheduler_config = OnlineScheduler::load_config(default_config);
    }

    OnlineScheduler scheduler(scheduler_config);
    const bool dynamic_resource_selection =
        options.resource == RuntimeResource::Auto;

    if (dynamic_resource_selection) {
        scheduler.start();
    }

    // The arena persists across all inference periods and levels so buffers
    // released by one inference can be reused by another level/period.
    BufferArena arena;

    std::uint32_t current_level = options.level;
    double period_measured_ms = 0.0;
    double period_expected_ms = 0.0;

    try {
        for (std::uint32_t iteration = 0;
             iteration < options.iterations;
             ++iteration) {

            auto& level = impl_->state->levels[current_level];
            if (level.empty()) {
                throw std::runtime_error(
                    "Selected segmentation level contains no segments");
            }

            std::unordered_map<std::string, std::size_t> future_uses;
            for (const auto& segment : level) {
                for (const auto& input_name : segment.manifest.input_names) {
                    ++future_uses[input_name];
                }
            }

            std::unordered_set<std::string> final_outputs(
                level.back().manifest.output_names.begin(),
                level.back().manifest.output_names.end());

            std::unordered_map<std::string, RuntimeTensor> tensor_store;
            std::vector<std::future<SegmentExecutionResult>> background_recoveries;

            RuntimeResource previous_resource = RuntimeResource::CPU;
            SegmentExecutionStats previous_stats{};
            SegmentManifest* previous_manifest = nullptr;

            // The first segment establishes the initial resource. In Auto
            // mode this uses predicted cost divided by current remaining
            // capacity, corresponding to the resource-selection policy.
            RuntimeResource first_resource = options.resource;
            SchedulerDecision first_decision{};

            if (dynamic_resource_selection) {
                first_decision = scheduler.select_first_resource(
                    scheduler_costs(level.front().manifest));
                first_resource = first_decision.resource;

                std::cout
                    << "[scheduler] iteration " << (iteration + 1)
                    << " initial resource=" << resource_name(first_resource)
                    << " score=" << first_decision.score
                    << " degradation=" << first_decision.degradation
                    << " remaining_capacity="
                    << first_decision.remaining_capacity << '\n';
            }

            auto* first_prepared =
                find_session(level.front(), first_resource);
            if (!first_prepared || !first_prepared->session) {
                throw std::runtime_error(
                    "First segment session is not prepared for selected resource");
            }

            add_initial_inputs(
                tensor_store,
                level.front().manifest.input_names,
                *first_prepared->session,
                impl_->state->allocator);

            std::cout << "[online] iteration " << (iteration + 1)
                      << "/" << options.iterations
                      << " level=" << current_level << '\n';

            for (std::size_t segment_index = 0;
                 segment_index < level.size();
                 ++segment_index) {

                auto& segment = level[segment_index];

                RuntimeResource requested_resource = options.resource;

                if (!options.resource_plan.empty()) {
                    requested_resource =
                        options.resource_plan[segment_index];
                } else if (dynamic_resource_selection &&
                           segment_index == 0) {
                    requested_resource = first_resource;
                } else if (dynamic_resource_selection) {
                    const auto decision = scheduler.select_next_resource(
                        scheduler_costs(*previous_manifest),
                        previous_resource,
                        previous_stats.elapsed_ms,
                        scheduler_costs(segment.manifest));

                    requested_resource = decision.resource;

                    std::cout
                        << "[scheduler] segment " << segment.manifest.id
                        << " selected=" << resource_name(requested_resource)
                        << " score=" << decision.score
                        << " degradation=" << decision.degradation
                        << " remaining_capacity="
                        << decision.remaining_capacity << '\n';
                }

                SegmentExecutionStats stats;
                if (dynamic_resource_selection &&
                    scheduler_config.recovery_enabled) {
                    stats = execute_segment_with_recovery(
                        segment,
                        requested_resource,
                        scheduler_costs(segment.manifest),
                        scheduler,
                        scheduler_config,
                        tensor_store,
                        arena,
                        impl_->state->allocator,
                        background_recoveries);
                } else {
                    stats = execute_segment(
                        segment,
                        requested_resource,
                        tensor_store,
                        arena,
                        impl_->state->allocator);
                }

                const auto selected_expected =
                    stats.resource == RuntimeResource::CPU
                        ? segment.manifest.cpu_mean_ms
                        : segment.manifest.cuda_mean_ms;

                period_measured_ms += stats.elapsed_ms;
                if (std::isfinite(selected_expected)) {
                    period_expected_ms += selected_expected;
                }

                previous_resource = stats.resource;
                previous_stats = stats;
                previous_manifest = &segment.manifest;

                // Release tensors whose final consumer has just completed.
                for (const auto& input_name : segment.manifest.input_names) {
                    auto remaining = future_uses.find(input_name);
                    if (remaining != future_uses.end() &&
                        remaining->second > 0) {
                        --remaining->second;
                        if (remaining->second == 0 &&
                            final_outputs.find(input_name) ==
                                final_outputs.end()) {
                            tensor_store.erase(input_name);
                        }
                    }
                }
            }

            for (auto& recovery : background_recoveries) {
                try {
                    recovery.get();
                } catch (const std::exception& ex) {
                    std::cout << "[recovery] speculative execution failed after "
                              << "winner was selected: " << ex.what() << '\n';
                } catch (...) {
                    std::cout << "[recovery] speculative execution failed after "
                              << "winner was selected\n";
                }
            }
            background_recoveries.clear();

            for (const auto& output_name :
                 level.back().manifest.output_names) {
                if (tensor_store.find(output_name) == tensor_store.end()) {
                    throw std::runtime_error(
                        "Final level output tensor was not produced: " +
                        output_name);
                }
            }

            if (options.adaptive_level) {
                const auto next_level = scheduler.select_next_level(
                    current_level,
                    static_cast<std::uint32_t>(
                        impl_->state->levels.size() - 1),
                    period_measured_ms,
                    period_expected_ms);

                std::cout
                    << "[scheduler] level decision"
                    << " measured_ms=" << period_measured_ms
                    << " expected_ms=" << period_expected_ms
                    << " level=" << current_level
                    << " -> " << next_level << '\n';

                current_level = next_level;
                period_measured_ms = 0.0;
                period_expected_ms = 0.0;
            }
        }
    } catch (...) {
        scheduler.stop();
        throw;
    }

    scheduler.stop();
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
