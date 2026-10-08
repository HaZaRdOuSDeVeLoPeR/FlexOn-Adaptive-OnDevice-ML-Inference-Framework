#include <algorithm>
#include <chrono>
#include <memory>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <flexon/offline/profiling/segment_profiler.hpp>
#include <flexon/offline/model/segment_model_generator.hpp>
#include <onnxruntime_cxx_api.h>

namespace flexon::offline::profiling {

namespace {

class UnsupportedResource final : public std::runtime_error {
public:
    explicit UnsupportedResource(const std::string& message)
        : std::runtime_error(message) {}
};

Ort::Env& ort_env() {
    // CUDA capability probes intentionally create strict sessions that may
    // reject CPU fallback. ORT logs that expected probe result at ERROR, so
    // keep the backend quiet and persist the actual reason in the artifact.
    static Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "FlexOnOffline");
    return env;
}

std::vector<float> make_float_data(std::size_t count) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<float> data(count);
    for (auto& value : data) {
        value = dist(rng);
    }
    return data;
}

std::vector<std::int64_t> make_int64_data(std::size_t count) {
    return std::vector<std::int64_t>(count, 0);
}

core::SegmentCost run_resource(
    const config::OfflineConfig& config,
    const std::filesystem::path& model_path,
    core::Resource resource) {

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);

    if (resource == core::Resource::CUDA) {
        // For offline capability profiling, do not allow ORT to silently
        // route an unsupported CUDA node to the CPU EP. A failed strict CUDA
        // session means this resource is not supported for the model/segment.
        options.AddConfigEntry(
            "session.disable_cpu_ep_fallback",
            "1");

        OrtCUDAProviderOptions cuda{};
        cuda.device_id = config.cuda_device_id;
        options.AppendExecutionProvider_CUDA(cuda);
    }

    std::unique_ptr<Ort::Session> session;
    try {
        session = std::make_unique<Ort::Session>(
            ort_env(), model_path.c_str(), options);
    } catch (const Ort::Exception& ex) {
        if (resource == core::Resource::CUDA) {
            throw UnsupportedResource(
                std::string("CUDA execution-provider capability probe failed: ") +
                ex.what());
        }
        throw;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::Value> inputs;
    std::vector<const char*> input_names;
    std::vector<std::string> input_name_storage;
    std::vector<std::vector<float>> float_buffers;
    std::vector<std::vector<std::int64_t>> int64_buffers;

    const auto input_count = session->GetInputCount();
    inputs.reserve(input_count);
    input_names.reserve(input_count);
    input_name_storage.reserve(input_count);
    float_buffers.reserve(input_count);
    int64_buffers.reserve(input_count);

    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session->GetInputNameAllocated(i, allocator);
        input_name_storage.emplace_back(name.get());
        input_names.push_back(input_name_storage.back().c_str());

        const auto type_info = session->GetInputTypeInfo(i);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const auto shape = tensor_info.GetShape();

        std::size_t element_count = 1;
        for (const auto dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error(
                    "Cannot profile segment input '" +
                    input_name_storage.back() +
                    "': unresolved dynamic dimension. The offline engine must "
                    "resolve model input shapes before segment generation.");
            }
            element_count *= static_cast<std::size_t>(dim);
        }

        const auto type = tensor_info.GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float_buffers.push_back(make_float_data(element_count));
            auto memory = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            inputs.emplace_back(
                Ort::Value::CreateTensor<float>(
                    memory, float_buffers.back().data(),
                    float_buffers.back().size(),
                    shape.data(), shape.size()));
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int64_buffers.push_back(make_int64_data(element_count));
            auto memory = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            inputs.emplace_back(
                Ort::Value::CreateTensor<std::int64_t>(
                    memory, int64_buffers.back().data(),
                    int64_buffers.back().size(),
                    shape.data(), shape.size()));
        } else {
            throw std::runtime_error(
                "Unsupported profiling input element type for " +
                input_name_storage.back());
        }
    }

    std::vector<const char*> output_names;
    std::vector<std::string> output_name_storage;
    const auto output_count = session->GetOutputCount();
    output_names.reserve(output_count);
    output_name_storage.reserve(output_count);

    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        output_name_storage.emplace_back(name.get());
        output_names.push_back(output_name_storage.back().c_str());
    }

    for (std::uint32_t i = 0; i < config.warmup_iterations; ++i) {
        (void)session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), inputs.data(), inputs.size(),
            output_names.data(), output_names.size());
    }

    std::vector<double> samples;
    samples.reserve(config.measurement_iterations);

    for (std::uint32_t i = 0; i < config.measurement_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();

        (void)session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), inputs.data(), inputs.size(),
            output_names.data(), output_names.size());

        const auto end = std::chrono::steady_clock::now();

        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());

    const double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) /
        static_cast<double>(samples.size());

    const double rank =
        (config.percentile / 100.0) *
        static_cast<double>(samples.size() - 1);

    const auto lower = static_cast<std::size_t>(std::floor(rank));
    const auto upper = static_cast<std::size_t>(std::ceil(rank));

    const double percentile =
        lower == upper
            ? samples[lower]
            : samples[lower] +
                  (samples[upper] - samples[lower]) *
                      (rank - static_cast<double>(lower));

    return core::SegmentCost{resource, mean, percentile, core::ResourceSupportStatus::Supported, {}};
}


core::OperatorCost run_operator_resource(
    const config::OfflineConfig& config,
    const std::filesystem::path& model_path,
    core::Resource resource) {

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);

    if (resource == core::Resource::CUDA) {
        // For offline capability profiling, do not allow ORT to silently
        // route an unsupported CUDA node to the CPU EP. A failed strict CUDA
        // session means this resource is not supported for the model/segment.
        options.AddConfigEntry(
            "session.disable_cpu_ep_fallback",
            "1");

        OrtCUDAProviderOptions cuda{};
        cuda.device_id = config.cuda_device_id;
        options.AppendExecutionProvider_CUDA(cuda);
    }

    std::unique_ptr<Ort::Session> session;
    try {
        session = std::make_unique<Ort::Session>(
            ort_env(), model_path.c_str(), options);
    } catch (const Ort::Exception& ex) {
        if (resource == core::Resource::CUDA) {
            throw UnsupportedResource(
                std::string("CUDA execution-provider capability probe failed: ") +
                ex.what());
        }
        throw;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::Value> inputs;
    std::vector<const char*> input_names;
    std::vector<std::string> input_name_storage;
    std::vector<std::vector<float>> float_buffers;
    std::vector<std::vector<std::int64_t>> int64_buffers;

    const auto input_count = session->GetInputCount();
    inputs.reserve(input_count);
    input_names.reserve(input_count);
    input_name_storage.reserve(input_count);
    float_buffers.reserve(input_count);
    int64_buffers.reserve(input_count);

    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session->GetInputNameAllocated(i, allocator);
        input_name_storage.emplace_back(name.get());
        input_names.push_back(input_name_storage.back().c_str());

        const auto type_info = session->GetInputTypeInfo(i);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const auto shape = tensor_info.GetShape();

        std::size_t element_count = 1;
        for (const auto dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error(
                    "Cannot profile operator input '" +
                    input_name_storage.back() +
                    "': unresolved dynamic dimension.");
            }
            element_count *= static_cast<std::size_t>(dim);
        }

        const auto type = tensor_info.GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float_buffers.push_back(make_float_data(element_count));
            auto memory = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            inputs.emplace_back(
                Ort::Value::CreateTensor<float>(
                    memory, float_buffers.back().data(),
                    float_buffers.back().size(),
                    shape.data(), shape.size()));
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int64_buffers.push_back(make_int64_data(element_count));
            auto memory = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            inputs.emplace_back(
                Ort::Value::CreateTensor<std::int64_t>(
                    memory, int64_buffers.back().data(),
                    int64_buffers.back().size(),
                    shape.data(), shape.size()));
        } else {
            throw std::runtime_error(
                "Unsupported profiling input element type for " +
                input_name_storage.back());
        }
    }

    std::vector<const char*> output_names;
    std::vector<std::string> output_name_storage;
    const auto output_count = session->GetOutputCount();
    output_names.reserve(output_count);
    output_name_storage.reserve(output_count);

    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        output_name_storage.emplace_back(name.get());
        output_names.push_back(output_name_storage.back().c_str());
    }

    for (std::uint32_t i = 0; i < config.warmup_iterations; ++i) {
        (void)session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), inputs.data(), inputs.size(),
            output_names.data(), output_names.size());
    }

    std::vector<double> samples;
    samples.reserve(config.measurement_iterations);

    for (std::uint32_t i = 0; i < config.measurement_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();

        (void)session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), inputs.data(), inputs.size(),
            output_names.data(), output_names.size());

        const auto end = std::chrono::steady_clock::now();

        samples.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());

    const double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) /
        static_cast<double>(samples.size());

    const double rank =
        (config.percentile / 100.0) *
        static_cast<double>(samples.size() - 1);

    const auto lower = static_cast<std::size_t>(std::floor(rank));
    const auto upper = static_cast<std::size_t>(std::ceil(rank));

    const double percentile =
        lower == upper
            ? samples[lower]
            : samples[lower] +
                  (samples[upper] - samples[lower]) *
                      (rank - static_cast<double>(lower));

    return core::OperatorCost{resource, mean, percentile, core::ResourceSupportStatus::Supported, {}};
}

}  // namespace

SegmentProfiler::SegmentProfiler(
    const config::OfflineConfig& config,
    const std::filesystem::path& work_directory)
    : config_(config), work_directory_(work_directory) {
    std::filesystem::create_directories(work_directory_);
}

core::SegmentProfile SegmentProfiler::profile(
    const onnx::ModelProto& source,
    const core::GraphInfo& graph,
    const core::SegmentInfo& segment) {

    const auto model =
        model::SegmentModelGenerator::build(source, graph, segment);

    const auto model_path =
        work_directory_ /
        ("level_" + std::to_string(segment.level) +
         "_segment_" + std::to_string(segment.id) + ".onnx");

    model::SegmentModelGenerator::save(model, model_path);

    core::SegmentProfile profile;
    profile.segment = segment;

    if (config_.cpu_enabled) {
        profile.costs.push_back(
            run_resource(config_, model_path, core::Resource::CPU));
    }

    if (config_.cuda_enabled) {
        try {
            profile.costs.push_back(
                run_resource(config_, model_path, core::Resource::CUDA));
        } catch (const UnsupportedResource& ex) {
            profile.costs.push_back({
                core::Resource::CUDA,
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                core::ResourceSupportStatus::Unsupported,
                ex.what()});
        } catch (const std::exception& ex) {
            profile.costs.push_back({
                core::Resource::CUDA,
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                core::ResourceSupportStatus::ProfilingFailed,
                ex.what()});
        }
    }

    return profile;
}

core::SegmentProfile SegmentProfiler::profile_under_contention(
    const onnx::ModelProto& source,
    const core::GraphInfo& graph,
    const core::SegmentInfo& segment) {

    // Reuse the exact profiling path used for ideal measurements. It already
    // performs the configured warmup iterations and then computes mean_ms
    // from the configured measurement iterations. The offline engine compares
    // this contended mean with the corresponding ideal mean and stores only
    // the resulting maximum degradation ratio in the artifact.
    return profile(source, graph, segment);
}

core::OperatorProfile SegmentProfiler::profile_operator(
    const onnx::ModelProto& source,
    const core::GraphInfo& graph,
    const std::uint32_t operator_index) {

    if (operator_index >= graph.operators.size()) {
        throw std::out_of_range("Invalid operator index for profiling");
    }

    const auto& op = graph.operators.at(operator_index);

    core::SegmentInfo probe;
    probe.level = 0;
    probe.id = operator_index;
    probe.operator_indices = {operator_index};
    probe.operator_names = {op.name};

    // Reconstruct the one-operator segment boundaries. This deliberately
    // mirrors the segment boundary semantics used by MultiLevelSegmenter.
    for (const auto& input : op.inputs) {
        const auto producer = graph.producer_operator.find(input);
        if (producer == graph.producer_operator.end() ||
            producer->second != operator_index) {
            if (std::find(probe.input_tensors.begin(),
                          probe.input_tensors.end(), input) ==
                probe.input_tensors.end()) {
                probe.input_tensors.push_back(input);
            }
        }
    }

    for (const auto& output : op.outputs) {
        const auto consumers = graph.consumer_operators.find(output);
        bool outside = false;

        if (consumers != graph.consumer_operators.end()) {
            for (const auto consumer : consumers->second) {
                if (consumer != operator_index) {
                    outside = true;
                    break;
                }
            }
        }

        const bool graph_output =
            std::find(graph.graph_outputs.begin(),
                      graph.graph_outputs.end(), output) !=
            graph.graph_outputs.end();

        if ((outside || graph_output) &&
            std::find(probe.output_tensors.begin(),
                      probe.output_tensors.end(), output) ==
                probe.output_tensors.end()) {
            probe.output_tensors.push_back(output);
        }
    }

    const auto model =
        model::SegmentModelGenerator::build(source, graph, probe);

    const auto model_path =
        work_directory_ /
        ("operator_" + std::to_string(operator_index) + ".onnx");

    model::SegmentModelGenerator::save(model, model_path);

    core::OperatorProfile profile;
    profile.op = op;

    auto run = [&](core::Resource resource) {
        return run_operator_resource(config_, model_path, resource);
    };

    if (config_.cpu_enabled) {
        profile.costs.push_back(run(core::Resource::CPU));
    }

    if (config_.cuda_enabled) {
        try {
            profile.costs.push_back(run(core::Resource::CUDA));
        } catch (const UnsupportedResource& ex) {
            profile.costs.push_back({
                core::Resource::CUDA,
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                core::ResourceSupportStatus::Unsupported,
                ex.what()});
        } catch (const std::exception& ex) {
            profile.costs.push_back({
                core::Resource::CUDA,
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                core::ResourceSupportStatus::ProfilingFailed,
                ex.what()});
        }
    }

    return profile;
}

}  // namespace flexon::offline::profiling
