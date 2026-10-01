#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnx/onnx_pb.h>

namespace flexon::offline::config {

/**
 * Runtime information used to turn an ONNX graph into a concrete,
 * reproducible offline profiling workload.
 */
struct ModelConfig {
    std::uint32_t batch_size{1};
    bool static_shapes_only{true};
    std::int64_t dynamic_dimension_default{0};
    std::unordered_map<std::string, std::vector<std::int64_t>> input_shapes;
};

/**
 * One model entry from models.yaml.
 */
struct ModelSpec {
    std::string name;
    std::filesystem::path path;
};

/**
 * Complete models.yaml configuration.
 *
 * The model catalog and runtime policy live together in models.yaml,
 * while offline profiling/segmentation policy remains in offline.yaml.
 */
struct ModelsConfig {
    std::vector<ModelSpec> models;
    ModelConfig runtime;
};

/**
 * Load the complete model catalog and runtime configuration.
 */
ModelsConfig load_models_config(const std::filesystem::path& path);

/**
 * Return a profiling-ready copy of the model with concrete graph-input
 * shapes. The source model is never modified.
 */
onnx::ModelProto resolve_profiling_shapes(
    const onnx::ModelProto& source,
    const ModelConfig& config);

}  // namespace flexon::offline::config