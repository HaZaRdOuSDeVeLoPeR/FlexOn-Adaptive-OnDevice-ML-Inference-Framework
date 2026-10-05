#include <stdexcept>

#include <flexon/offline/config/offline_config.hpp>
#include <yaml-cpp/yaml.h>

namespace flexon::offline::config {

namespace {

template <typename T>
T optional(const YAML::Node& node, const char* key, T value) {
    if (node[key]) {
        return node[key].as<T>();
    }
    return value;
}

}  // namespace

OfflineConfig load(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Offline config not found: " + path.string());
    }

    const auto root = YAML::LoadFile(path.string());
    OfflineConfig config;

    const auto profiling = root["profiling"];
    const auto segmentation = root["segmentation"];
    const auto validation = root["validation"];
    const auto provider = root["provider_partition"];
    const auto resources = root["resources"];

    if (profiling) {
        config.warmup_iterations =
            optional(profiling, "warmup_iterations", config.warmup_iterations);
        config.measurement_iterations =
            optional(profiling, "measurement_iterations",
                     config.measurement_iterations);
        config.percentile =
            optional(profiling, "percentile", config.percentile);
    }

    if (segmentation) {
        config.max_levels =
            optional(segmentation, "max_levels", config.max_levels);
        config.average_segment_cost_threshold_ms =
            optional(segmentation, "average_segment_cost_threshold_ms",
                     config.average_segment_cost_threshold_ms);
        config.split_policy =
            optional(segmentation, "split_policy", config.split_policy);
        config.initial_partition =
            optional(segmentation, "initial_partition",
                     config.initial_partition);
    }

    if (validation) {
        config.numerical_tolerance =
            optional(validation, "numerical_tolerance",
                     config.numerical_tolerance);
    }

    if (root["artifact"]) {
        config.artifact_root =
            optional(root["artifact"], "root", config.artifact_root);
    }

    if (provider && provider["cuda_fallback_ops"]) {
        for (const auto& op : provider["cuda_fallback_ops"]) {
            config.cuda_fallback_ops.push_back(op.as<std::string>());
        }
    }

    if (resources) {
        config.cpu_enabled =
            optional(resources, "cpu_enabled", config.cpu_enabled);
        config.cuda_enabled =
            optional(resources, "cuda_enabled", config.cuda_enabled);
        config.cuda_device_id =
            optional(resources, "cuda_device_id", config.cuda_device_id);
    }

    if (config.warmup_iterations == 0 ||
        config.measurement_iterations == 0 ||
        config.percentile <= 0.0 ||
        config.percentile > 100.0 ||
        config.max_levels == 0) {
        throw std::invalid_argument("Invalid offline profiling/segmentation configuration");
    }

    return config;
}

}  // namespace flexon::offline::config
