#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <cstdint>

#include <yaml-cpp/yaml.h>

namespace flexon::online::manifest {

struct SegmentManifest {
    std::uint32_t id{0};
    std::filesystem::path model_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    bool cpu_supported{false};
    bool cuda_supported{false};
    double cpu_mean_ms{std::numeric_limits<double>::infinity()};
    double cuda_mean_ms{std::numeric_limits<double>::infinity()};
    double cpu_max_degradation_ratio{1.0};
    double cuda_max_degradation_ratio{1.0};
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

ArtifactManifest load_manifest(const std::filesystem::path& directory);
bool is_supported_status(const YAML::Node& cost);

} // namespace flexon::online::manifest