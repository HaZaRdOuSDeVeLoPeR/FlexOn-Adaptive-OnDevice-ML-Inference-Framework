#include <flexon/online/artifact/manifest_loader.hpp>

namespace flexon::online::manifest{

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
                    const double max_degradation_ratio =
                        cost["max_degradation_ratio"]
                            ? cost["max_degradation_ratio"].as<double>()
                            : 1.0;
                    if (resource == "cpu") {
                        segment.cpu_supported = true;
                        segment.cpu_mean_ms = mean;
                        segment.cpu_max_degradation_ratio =
                            std::max(1.0, max_degradation_ratio);
                    } else if (resource == "cuda") {
                        segment.cuda_supported = true;
                        segment.cuda_mean_ms = mean;
                        segment.cuda_max_degradation_ratio =
                            std::max(1.0, max_degradation_ratio);
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

} // namespace flexon::online::artifact