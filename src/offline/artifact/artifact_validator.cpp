#include <stdexcept>

#include <flexon/offline/artifact/artifact_validator.hpp>
#include <flexon/offline/model/onnx_model_loader.hpp>
#include <yaml-cpp/yaml.h>

namespace flexon::offline::artifact {

void ArtifactValidator::validate(
    const std::filesystem::path& directory) {

    const auto manifest = directory / "manifest.yaml";
    if (!std::filesystem::exists(manifest)) {
        throw std::runtime_error(
            "Offline artifact manifest not found: " + manifest.string());
    }

    const auto root = YAML::LoadFile(manifest.string());

    if (!root["format_version"] ||
        root["format_version"].as<int>() != 2) {
        throw std::runtime_error("Unsupported offline artifact format");
    }

    if (!root["model"] || !root["levels"] ||
        !root["operator_capabilities"]) {
        throw std::runtime_error(
            "Offline artifact is missing model, operator capabilities, or levels");
    }

    for (const auto& op : root["operator_capabilities"]) {
        if (!op["graph_index"] || !op["name"] || !op["op_type"] ||
            !op["resources"]) {
            throw std::runtime_error(
                "Operator capability entry is incomplete");
        }
        for (const auto& resource : op["resources"]) {
            if (!resource["resource"] || !resource["status"] ||
                !resource["mean_ms"] || !resource["percentile_ms"]) {
                throw std::runtime_error(
                    "Operator resource capability entry is incomplete");
            }
        }
    }

    for (const auto& level : root["levels"]) {
        if (!level["segments"] || !level["segment_count"] ||
            !level["average_cost_ms"]) {
            throw std::runtime_error(
                "Artifact level is missing segmentation statistics");
        }

        if (level["segment_count"].as<std::size_t>() !=
            level["segments"].size()) {
            throw std::runtime_error(
                "Artifact level segment_count does not match segments");
        }

        for (const auto& segment : level["segments"]) {
            if (!segment["model"] || !segment["costs"]) {
                throw std::runtime_error(
                    "Artifact segment is missing model path or costs");
            }

            for (const auto& cost : segment["costs"]) {
                if (!cost["resource"] || !cost["status"] ||
                    !cost["mean_ms"] || !cost["percentile_ms"] ||
                    !cost["max_degradation_ratio"]) {
                    throw std::runtime_error(
                        "Artifact segment resource cost is incomplete");
                }
            }

            const auto model =
                directory / segment["model"].as<std::string>();

            if (!std::filesystem::exists(model)) {
                throw std::runtime_error(
                    "Artifact segment model is missing: " +
                    model.string());
            }

            if (std::filesystem::file_size(model) == 0) {
                throw std::runtime_error(
                    "Artifact segment model is empty: " +
                    model.string());
            }

            // Parse every generated model so a successful artifact write
            // cannot hide a corrupt or incomplete ONNX segment.
            (void)model::OnnxModelLoader::load(model);
        }
    }
}

}  // namespace flexon::offline::artifact
