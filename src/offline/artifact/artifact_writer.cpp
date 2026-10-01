#include <flexon/offline/artifact/artifact_writer.hpp>

#include <flexon/offline/model/segment_model_generator.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace flexon::offline::artifact {

namespace {

const char* resource_name(core::Resource resource) {
    return resource == core::Resource::CPU ? "cpu" : "cuda";
}

const char* status_name(core::ResourceSupportStatus status) {
    switch (status) {
        case core::ResourceSupportStatus::Supported:
            return "supported";
        case core::ResourceSupportStatus::Unsupported:
            return "unsupported";
        case core::ResourceSupportStatus::ProfilingFailed:
            return "profiling_failed";
    }
    return "unknown";
}

}  // namespace

void ArtifactWriter::write(
    const std::filesystem::path& directory,
    const std::filesystem::path& source_model_path,
    const onnx::ModelProto& model,
    const core::GraphInfo& graph,
    const core::OperatorProfileMap& operator_profiles,
    const segmentation::SegmentationResult& result,
    const config::OfflineConfig& config) {

    std::filesystem::create_directories(directory / "segments");

    YAML::Emitter yaml;
    yaml << YAML::BeginMap;
    yaml << YAML::Key << "format_version" << YAML::Value << 2;
    yaml << YAML::Key << "model" << YAML::Value << YAML::BeginMap;
    yaml << YAML::Key << "path" << YAML::Value << source_model_path.string();
    yaml << YAML::Key << "name" << YAML::Value << graph.name;
    yaml << YAML::Key << "ir_version" << YAML::Value << model.ir_version();
    yaml << YAML::Key << "opset" << YAML::Value
         << (model.opset_import_size() ? model.opset_import(0).version() : 0);
    yaml << YAML::EndMap;

    yaml << YAML::Key << "configuration" << YAML::Value << YAML::BeginMap;
    yaml << YAML::Key << "max_levels" << YAML::Value << config.max_levels;
    yaml << YAML::Key << "warmup_iterations" << YAML::Value
         << config.warmup_iterations;
    yaml << YAML::Key << "measurement_iterations" << YAML::Value
         << config.measurement_iterations;
    yaml << YAML::Key << "percentile" << YAML::Value << config.percentile;
    yaml << YAML::Key << "threshold_ms" << YAML::Value
         << config.average_segment_cost_threshold_ms;
    yaml << YAML::EndMap;

    yaml << YAML::Key << "operator_capabilities" << YAML::Value
         << YAML::BeginSeq;
    for (const auto& op : graph.operators) {
        const auto it = operator_profiles.find(op.graph_index);
        if (it == operator_profiles.end()) {
            throw std::runtime_error(
                "Missing operator profile for manifest: " + op.name);
        }

        yaml << YAML::BeginMap;
        yaml << YAML::Key << "graph_index" << YAML::Value << op.graph_index;
        yaml << YAML::Key << "name" << YAML::Value << op.name;
        yaml << YAML::Key << "op_type" << YAML::Value << op.op_type;
        yaml << YAML::Key << "resources" << YAML::Value << YAML::BeginSeq;
        for (const auto& cost : it->second.costs) {
            yaml << YAML::BeginMap;
            yaml << YAML::Key << "resource" << YAML::Value
                 << resource_name(cost.resource);
            yaml << YAML::Key << "status" << YAML::Value
                 << status_name(cost.status);
            yaml << YAML::Key << "mean_ms" << YAML::Value << cost.mean_ms;
            yaml << YAML::Key << "percentile_ms" << YAML::Value
                 << cost.percentile_ms;
            if (!cost.reason.empty()) {
                yaml << YAML::Key << "reason" << YAML::Value << cost.reason;
            }
            yaml << YAML::EndMap;
        }
        yaml << YAML::EndSeq;
        yaml << YAML::EndMap;
    }
    yaml << YAML::EndSeq;

    yaml << YAML::Key << "levels" << YAML::Value << YAML::BeginSeq;

    for (std::size_t level = 0; level < result.levels.size(); ++level) {
        yaml << YAML::BeginMap;
        yaml << YAML::Key << "level" << YAML::Value << level;
        if (level < result.level_statistics.size()) {
            const auto& stats = result.level_statistics[level];
            yaml << YAML::Key << "segment_count" << YAML::Value
                 << stats.segment_count;
            yaml << YAML::Key << "average_cost_ms" << YAML::Value
                 << stats.average_cost_ms;
        }
        yaml << YAML::Key << "segments" << YAML::Value << YAML::BeginSeq;

        for (const auto& profile : result.levels[level]) {
            const auto segment_dir =
                directory / "segments" /
                ("level_" + std::to_string(profile.segment.level));

            std::filesystem::create_directories(segment_dir);

            const auto model_path =
                segment_dir /
                ("segment_" + std::to_string(profile.segment.id) + ".onnx");

            const auto segment_model =
                model::SegmentModelGenerator::build(
                    model, graph, profile.segment);

            model::SegmentModelGenerator::save(segment_model, model_path);

            yaml << YAML::BeginMap;
            yaml << YAML::Key << "id" << YAML::Value << profile.segment.id;
            yaml << YAML::Key << "model" << YAML::Value
                 << std::filesystem::relative(model_path, directory).string();

            yaml << YAML::Key << "operators" << YAML::Value << YAML::BeginSeq;
            for (const auto& name : profile.segment.operator_names) {
                yaml << name;
            }
            yaml << YAML::EndSeq;

            yaml << YAML::Key << "inputs" << YAML::Value << YAML::BeginSeq;
            for (const auto& name : profile.segment.input_tensors) {
                yaml << name;
            }
            yaml << YAML::EndSeq;

            yaml << YAML::Key << "outputs" << YAML::Value << YAML::BeginSeq;
            for (const auto& name : profile.segment.output_tensors) {
                yaml << name;
            }
            yaml << YAML::EndSeq;

            yaml << YAML::Key << "costs" << YAML::Value << YAML::BeginSeq;
            for (const auto& cost : profile.costs) {
                yaml << YAML::BeginMap;
                yaml << YAML::Key << "resource" << YAML::Value
                     << resource_name(cost.resource);
                yaml << YAML::Key << "status" << YAML::Value
                     << status_name(cost.status);
                yaml << YAML::Key << "mean_ms" << YAML::Value
                     << cost.mean_ms;
                yaml << YAML::Key << "percentile_ms" << YAML::Value
                     << cost.percentile_ms;
                if (!cost.reason.empty()) {
                    yaml << YAML::Key << "reason" << YAML::Value
                         << cost.reason;
                }
                yaml << YAML::EndMap;
            }
            yaml << YAML::EndSeq;

            yaml << YAML::EndMap;
        }

        yaml << YAML::EndSeq;
        yaml << YAML::EndMap;
    }

    yaml << YAML::EndSeq;
    yaml << YAML::EndMap;

    const auto manifest = directory / "manifest.yaml";
    std::ofstream out(manifest);
    if (!out) {
        throw std::runtime_error(
            "Failed to create artifact manifest: " + manifest.string());
    }
    out << yaml.c_str();
}

}  // namespace flexon::offline::artifact
