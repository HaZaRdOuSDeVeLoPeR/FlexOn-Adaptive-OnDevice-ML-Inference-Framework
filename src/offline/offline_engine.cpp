#include <flexon/offline/offline_engine.hpp>

#include <flexon/offline/artifact/artifact_validator.hpp>
#include <flexon/offline/artifact/artifact_writer.hpp>
#include <flexon/offline/config/offline_config.hpp>
#include <flexon/offline/config/model_config.hpp>
#include <flexon/offline/graph/graph_analyzer.hpp>
#include <flexon/offline/model/onnx_model_loader.hpp>
#include <flexon/offline/profiling/segment_profiler.hpp>
#include <flexon/offline/segmentation/initial_segmenter.hpp>
#include <flexon/offline/segmentation/multi_level_segmenter.hpp>
#include <flexon/offline/shapes/runtime_shape_resolver.hpp>

#include <iostream>
#include <stdexcept>

namespace flexon::offline {

OfflineEngine::OfflineEngine(
    std::filesystem::path config_path,
    std::filesystem::path models_config_path)
    : config_path_(std::move(config_path)),
      models_config_path_(std::move(models_config_path)) {}

void OfflineEngine::run(
    const std::filesystem::path& model_path,
    const std::filesystem::path& artifact_directory) {

    const auto config = config::load(config_path_);

    const auto models_config =
        config::load_models_config(models_config_path_);

    const auto source_model =
        model::OnnxModelLoader::load(model_path);

    // Resolve all model inputs to concrete profiling shapes before
    // graph analysis. Shape inference can then propagate those concrete
    // shapes to internal segment boundaries.
    const auto resolved_model =
        config::resolve_profiling_shapes(
            *source_model,
            models_config.runtime);

    auto graph =
        graph::GraphAnalyzer::analyze(resolved_model);

    // Static ONNX shape inference is best-effort. Resolve any remaining
    // intermediate shapes by executing temporary probe graphs.
    shapes::RuntimeShapeResolver::resolve(
        resolved_model,
        graph);

    if (graph.operators.empty()) {
        throw std::runtime_error(
            "ONNX graph contains no operators");
    }

    std::cout
        << "[offline] model: "
        << model_path << '\n';

    std::cout
        << "[offline] graph: "
        << graph.name
        << " (" << graph.operators.size()
        << " operators)\n";

    const auto work_directory =
        artifact_directory / ".work";

    profiling::SegmentProfiler profiler(
        config,
        work_directory);

    // Profile every source operator first. These measurements establish
    // resource capability for the initial partition and are then reused by
    // the multi-level segmentation algorithm's balanced split criterion.
    core::OperatorProfileMap operator_profiles;
    operator_profiles.reserve(graph.operators.size());

    for (const auto& op : graph.operators) {
        operator_profiles.emplace(
            op.graph_index,
            profiler.profile_operator(
                resolved_model,
                graph,
                op.graph_index));
    }

    std::size_t both_resources = 0;
    std::size_t cpu_only = 0;
    std::size_t cuda_only = 0;

    for (const auto& op : graph.operators) {
        const auto& profile = operator_profiles.at(op.graph_index);
        bool cpu_supported = false;
        bool cuda_supported = false;
        for (const auto& cost : profile.costs) {
            if (cost.status != core::ResourceSupportStatus::Supported) {
                continue;
            }
            if (cost.resource == core::Resource::CPU) {
                cpu_supported = true;
            } else if (cost.resource == core::Resource::CUDA) {
                cuda_supported = true;
            }
        }

        if (cpu_supported && cuda_supported) {
            ++both_resources;
        } else if (cpu_supported) {
            ++cpu_only;
        } else if (cuda_supported) {
            ++cuda_only;
        }
    }

    std::cout
        << "[offline] operator capabilities: both=" << both_resources
        << ", cpu-only=" << cpu_only
        << ", cuda-only=" << cuda_only << '\n';

    const auto initial =
        segmentation::InitialSegmenter::create(
            graph,
            config,
            operator_profiles);

    std::cout
        << "[offline] initial segments: "
        << initial.size() << '\n';

    const auto levels =
        segmentation::MultiLevelSegmenter::build(
            resolved_model,
            graph,
            initial,
            operator_profiles,
            profiler,
            config);

    std::cout
        << "[offline] generated levels: "
        << levels.levels.size() << '\n';

    artifact::ArtifactWriter::write(
        artifact_directory,
        model_path,
        resolved_model,
        graph,
        operator_profiles,
        levels,
        config);

    artifact::ArtifactValidator::validate(
        artifact_directory);

    std::filesystem::remove_all(
        work_directory);

    std::cout
        << "[offline] artifact validated: "
        << artifact_directory << '\n';
}

}  // namespace flexon::offline