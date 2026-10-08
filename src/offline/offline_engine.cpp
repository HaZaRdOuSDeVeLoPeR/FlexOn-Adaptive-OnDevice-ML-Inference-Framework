#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <flexon/contention/contention_generator.hpp>
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

    auto levels =
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
    
    for (size_t i = 0; i < levels.levels.size(); i++)
    {
        std::cout
            << "[offline] level " << i <<": " 
            << levels.levels[i].size() << " segments" << '\n';
    }

    // Profile every generated segment again while the standardized external
    // contention workload is active. The normal mean_ms/percentile_ms fields
    // remain the ideal profile. Each segment/resource pair receives its own
    // maximum degradation ratio: contended_mean_ms / ideal_mean_ms. The
    // runtime uses this precomputed ratio directly, avoiding an expensive
    // division inside the online scheduling loop.
    if (config.degradation_profiling_enabled) {
        contention::Config contention_config;
        contention_config.targets.cpu_percent = config.degradation_cpu_percent;
        contention_config.targets.gpu_percent = config.degradation_gpu_percent;
        contention_config.targets.dram_percent = config.degradation_dram_percent;
        contention_config.targets.vram_percent = config.degradation_vram_percent;
        contention_config.safety_factor = config.degradation_safety_factor;
        contention_config.cuda_device_id = config.cuda_device_id;
        contention_config.verbose = false;

        std::cout
            << "[degradation] starting standardized contention: "
            << "cpu=" << config.degradation_cpu_percent << "% "
            << "gpu=" << config.degradation_gpu_percent << "% "
            << "dram=" << config.degradation_dram_percent << "% "
            << "vram=" << config.degradation_vram_percent << "%\n";

        contention::ContentionGenerator contention_generator(
            std::move(contention_config));
        contention_generator.start();

        std::cout
            << "[degradation] waiting "
            << config.degradation_stabilization_seconds
            << " s for contention to stabilize\n";
        std::this_thread::sleep_for(
            std::chrono::seconds(config.degradation_stabilization_seconds));

        const auto state = contention_generator.state();
        std::cout
            << "[degradation] stabilized sample: "
            << "cpu=" << state.cpu_utilization_percent << "% "
            << "gpu=" << state.gpu_utilization_percent << "% "
            << "dram=" << state.dram_bandwidth_gbps << " GB/s "
            << "vram=" << state.vram_bandwidth_gbps << " GB/s\n";

        for (std::size_t level = 0; level < levels.levels.size(); ++level) {
            for (auto& ideal_profile : levels.levels[level]) {
                const auto observed_profile = profiler.profile_under_contention(
                    resolved_model, graph, ideal_profile.segment);

                for (auto& ideal_cost : ideal_profile.costs) {
                    if (ideal_cost.status !=
                            core::ResourceSupportStatus::Supported ||
                        !std::isfinite(ideal_cost.mean_ms) ||
                        ideal_cost.mean_ms <= 0.0) {
                        continue;
                    }

                    const auto observed = std::find_if(
                        observed_profile.costs.begin(),
                        observed_profile.costs.end(),
                        [&](const auto& cost) {
                            return cost.resource == ideal_cost.resource &&
                                   cost.status ==
                                       core::ResourceSupportStatus::Supported &&
                                   std::isfinite(cost.mean_ms) &&
                                   cost.mean_ms > 0.0;
                        });

                    if (observed == observed_profile.costs.end()) {
                        throw std::runtime_error(
                            "Missing contended degradation measurement for "
                            "level " + std::to_string(level) +
                            ", segment " +
                            std::to_string(ideal_profile.segment.id) +
                            ", resource " +
                            (ideal_cost.resource == core::Resource::CPU
                                 ? "cpu"
                                 : "cuda"));
                    }

                    const double ratio =
                        observed->mean_ms / ideal_cost.mean_ms;
                    if (!std::isfinite(ratio) || ratio <= 0.0) {
                        throw std::runtime_error(
                            "Invalid degradation ratio for level " +
                            std::to_string(level) + ", segment " +
                            std::to_string(ideal_profile.segment.id));
                    }

                    ideal_cost.max_degradation_ratio =
                        std::max(1.0, ratio);

                    std::cout
                        << "[degradation] level=" << level
                        << " segment=" << ideal_profile.segment.id
                        << " resource="
                        << (ideal_cost.resource == core::Resource::CPU
                                ? "cpu" : "cuda")
                        << " ideal_ms=" << ideal_cost.mean_ms
                        << " contended_ms=" << observed->mean_ms
                        << " max_degradation_ratio="
                        << ideal_cost.max_degradation_ratio << '\n';
                }
            }

        }

        // The generator owns RAII cleanup. It is destroyed before artifact
        // writing continues, so all contention threads are joined before the
        // generated artifacts are touched again.
    } else {
        std::cout
            << "[degradation] contention profiling disabled; "
            << "max_degradation_ratio remains 1.0\n";
    }

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
