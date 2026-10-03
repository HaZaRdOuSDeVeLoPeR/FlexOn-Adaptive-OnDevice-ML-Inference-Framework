#include "flexon/core/types.hpp"
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>
#include <flexon/online/execution/helper.hpp>
#include <flexon/online/runtime_engine.hpp>


namespace flexon::online {

struct FlexOnRuntimeImpl {
    std::unique_ptr<runtime::RuntimeState> state;
};

FlexOnRuntime::FlexOnRuntime()
    : impl_(std::make_unique<FlexOnRuntimeImpl>()) {}

FlexOnRuntime::~FlexOnRuntime() = default;
FlexOnRuntime::FlexOnRuntime(FlexOnRuntime&&) noexcept = default;
FlexOnRuntime& FlexOnRuntime::operator=(FlexOnRuntime&&) noexcept = default;

void FlexOnRuntime::load(const std::filesystem::path& artifact_directory) {
    auto artifact = manifest::load_manifest(artifact_directory);

    auto state = std::make_unique<runtime::RuntimeState>();
    state->artifact = std::move(artifact);
    state->levels.resize(state->artifact.levels.size());

    for (std::size_t level_index = 0;
         level_index < state->artifact.levels.size(); ++level_index) {
        const auto& level = state->artifact.levels[level_index];
        auto& runtime_level = state->levels[level_index];
        runtime_level.reserve(level.segments.size());

        for (const auto& manifest_segment : level.segments) {
            runtime::SegmentRuntime runtime_segment;
            runtime_segment.manifest = manifest_segment;

            // Pre-create every resource variant supported by the offline
            // artifact. No session construction occurs in run().
            if (manifest_segment.cpu_supported) {
                runtime_segment.sessions.push_back(
                    runtime::PreparedSession{
                        core::Resource::CPU,
                        helper::create_session(
                            state->env,
                            manifest_segment.model_path,
                            core::Resource::CPU)});
            }
            if (manifest_segment.cuda_supported) {
                runtime_segment.sessions.push_back(
                    runtime::PreparedSession{
                        core::Resource::CUDA,
                        helper::create_session(
                        state->env,
                        manifest_segment.model_path,
                        core::Resource::CUDA)
                    });
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
        for (const auto resource : options.resource_plan) {
            if (resource == core::Resource::Auto) {
                throw std::invalid_argument(
                    "resource_plan entries must be concrete cpu/cuda resources; "
                    "use --resource auto for FlexOn dynamic resource selection");
            }
        }
    }

    scheduler::SchedulerConfig scheduler_config;
    if (!options.scheduler_config_path.empty()) {
        scheduler_config =
            scheduler::OnlineScheduler::load_config(options.scheduler_config_path);
    } else {
        const std::filesystem::path default_config =
            "config/scheduler.yaml";
        scheduler_config = scheduler::OnlineScheduler::load_config(default_config);
    }

    // Explicit runtime overrides take precedence over scheduler.yaml. The
    // executable deliberately controls recovery through --with-recovery so
    // notebook experiments do not depend on a mutable YAML switch.
    if (options.scheduler_alpha_override) {
        scheduler_config.alpha = *options.scheduler_alpha_override;
    }
    if (options.scheduler_beta_override) {
        scheduler_config.beta = *options.scheduler_beta_override;
    }
    if (options.scheduler_gamma_override) {
        scheduler_config.gamma = *options.scheduler_gamma_override;
    }
    scheduler_config.recovery_enabled = options.with_recovery;

    scheduler_config.alpha = std::max(0.0, scheduler_config.alpha);
    scheduler_config.beta = std::max(scheduler_config.alpha, scheduler_config.beta);
    scheduler_config.gamma = std::max(0.0, scheduler_config.gamma);

    scheduler::OnlineScheduler scheduler(scheduler_config);
    const bool dynamic_resource_selection =
        options.resource == core::Resource::Auto;

    if (dynamic_resource_selection) {
        scheduler.start();
    }

    // The arena persists across all inference periods and levels so buffers
    // released by one inference can be reused by another level/period.
    arena::BufferArena arena;

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

            std::unordered_map<std::string, runtime::RuntimeTensor> tensor_store;
            std::vector<std::future<executor::SegmentExecutionResult>> background_recoveries;

            core::Resource previous_resource = core::Resource::CPU;
            executor::SegmentExecutionStats previous_stats{};
            manifest::SegmentManifest* previous_manifest = nullptr;

            std::cout 
                << "[online] iteration " << (iteration + 1)
                << "/" << options.iterations
                << " level=" << current_level << '\n';

            // The first segment establishes the initial resource. In Auto
            // mode this uses predicted cost divided by current remaining
            // capacity, corresponding to the resource-selection policy.
            core::Resource first_resource = options.resource;
            
            if (dynamic_resource_selection) {
                std::array<scheduler::SchedulerDecision, 2> decisions;

                decisions = scheduler.select_first_resource(
                    scheduler::scheduler_costs(level.front().manifest));
                first_resource = decisions[0].resource;

                std::cout
                    << "[scheduler] segment 0"
                    << " selected=" << helper::resource_name(first_resource)
                    << " score=" << decisions[0].score
                    << " degradation=" << decisions[0].degradation
                    << " remaining_capacity=" << decisions[0].remaining_capacity << '\n';
                
                std::cout
                    << "[scheduler] segment 0" 
                    << " alternative=" << helper::resource_name(decisions[1].resource)
                    << " score=" << decisions[1].score
                    << " degradation=" << decisions[1].degradation
                    << " remaining_capacity=" << decisions[1].remaining_capacity << '\n';
            }

            auto* first_prepared =
                helper::find_session(level.front(), first_resource);
            if (!first_prepared || !first_prepared->session) {
                throw std::runtime_error(
                    "First segment session is not prepared for selected resource");
            }

            helper::add_initial_inputs(
                tensor_store,
                level.front().manifest.input_names,
                *first_prepared->session,
                impl_->state->allocator);

            for (std::size_t segment_index = 0;
                 segment_index < level.size();
                 ++segment_index) {

                auto& segment = level[segment_index];
                core::Resource requested_resource = options.resource;

                if (!options.resource_plan.empty()) {
                    requested_resource =
                        options.resource_plan[segment_index];
                }
                else if (dynamic_resource_selection &&
                           segment_index == 0) {
                    requested_resource = first_resource;
                }
                else if (dynamic_resource_selection) {
                    const auto decisions = scheduler.select_next_resource(
                        scheduler::scheduler_costs(*previous_manifest),
                        previous_resource,
                        previous_stats.elapsed_ms,
                        scheduler::scheduler_costs(segment.manifest));

                    requested_resource = decisions[0].resource;

                    std::cout
                        << "[scheduler] segment " << segment.manifest.id
                        << " selected=" << helper::resource_name(requested_resource)
                        << " score=" << decisions[0].score
                        << " degradation=" << decisions[0].degradation
                        << " remaining_capacity=" << decisions[0].remaining_capacity << '\n';

                    std::cout
                        << "[scheduler] segment " << segment.manifest.id
                        << " alternative=" << helper::resource_name(decisions[1].resource)
                        << " score=" << decisions[1].score
                        << " degradation=" << decisions[1].degradation
                        << " remaining_capacity=" << decisions[1].remaining_capacity << '\n';
                }

                executor::SegmentExecutionStats stats;
                if (dynamic_resource_selection &&
                    scheduler_config.recovery_enabled) {
                    stats = recovery::execute_segment_with_recovery(
                        segment,
                        requested_resource,
                        scheduler::scheduler_costs(segment.manifest),
                        scheduler,
                        scheduler_config,
                        tensor_store,
                        arena,
                        impl_->state->allocator,
                        background_recoveries);
                } else {
                    stats = executor::execute_segment(
                        segment,
                        requested_resource,
                        tensor_store,
                        arena,
                        impl_->state->allocator);
                }

                const auto selected_expected =
                    stats.resource == core::Resource::CPU
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
