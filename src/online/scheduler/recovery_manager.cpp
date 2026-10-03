#include <iostream>
#include <thread>
#include <optional>

#include <flexon/online/scheduler/recovery_manager.hpp>
#include <flexon/online/execution/executor.hpp>
#include <flexon/online/execution/helper.hpp>

namespace flexon::online::recovery {

executor::SegmentExecutionStats execute_segment_with_recovery(
    runtime::SegmentRuntime& segment,
    core::Resource primary_resource,
    const scheduler::SchedulerSegmentCosts& costs,
    scheduler::OnlineScheduler& scheduler,
    const scheduler::SchedulerConfig& scheduler_config,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    arena::BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::future<executor::SegmentExecutionResult>>& background_recoveries) {

    const auto primary_ready = std::make_shared<std::promise<void>>();
    auto primary_ready_future = primary_ready->get_future();

    auto launch = [&](core::Resource resource,
                      const std::shared_ptr<std::promise<void>>& ready) {
        return std::async(
            std::launch::async,
            [&segment, resource, &tensor_store, &arena, &allocator, ready]() {
                try {
                    return executor::execute_segment_once(
                        segment,
                        resource,
                        tensor_store,
                        arena,
                        allocator,
                        ready);
                } catch (...) {
                    try {
                        ready->set_exception(std::current_exception());
                    } catch (...) {
                    }
                    throw;
                }
            });
    };

    auto primary_future = launch(primary_resource, primary_ready);

    // Wait until the primary worker has finished reading its input tensors.
    // From this point onward a speculative recovery worker can safely continue
    // without racing tensor-store liveness updates in the caller.
    primary_ready_future.get();

    bool recovery_triggered = false;
    core::Resource recovery_resource = core::Resource::CPU;
    std::future<executor::SegmentExecutionResult> recovery_future;

    const auto segment_start = std::chrono::steady_clock::now();
    while (primary_future.wait_for(std::chrono::milliseconds(0)) !=
           std::future_status::ready) {

        const auto now = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(now - segment_start).count();

        if (!recovery_triggered && scheduler_config.recovery_enabled) {
            const auto alternative =
                scheduler.select_recovery_resource(costs, primary_resource);

            if (scheduler.should_trigger_recovery(
                    elapsed_ms, alternative.score)) {

                // Avoid launching a speculative execution if the primary
                // completed between the polling check and this decision.
                if (primary_future.wait_for(std::chrono::milliseconds(0)) ==
                    std::future_status::ready) {
                    break;
                }

                recovery_triggered = true;
                recovery_resource = alternative.resource;

                std::cout
                    << "[recovery] segment " << segment.manifest.id
                    << " primary=" << helper::resource_name(primary_resource)
                    << " elapsed_ms=" << elapsed_ms
                    << " alternative=" << helper::resource_name(recovery_resource)
                    << " alternative_score=" << alternative.score
                    << " gamma=" << scheduler_config.gamma << '\n';

                const auto recovery_ready =
                    std::make_shared<std::promise<void>>();
                auto recovery_ready_future = recovery_ready->get_future();
                recovery_future = launch(recovery_resource, recovery_ready);
                recovery_ready_future.get();

                // The primary may have completed while the alternative was
                // being prepared. Prefer whichever result is observed first.
                while (recovery_future.wait_for(std::chrono::milliseconds(0)) !=
                           std::future_status::ready &&
                       primary_future.wait_for(std::chrono::milliseconds(0)) !=
                           std::future_status::ready) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                const bool recovery_ready_now =
                    recovery_future.wait_for(std::chrono::milliseconds(0)) ==
                    std::future_status::ready;
                const bool primary_ready_now =
                    primary_future.wait_for(std::chrono::milliseconds(0)) ==
                    std::future_status::ready;

                if (recovery_ready_now && primary_ready_now) {
                    std::optional<executor::SegmentExecutionResult> recovery_result;
                    std::optional<executor::SegmentExecutionResult> primary_result;

                    try {
                        recovery_result.emplace(recovery_future.get());
                    } catch (...) {
                    }
                    try {
                        primary_result.emplace(primary_future.get());
                    } catch (...) {
                    }

                    if (recovery_result && primary_result) {
                        if (recovery_result->completed_at <
                            primary_result->completed_at) {
                            recovery_result->stats.elapsed_ms =
                                std::chrono::duration<double, std::milli>(
                                    recovery_result->completed_at - segment_start).count();
                            executor::commit_segment_outputs(*recovery_result, tensor_store);
                            helper::print_segment_stats(
                                segment.manifest, recovery_result->stats, arena);
                            std::cout << "[recovery] alternative result won\n";
                            return recovery_result->stats;
                        }

                        primary_result->stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                primary_result->completed_at - segment_start).count();
                        executor::commit_segment_outputs(*primary_result, tensor_store);
                        helper::print_segment_stats(
                            segment.manifest, primary_result->stats, arena);
                        std::cout << "[recovery] primary result won\n";
                        return primary_result->stats;
                    }

                    if (recovery_result) {
                        recovery_result->stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                recovery_result->completed_at - segment_start).count();
                        executor::commit_segment_outputs(*recovery_result, tensor_store);
                        helper::print_segment_stats(
                            segment.manifest, recovery_result->stats, arena);
                        std::cout << "[recovery] alternative result won\n";
                        return recovery_result->stats;
                    }

                    if (primary_result) {
                        primary_result->stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                primary_result->completed_at - segment_start).count();
                        executor::commit_segment_outputs(*primary_result, tensor_store);
                        helper::print_segment_stats(
                            segment.manifest, primary_result->stats, arena);
                        std::cout << "[recovery] primary result won\n";
                        return primary_result->stats;
                    }

                    throw std::runtime_error(
                        "Both primary and recovery executions failed");
                }

                if (recovery_ready_now) {
                    try {
                        auto result = recovery_future.get();
                        result.stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                result.completed_at - segment_start).count();
                        background_recoveries.push_back(std::move(primary_future));
                        executor::commit_segment_outputs(result, tensor_store);
                        helper::print_segment_stats(segment.manifest, result.stats, arena);
                        std::cout << "[recovery] alternative result won\n";
                        return result.stats;
                    } catch (...) {
                        // A speculative recovery failure must not invalidate a
                        // successful primary execution.
                    }
                }

                auto result = primary_future.get();
                result.stats.elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        result.completed_at - segment_start).count();
                if (recovery_future.valid()) {
                    background_recoveries.push_back(std::move(recovery_future));
                }
                executor::commit_segment_outputs(result, tensor_store);
                helper::print_segment_stats(segment.manifest, result.stats, arena);
                std::cout << "[recovery] primary result won\n";
                return result.stats;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto result = primary_future.get();
    executor::commit_segment_outputs(result, tensor_store);
    helper::print_segment_stats(segment.manifest, result.stats, arena);
    return result.stats;
}

} // namespace flexon::online::scheduler