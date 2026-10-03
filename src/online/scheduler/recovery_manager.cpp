#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>

#include <flexon/online/scheduler/recovery_manager.hpp>
#include <flexon/online/execution/executor.hpp>
#include <flexon/online/execution/helper.hpp>

namespace flexon::online::recovery {
namespace {

struct CompletionState {
    std::mutex mutex;
    std::condition_variable cv;
    bool primary_completed{false};
    bool recovery_completed{false};
};

void mark_completed(
    const std::shared_ptr<CompletionState>& state,
    bool recovery) {

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (recovery) {
            state->recovery_completed = true;
        } else {
            state->primary_completed = true;
        }
    }
    state->cv.notify_all();
}

} // namespace

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
    const auto completion = std::make_shared<CompletionState>();

    auto launch = [&](core::Resource resource,
                      const std::shared_ptr<std::promise<void>>& ready,
                      bool recovery) {
        return std::async(
            std::launch::async,
            [&segment, resource, &tensor_store, &arena, &allocator,
             ready, completion, recovery]() {
                try {
                    auto result = executor::execute_segment_once(
                        segment,
                        resource,
                        tensor_store,
                        arena,
                        allocator,
                        ready);
                    mark_completed(completion, recovery);
                    return result;
                } catch (...) {
                    try {
                        ready->set_exception(std::current_exception());
                    } catch (...) {
                    }
                    mark_completed(completion, recovery);
                    throw;
                }
            });
    };

    auto primary_future = launch(primary_resource, primary_ready, false);

    // Wait until the primary worker has finished reading its input tensors.
    // From this point onward a speculative recovery worker can safely continue
    // without racing tensor-store liveness updates in the caller.
    primary_ready_future.get();

    const auto segment_start = std::chrono::steady_clock::now();

    while (true) {
        const auto alternative =
            scheduler.select_recovery_resource(costs, primary_resource);

        if (!std::isfinite(alternative.score)) {
            std::unique_lock<std::mutex> lock(completion->mutex);
            completion->cv.wait(
                lock,
                [&] { return completion->primary_completed; });
            break;
        }

        const auto elapsed_ms = [&] {
            const auto now = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(
                now - segment_start).count();
        }();

        if (scheduler.should_trigger_recovery(
                elapsed_ms, alternative.score)) {

            // Avoid launching a speculative execution if the primary has
            // completed since the recovery decision was evaluated.
            {
                std::lock_guard<std::mutex> lock(completion->mutex);
                if (completion->primary_completed) {
                    break;
                }
            }

            std::cout
                << "[recovery] segment " << segment.manifest.id
                << " primary=" << helper::resource_name(primary_resource)
                << " elapsed_ms=" << elapsed_ms
                << " alternative=" << helper::resource_name(alternative.resource)
                << " alternative_score=" << alternative.score
                << " gamma=" << scheduler_config.gamma << '\n';

            const auto recovery_ready =
                std::make_shared<std::promise<void>>();
            auto recovery_ready_future = recovery_ready->get_future();
            auto recovery_future =
                launch(alternative.resource, recovery_ready, true);
            try {
                recovery_ready_future.get();
            } catch (...) {
                // Failure while binding recovery inputs means the speculative
                // execution never became viable. Consume its future and keep
                // the already-running primary execution authoritative.
                try {
                    recovery_future.get();
                } catch (...) {
                }

                auto result = primary_future.get();
                result.stats.elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        result.completed_at - segment_start).count();
                executor::commit_segment_outputs(result, tensor_store);
                helper::print_segment_stats(
                    segment.manifest, result.stats, arena);
                std::cout
                    << "[recovery] elapsed_ms=" << result.stats.elapsed_ms
                    << " | primary won\n";
                return result.stats;
            }

            // Wait for an execution-completion notification instead of
            // repeatedly polling both futures. The losing speculative
            // execution is deliberately not cancelled because ONNX Runtime
            // does not provide safe cancellation semantics for an in-flight
            // segment.
            {
                std::unique_lock<std::mutex> lock(completion->mutex);
                completion->cv.wait(
                    lock,
                    [&] {
                        return completion->primary_completed ||
                               completion->recovery_completed;
                    });
            }

            bool primary_completed = false;
            bool recovery_completed = false;
            {
                std::lock_guard<std::mutex> lock(completion->mutex);
                primary_completed = completion->primary_completed;
                recovery_completed = completion->recovery_completed;
            }

            if (primary_completed && recovery_completed) {
                std::optional<executor::SegmentExecutionResult> primary_result;
                std::optional<executor::SegmentExecutionResult> recovery_result;

                try {
                    primary_result.emplace(primary_future.get());
                } catch (...) {
                }
                try {
                    recovery_result.emplace(recovery_future.get());
                } catch (...) {
                }

                if (primary_result && recovery_result) {
                    if (recovery_result->completed_at <
                        primary_result->completed_at) {
                        recovery_result->stats.elapsed_ms =
                            std::chrono::duration<double, std::milli>(
                                recovery_result->completed_at - segment_start).count();
                        executor::commit_segment_outputs(
                            *recovery_result, tensor_store);
                        helper::print_segment_stats(
                            segment.manifest, recovery_result->stats, arena);
                        std::cout
                            << "[recovery] elapsed_ms=" << recovery_result->stats.elapsed_ms
                            << " alternative won\n";
                        return recovery_result->stats;
                    }

                    primary_result->stats.elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            primary_result->completed_at - segment_start).count();
                    executor::commit_segment_outputs(
                        *primary_result, tensor_store);
                    helper::print_segment_stats(
                        segment.manifest, primary_result->stats, arena);
                    std::cout
                        << "[recovery] elapsed_ms=" << primary_result->stats.elapsed_ms
                        << " primary won\n";
                    return primary_result->stats;
                }

                if (recovery_result) {
                    recovery_result->stats.elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            recovery_result->completed_at - segment_start).count();
                    executor::commit_segment_outputs(
                        *recovery_result, tensor_store);
                    helper::print_segment_stats(
                        segment.manifest, recovery_result->stats, arena);
                    std::cout
                        << "[recovery] elapsed_ms=" << recovery_result->stats.elapsed_ms
                        << " alternative won\n";
                    return recovery_result->stats;
                }

                if (primary_result) {
                    primary_result->stats.elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            primary_result->completed_at - segment_start).count();
                    executor::commit_segment_outputs(
                        *primary_result, tensor_store);
                    helper::print_segment_stats(
                        segment.manifest, primary_result->stats, arena);
                    std::cout
                        << "[recovery] elapsed_ms=" << primary_result->stats.elapsed_ms
                        << " primary won\n";
                    return primary_result->stats;
                }

                throw std::runtime_error(
                    "Both primary and recovery executions failed");
            }

            if (recovery_completed) {
                try {
                    auto result = recovery_future.get();
                    result.stats.elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            result.completed_at - segment_start).count();
                    background_recoveries.push_back(std::move(primary_future));
                    executor::commit_segment_outputs(result, tensor_store);
                    helper::print_segment_stats(
                        segment.manifest, result.stats, arena);
                    std::cout
                        << "[recovery] elapsed_ms=" << result.stats.elapsed_ms
                        << " | alternative won\n";
                    return result.stats;
                } catch (...) {
                    // A speculative recovery failure must not invalidate a
                    // primary execution that may still succeed.
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
            std::cout
                << "[recovery] elapsed_ms=" << result.stats.elapsed_ms
                << " | primary won\n";
            return result.stats;
        }

        // The resource monitor is sampled periodically. Wait until either the
        // primary completes or the next recovery evaluation point is reached.
        // This avoids the old 1-ms active polling loop while retaining prompt
        // response to primary completion.
        const auto sample_interval = std::chrono::milliseconds(
            std::max<std::uint32_t>(
                1, scheduler_config.resource_sample_interval_ms));
        const auto recovery_threshold = std::chrono::milliseconds(
            static_cast<std::int64_t>(std::ceil(
                std::max(0.0, scheduler_config.gamma * alternative.score))));
        const auto now = std::chrono::steady_clock::now();
        const auto next_sample = now + sample_interval;
        const auto next_threshold = segment_start + recovery_threshold;
        const auto wake_at = std::min(next_sample, next_threshold);

        std::unique_lock<std::mutex> lock(completion->mutex);
        completion->cv.wait_until(
            lock,
            wake_at,
            [&] { return completion->primary_completed; });
        if (completion->primary_completed) {
            break;
        }
    }

    auto result = primary_future.get();
    executor::commit_segment_outputs(result, tensor_store);
    helper::print_segment_stats(segment.manifest, result.stats, arena);
    return result.stats;
}

} // namespace flexon::online::recovery
