#include <flexon/online/scheduler/scheduler.hpp>

#include <cassert>
#include <cmath>

int main() {
    flexon::online::scheduler::SchedulerConfig config;
    config.alpha = 1.2;
    config.beta = 1.6;

    flexon::online::scheduler::OnlineScheduler scheduler(config);

    constexpr int current_level = 2;
    constexpr int max_level = 4;
    constexpr double expected_period_ms = 10.0;

    const double alpha_boundary =
        config.alpha * expected_period_ms;
    const double beta_boundary =
        config.beta * expected_period_ms;

    // Below alpha * expected -> move to a finer level.
    assert(scheduler.select_next_level(
        current_level, max_level,
        alpha_boundary - 0.1,
        expected_period_ms) == current_level - 1);

    // Exactly at alpha * expected -> paper condition is strict (<).
    assert(scheduler.select_next_level(
        current_level, max_level,
        alpha_boundary,
        expected_period_ms) == current_level);

    // Between alpha and beta -> remain at current level.
    assert(scheduler.select_next_level(
        current_level, max_level,
        (alpha_boundary + beta_boundary) / 2.0,
        expected_period_ms) == current_level);

    // Exactly at beta * expected -> strict (>) means remain.
    assert(scheduler.select_next_level(
        current_level, max_level,
        beta_boundary,
        expected_period_ms) == current_level);

    // Above beta * expected -> jump to maximum level.
    assert(scheduler.select_next_level(
        current_level, max_level,
        beta_boundary + 0.1,
        expected_period_ms) == max_level);

    // Lower level should not go below 0.
    assert(scheduler.select_next_level(
        0, max_level,
        alpha_boundary - 0.1,
        expected_period_ms) == 0);

    flexon::online::scheduler::SchedulerSegmentCosts costs;
    costs.cpu = {true, 10.0};
    costs.cuda = {true, 2.0};

    const auto first = scheduler.select_first_resource(costs);
    assert(first.resource == flexon::core::Resource::CUDA);

    const auto next = scheduler.select_next_resource(
        costs,
        flexon::core::Resource::CPU,
        30.0,
        costs);

    assert(next.resource == flexon::core::Resource::CUDA);
    assert(next.score > 0.0);
    assert(std::isfinite(next.score));

    const auto recovery = scheduler.select_recovery_resource(
        costs, flexon::core::Resource::CPU);
    assert(recovery.resource == flexon::core::Resource::CUDA);
    assert(recovery.score > 0.0);
    assert(std::isfinite(recovery.score));
    assert(!scheduler.should_trigger_recovery(2.5, recovery.score));
    assert(scheduler.should_trigger_recovery(2.6 + 1.0e-6, recovery.score));

    return 0;
}
