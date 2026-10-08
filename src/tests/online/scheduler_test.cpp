#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

#include <flexon/online/scheduler/scheduler.hpp>

namespace {

template <typename Fn>
void expect_throw(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    flexon::online::scheduler::SchedulerConfig config;
    config.alpha = 1.2;
    config.beta = 1.6;
    config.gamma = 1.3;

    flexon::online::scheduler::OnlineScheduler scheduler(config);

    constexpr std::uint32_t current_level = 2;
    constexpr std::uint32_t max_level = 4;
    constexpr double expected_period_ms = 10.0;

    const double alpha_boundary = config.alpha * expected_period_ms;
    const double beta_boundary = config.beta * expected_period_ms;

    assert(scheduler.select_next_level(
        current_level, max_level, alpha_boundary - 0.1,
        expected_period_ms) == current_level - 1);
    assert(scheduler.select_next_level(
        current_level, max_level, alpha_boundary,
        expected_period_ms) == current_level);
    assert(scheduler.select_next_level(
        current_level, max_level,
        (alpha_boundary + beta_boundary) / 2.0,
        expected_period_ms) == current_level);
    assert(scheduler.select_next_level(
        current_level, max_level, beta_boundary,
        expected_period_ms) == current_level);
    assert(scheduler.select_next_level(
        current_level, max_level, beta_boundary + 0.1,
        expected_period_ms) == max_level);
    assert(scheduler.select_next_level(
        0, max_level, alpha_boundary - 0.1,
        expected_period_ms) == 0);

    // Invalid level and invalid expected period are handled explicitly.
    expect_throw([&] {
        (void)scheduler.select_next_level(5, max_level, 10.0, 10.0);
    });
    assert(scheduler.select_next_level(
        current_level, max_level,
        std::numeric_limits<double>::quiet_NaN(),
        expected_period_ms) == current_level);
    assert(scheduler.select_next_level(
        current_level, max_level, 10.0, 0.0) == current_level);

    flexon::online::scheduler::SchedulerSegmentCosts costs;
    costs.cpu = {true, 10.0, 5.0};
    costs.cuda = {true, 2.0, 10.0};

    const auto first = scheduler.select_first_resource(costs)[0];
    assert(first.resource == flexon::core::Resource::CUDA);
    assert(std::isfinite(first.score));

    const auto next = scheduler.select_next_resource(
        costs, flexon::core::Resource::CPU, 30.0, costs)[0];
    assert(next.resource == flexon::core::Resource::CUDA);
    assert(next.score > 0.0);
    assert(std::isfinite(next.score));

    const auto recovery = scheduler.select_recovery_resource(
        costs, flexon::core::Resource::CPU);
    assert(recovery.resource == flexon::core::Resource::CUDA);
    assert(recovery.score > 0.0);
    assert(std::isfinite(recovery.score));

    assert(!scheduler.should_trigger_recovery(2.5, recovery.score));
    assert(scheduler.should_trigger_recovery(
        config.gamma * recovery.score + 1.0e-6,
        recovery.score));
    assert(!scheduler.should_trigger_recovery(
        std::numeric_limits<double>::quiet_NaN(), recovery.score));
    assert(!scheduler.should_trigger_recovery(-1.0, recovery.score));
    assert(!scheduler.should_trigger_recovery(10.0, -1.0));


    // Linear degradation is anchored at (Umin=0.01, max_dr) and (1, 1).
    assert(std::abs(
        flexon::online::scheduler::linear_degradation(10.0, 1.0) -
        1.0) < 1.0e-12);
    assert(std::abs(
        flexon::online::scheduler::linear_degradation(10.0, 0.01) -
        10.0) < 1.0e-12);
    assert(std::abs(
        flexon::online::scheduler::linear_degradation(10.0, 0.0) -
        10.0) < 1.0e-12);
    assert(std::abs(
        flexon::online::scheduler::linear_degradation(10.0, 0.5) -
        (10.0 + 9.0 * (0.01 - 0.5) / (1.0 - 0.01))) <
        1.0e-12);
    assert(flexon::online::scheduler::linear_degradation(10.0, 0.5) <
           flexon::online::scheduler::linear_degradation(10.0, 0.25));

    // If only one resource is supported, recovery has no alternative.
    flexon::online::scheduler::SchedulerSegmentCosts cpu_only;
    cpu_only.cpu = {true, 5.0, 5.0};
    cpu_only.cuda = {false, std::numeric_limits<double>::infinity()};
    const auto no_recovery = scheduler.select_recovery_resource(
        cpu_only, flexon::core::Resource::CPU);
    assert(!std::isfinite(no_recovery.score));

    // Config precedence: resource_monitor is the authoritative setting.
    const auto config_path =
        std::filesystem::temp_directory_path() /
        "flexon_scheduler_test.yaml";
    {
        std::ofstream file(config_path);
        file<< "level_selection:\n"
            << "  alpha: 1.4\n"
            << "  beta: 1.8\n"
            << "recovery:\n"
            << "  gamma: 1.5\n"
            << "resource_monitor:\n"
            << "  sample_interval_ms: 25\n";
    }

    const auto loaded =
        flexon::online::scheduler::OnlineScheduler::load_config(config_path);
    assert(loaded.alpha == 1.4);
    assert(loaded.beta == 1.8);
    assert(loaded.gamma == 1.5);
    assert(loaded.resource_sample_interval_ms == 25);
    std::filesystem::remove(config_path);

    return 0;
}
