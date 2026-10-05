#include <future>
#include <thread>
#include <iostream>

#include <flexon/online/scheduler/scheduler.hpp>
#include <flexon/online/resource/resource_monitor.hpp>
#include <flexon/online/scheduler/thread_priority.hpp>

namespace flexon::online::scheduler {

SchedulerSegmentCosts scheduler_costs(const manifest::SegmentManifest& segment) {
    return SchedulerSegmentCosts{
        SchedulerResourceCost{
            segment.cpu_supported,
            segment.cpu_mean_ms},
        SchedulerResourceCost{
            segment.cuda_supported,
            segment.cuda_mean_ms}};
}

struct OnlineScheduler::Impl {
    mutable std::mutex mutex;
    monitor::MonitorState state;
    monitor::CpuSample previous_cpu;
    std::thread monitor_thread;
    std::atomic_bool running{false};

    void update_once() {
        const auto cpu = monitor::cpu_remaining_capacity(previous_cpu);
        const auto cuda = monitor::query_cuda_remaining_capacity();

        std::lock_guard<std::mutex> lock(mutex);
        state.cpu_remaining = cpu;
        state.cuda_remaining = cuda;
    }
};

OnlineScheduler::OnlineScheduler(SchedulerConfig config)
    : config_(std::move(config)),
      impl_(std::make_unique<Impl>()) {}

OnlineScheduler::~OnlineScheduler() {
    stop();
}

SchedulerConfig OnlineScheduler::load_config(
    const std::filesystem::path& path) {

    SchedulerConfig config;
    if (!std::filesystem::exists(path)) {
        return config;
    }

    const auto root = YAML::LoadFile(path.string());

    // scheduler.yaml owns the level-selection parameters.
    if (root["level_selection"]) {
        const auto node = root["level_selection"];
        if (node["alpha"]) config.alpha = node["alpha"].as<double>();
        if (node["beta"]) config.beta = node["beta"].as<double>();
    }

    // scheduler.yaml owns the recovery parameters.
    if (root["recovery"]) {
        const auto node = root["recovery"];
        if (node["gamma"]) config.gamma = node["gamma"].as<double>();
        if (node["enabled"]) config.recovery_enabled = node["enabled"].as<bool>();
    }

    // scheduler.yaml owns the resource-monitor sampling interval.
    if (root["resource_monitor"] &&
        root["resource_monitor"]["sample_interval_ms"]) {
        config.resource_sample_interval_ms =
            root["resource_monitor"]["sample_interval_ms"].as<std::uint32_t>();
    }

    config.alpha = std::max(0.0, config.alpha);
    config.beta = std::max(config.alpha, config.beta);
    config.gamma = std::max(0.0, config.gamma);
    config.resource_sample_interval_ms = std::max<std::uint32_t>(
        10, config.resource_sample_interval_ms);

    return config;
}

void OnlineScheduler::start() {
    if (impl_->running) return;

    impl_->previous_cpu = monitor::read_cpu_sample();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->state = monitor::MonitorState{};
    }

    impl_->running = true;
    auto startup = std::make_shared<std::promise<void>>();
    auto startup_result = startup->get_future();

    impl_->monitor_thread = std::thread([this, startup]() {
        try {
            if (config_.priority_isolation) {
                priority::promote_current_thread();
            }

            std::cout << "[priority] scheduler thread ready tid="
                      << priority::current_thread_id() << " "
                      << priority::describe_current_thread() << '\n';
            startup->set_value();

            while (impl_->running) {
                impl_->update_once();

                std::this_thread::sleep_for(
                    std::chrono::milliseconds(
                        config_.resource_sample_interval_ms));
            }
        } catch (...) {
            impl_->running = false;
            try {
                startup->set_exception(std::current_exception());
            } catch (...) {
            }
        }
    });

    try {
        startup_result.get();
    } catch (...) {
        stop();
        throw;
    }
}

void OnlineScheduler::stop() {
    impl_->running = false;
    if (impl_->monitor_thread.joinable()) {
        impl_->monitor_thread.join();
    }
}

double OnlineScheduler::remaining_capacity(
    core::Resource resource) const {

    std::lock_guard<std::mutex> lock(impl_->mutex);
    switch (resource) {
        case core::Resource::CPU:
            return impl_->state.cpu_remaining;
        case core::Resource::CUDA:
            return impl_->state.cuda_remaining;
        case core::Resource::Auto:
            break;
    }
    return 1.0;
}

std::array<SchedulerDecision, 2> OnlineScheduler::select_first_resource(
    const SchedulerSegmentCosts& next) const {

    std::array<SchedulerDecision, 2> best;

    for (auto candidate :
         {core::Resource::CPU, core::Resource::CUDA}) {
        const auto cost =
            candidate == core::Resource::CPU ? next.cpu : next.cuda;
        if (!cost.supported || !std::isfinite(cost.expected_ms)) continue;

        constexpr double kCapacityEpsilon = 1.0e-9;
        const double capacity =
            std::clamp(remaining_capacity(candidate), 0.0, 1.0);
        const double degradation =
            1.0 / (capacity + kCapacityEpsilon);
        const double score = degradation * cost.expected_ms;

        best[(int)candidate] =
            {candidate, score, degradation, capacity};
    }

    std::sort(best.begin(), best.end(), 
    [](const SchedulerDecision &a, const SchedulerDecision &b){
        return a.score < b.score;
    });

    if (!std::isfinite(best[0].score)) {
        throw std::runtime_error(
            "No supported resource is available for the next segment");
    }

    return best;
}

std::array<SchedulerDecision, 2> OnlineScheduler::select_next_resource(
    const SchedulerSegmentCosts& current,
    core::Resource current_resource,
    double current_measured_ms,
    const SchedulerSegmentCosts& next) const {

    std::array<SchedulerDecision, 2> best;

    double current_expected = std::numeric_limits<double>::infinity();
    if (current_resource == core::Resource::CPU)
        current_expected = current.cpu.expected_ms;
    else if (current_resource == core::Resource::CUDA)
        current_expected = current.cuda.expected_ms;

    const double current_degradation =
        (std::isfinite(current_expected) && current_expected > 0.0 &&
         std::isfinite(current_measured_ms) && current_measured_ms > 0.0)
            ? current_measured_ms / current_expected
            : 1.0;

    for (const auto candidate :
         {core::Resource::CPU, core::Resource::CUDA}) {
        const auto cost =
            candidate == core::Resource::CPU ? next.cpu : next.cuda;
        if (!cost.supported || !std::isfinite(cost.expected_ms)) continue;

        double degradation = 1.0;
        double capacity = 1.0;

        if (candidate == current_resource) {
            degradation = current_degradation;
            capacity = std::clamp(
                1.0 / std::max(degradation, 1.0e-9), 0.0, 1.0);
        } else {
            constexpr double kCapacityEpsilon = 1.0e-9;
            capacity = std::clamp(
                remaining_capacity(candidate), 0.0, 1.0);
            degradation = 1.0 / (capacity + kCapacityEpsilon);
        }

        const double score = degradation * cost.expected_ms;
        best[(int)candidate] =
            {candidate, score, degradation, capacity};
    }

    std::sort(best.begin(), best.end(), 
    [](const SchedulerDecision &a, const SchedulerDecision &b){
        return a.score < b.score;
    });

    if (!std::isfinite(best[0].score)) {
        throw std::runtime_error(
            "No supported resource is available for the next segment");
    }

    return best;
}

SchedulerDecision OnlineScheduler::select_recovery_resource(
    const SchedulerSegmentCosts& current,
    core::Resource current_resource) const {

    SchedulerDecision best;

    for (const auto candidate :
         {core::Resource::CPU, core::Resource::CUDA}) {
        if (candidate == current_resource) continue;

        const auto cost =
            candidate == core::Resource::CPU ? current.cpu : current.cuda;
        if (!cost.supported || !std::isfinite(cost.expected_ms)) continue;

        constexpr double kCapacityEpsilon = 1.0e-9;
        const double capacity =
            std::clamp(remaining_capacity(candidate), 0.0, 1.0);
        const double degradation =
            1.0 / (capacity + kCapacityEpsilon);
        const double score = degradation * cost.expected_ms;

        if (score < best.score) {
            best = SchedulerDecision{
                candidate, score, degradation, capacity};
        }
    }

    return best;
}

bool OnlineScheduler::should_trigger_recovery(
    double current_elapsed_ms,
    double alternative_score) const {

    if (!config_.recovery_enabled ||
        !std::isfinite(current_elapsed_ms) ||
        !std::isfinite(alternative_score) ||
        current_elapsed_ms < 0.0 ||
        alternative_score < 0.0) {
        return false;
    }

    // Eq. (5): d* C*(s_i) > gamma * min_{r != r*} d_r C_r(s_i).
    // While the segment is executing, the observed elapsed time is the
    // current estimate of d* C*(s_i), since d* = R/C*(s_i).
    return current_elapsed_ms > config_.gamma * alternative_score;
}

std::uint32_t OnlineScheduler::select_next_level(
    std::uint32_t current_level,
    std::uint32_t max_level,
    double measured_period_ms,
    double expected_period_ms) const {

    if (current_level > max_level) {
        throw std::invalid_argument(
            "Current segmentation level exceeds maximum level");
    }

    if (!std::isfinite(measured_period_ms) ||
        !std::isfinite(expected_period_ms) ||
        expected_period_ms <= 0.0) {
        return current_level;
    }

    if (measured_period_ms <
            config_.alpha * expected_period_ms &&
        current_level >= 1) {
        return current_level - 1;
    }

    if (measured_period_ms >
            config_.beta * expected_period_ms) {
        return max_level;
    }

    return current_level;
}

};