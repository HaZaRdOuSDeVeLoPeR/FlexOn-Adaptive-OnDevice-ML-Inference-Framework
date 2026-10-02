#include <flexon/online/scheduler.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace flexon::online {
namespace {

struct MonitorState {
    double cpu_remaining{1.0};
    double cuda_remaining{1.0};
};

struct CpuSample {
    std::uint64_t idle{0};
    std::uint64_t total{0};
};

CpuSample read_cpu_sample() {
    std::ifstream input("/proc/stat");
    std::string label;
    std::uint64_t user = 0;
    std::uint64_t nice = 0;
    std::uint64_t system = 0;
    std::uint64_t idle = 0;
    std::uint64_t iowait = 0;
    std::uint64_t irq = 0;
    std::uint64_t softirq = 0;
    std::uint64_t steal = 0;

    if (!(input >> label >> user >> nice >> system >> idle >> iowait
                 >> irq >> softirq >> steal) || label != "cpu") {
        return {};
    }

    const auto idle_all = idle + iowait;
    const auto total = user + nice + system + idle_all +
                       irq + softirq + steal;
    return CpuSample{idle_all, total};
}

double cpu_remaining_capacity(CpuSample& previous) {
    const auto current = read_cpu_sample();
    if (current.total <= previous.total ||
        current.idle < previous.idle) {
        previous = current;
        return 1.0;
    }

    const auto total_delta = current.total - previous.total;
    const auto idle_delta = current.idle - previous.idle;
    previous = current;

    if (total_delta == 0) return 1.0;

    const double utilization =
        std::min(1.0, static_cast<double>(idle_delta) /
                          static_cast<double>(total_delta));
    return std::clamp(1.0 - utilization, 0.0, 1.0);
}

double query_cuda_remaining_capacity() {
    // GPU utilization is intentionally sampled out-of-band so nvidia-smi
    // startup latency does not become part of the segment scheduling path.
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=utilization.gpu "
        "--format=csv,noheader,nounits -i 0 2>/dev/null",
        "r");
    if (pipe == nullptr) return 1.0;

    char buffer[64]{};
    const bool read_ok = std::fgets(buffer, sizeof(buffer), pipe) != nullptr;
    const int status = pclose(pipe);
    if (!read_ok || status != 0) return 1.0;

    try {
        const double utilization =
            std::stod(std::string(buffer));
        return std::clamp(1.0 - utilization / 100.0, 0.0, 1.0);
    } catch (...) {
        return 1.0;
    }
}

}  // namespace

struct OnlineScheduler::Impl {
    mutable std::mutex mutex;
    MonitorState state;
    CpuSample previous_cpu;
    std::thread monitor_thread;
    std::atomic_bool running{false};

    void update_once() {
        const auto cpu = cpu_remaining_capacity(previous_cpu);
        const auto cuda = query_cuda_remaining_capacity();

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

    if (root["level_selection"]) {
        const auto node = root["level_selection"];
        if (node["alpha"]) config.alpha = node["alpha"].as<double>();
        if (node["beta"]) config.beta = node["beta"].as<double>();
    }

    if (root["recovery"]) {
        const auto node = root["recovery"];
        if (node["gamma"]) config.gamma = node["gamma"].as<double>();
        if (node["enabled"]) config.recovery_enabled = node["enabled"].as<bool>();
    }

    // experiments.yaml owns the monitoring sampling interval. Keep the
    // scheduler self-contained if that file is absent.
    if (root["resource_selection"] &&
        root["resource_selection"]["sample_interval_ms"]) {
        config.resource_sample_interval_ms =
            root["resource_selection"]["sample_interval_ms"].as<std::uint32_t>();
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

    impl_->previous_cpu = read_cpu_sample();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->state = MonitorState{};
    }

    impl_->running = true;
    impl_->monitor_thread = std::thread([this]() {
        while (impl_->running) {
            impl_->update_once();

            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    config_.resource_sample_interval_ms));
        }
    });
}

void OnlineScheduler::stop() {
    if (!impl_->running) return;

    impl_->running = false;
    if (impl_->monitor_thread.joinable()) {
        impl_->monitor_thread.join();
    }
}

double OnlineScheduler::remaining_capacity(
    RuntimeResource resource) const {

    std::lock_guard<std::mutex> lock(impl_->mutex);
    switch (resource) {
        case RuntimeResource::CPU:
            return impl_->state.cpu_remaining;
        case RuntimeResource::CUDA:
            return impl_->state.cuda_remaining;
        case RuntimeResource::Auto:
            break;
    }
    return 1.0;
}

SchedulerDecision OnlineScheduler::select_first_resource(
    const SchedulerSegmentCosts& next) const {

    SchedulerDecision best;

    for (const auto candidate :
         {RuntimeResource::CPU, RuntimeResource::CUDA}) {
        const auto cost =
            candidate == RuntimeResource::CPU ? next.cpu : next.cuda;
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

    if (!std::isfinite(best.score)) {
        throw std::runtime_error(
            "No supported resource is available for the next segment");
    }

    return best;
}

SchedulerDecision OnlineScheduler::select_next_resource(
    const SchedulerSegmentCosts& current,
    RuntimeResource current_resource,
    double current_measured_ms,
    const SchedulerSegmentCosts& next) const {

    SchedulerDecision best;

    double current_expected = std::numeric_limits<double>::infinity();
    if (current_resource == RuntimeResource::CPU) {
        current_expected = current.cpu.expected_ms;
    } else if (current_resource == RuntimeResource::CUDA) {
        current_expected = current.cuda.expected_ms;
    }

    const double current_degradation =
        (std::isfinite(current_expected) && current_expected > 0.0 &&
         std::isfinite(current_measured_ms) && current_measured_ms > 0.0)
            ? current_measured_ms / current_expected
            : 1.0;

    for (const auto candidate :
         {RuntimeResource::CPU, RuntimeResource::CUDA}) {
        const auto cost =
            candidate == RuntimeResource::CPU ? next.cpu : next.cuda;
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
        if (score < best.score) {
            best = SchedulerDecision{
                candidate, score, degradation, capacity};
        }
    }

    if (!std::isfinite(best.score)) {
        throw std::runtime_error(
            "No supported resource is available for the next segment");
    }

    return best;
}

SchedulerDecision OnlineScheduler::select_recovery_resource(
    const SchedulerSegmentCosts& current,
    RuntimeResource current_resource) const {

    SchedulerDecision best;

    for (const auto candidate :
         {RuntimeResource::CPU, RuntimeResource::CUDA}) {
        if (candidate == current_resource) continue;

        const auto cost =
            candidate == RuntimeResource::CPU ? current.cpu : current.cuda;
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

}  // namespace flexon::online
