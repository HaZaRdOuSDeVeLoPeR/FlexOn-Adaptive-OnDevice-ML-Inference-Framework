#pragma once

#include <flexon/core/types.hpp>
#include <flexon/online/runtime_engine.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>

namespace flexon::online {

/**
 * Runtime scheduler parameters corresponding to FlexOn's online policy.
 *
 * Defaults match the values reported in the paper: alpha=1.2,
 * beta=1.6 and gamma=1.3.
 */
struct SchedulerConfig {
    double alpha{1.2};
    double beta{1.6};
    double gamma{1.3};
    bool recovery_enabled{true};
    std::uint32_t resource_sample_interval_ms{50};
};

struct SchedulerResourceCost {
    bool supported{false};
    double expected_ms{std::numeric_limits<double>::infinity()};
};

struct SchedulerSegmentCosts {
    SchedulerResourceCost cpu;
    SchedulerResourceCost cuda;
};

struct SchedulerDecision {
    RuntimeResource resource{RuntimeResource::CPU};
    double score{std::numeric_limits<double>::infinity()};
    double degradation{1.0};
    double remaining_capacity{1.0};
};

class OnlineScheduler {
public:
    explicit OnlineScheduler(SchedulerConfig config = {});
    ~OnlineScheduler();

    OnlineScheduler(const OnlineScheduler&) = delete;
    OnlineScheduler& operator=(const OnlineScheduler&) = delete;

    static SchedulerConfig load_config(
        const std::filesystem::path& path);

    void start();
    void stop();

    SchedulerDecision select_first_resource(
        const SchedulerSegmentCosts& next) const;

    SchedulerDecision select_next_resource(
        const SchedulerSegmentCosts& current,
        RuntimeResource current_resource,
        double current_measured_ms,
        const SchedulerSegmentCosts& next) const;

    // Recovery is evaluated while the current segment is executing. The
    // returned decision is the best alternative resource according to the
    // paper's Eq. (5) right-hand side. Its score is d_r * C_r(s_i).
    SchedulerDecision select_recovery_resource(
        const SchedulerSegmentCosts& current,
        RuntimeResource current_resource) const;

    bool should_trigger_recovery(
        double current_elapsed_ms,
        double alternative_score) const;

    std::uint32_t select_next_level(
        std::uint32_t current_level,
        std::uint32_t max_level,
        double measured_period_ms,
        double expected_period_ms) const;

    double remaining_capacity(RuntimeResource resource) const;

    const SchedulerConfig& config() const noexcept { return config_; }

private:
    struct Impl;
    SchedulerConfig config_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flexon::online
