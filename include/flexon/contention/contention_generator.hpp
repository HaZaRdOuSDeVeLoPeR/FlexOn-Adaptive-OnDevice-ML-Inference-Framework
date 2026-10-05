#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace flexon::contention {

enum class Resource : std::uint8_t {
    CPU,
    GPU,
    DRAM,
    VRAM
};

struct Targets {
    double cpu_percent{0.0};
    double gpu_percent{0.0};
    double dram_percent{0.0};
    double vram_percent{0.0};
};

struct BandwidthCeilings {
    double dram_gbps{0.0};
    double vram_gbps{0.0};
};

struct Config {
    Targets targets{};
    BandwidthCeilings bandwidth{};

    // Fraction of the configured benchmark ceiling that is considered the
    // safe maximum. The requested target percentages are applied to this
    // effective ceiling rather than directly to the raw benchmark value.
    double safety_factor{0.85};

    std::chrono::milliseconds monitor_interval{100};
    std::chrono::milliseconds control_interval{500};
    std::chrono::milliseconds stabilization_window{2000};

    double utilization_deadband_percent{2.0};
    double bandwidth_deadband_percent{2.0};
    double max_duty_step{0.05};

    unsigned cpu_workers{0}; // zero means hardware_concurrency().
    int cuda_device_id{0};

    bool verbose{true};
};

struct MonitorState {
    double cpu_utilization_percent{0.0};
    double gpu_utilization_percent{0.0};

    double dram_bandwidth_gbps{0.0};
    double vram_bandwidth_gbps{0.0};

    double dram_utilization_percent{0.0};
    double vram_utilization_percent{0.0};

    bool gpu_available{false};
    bool gpu_monitor_valid{false};

    std::chrono::steady_clock::time_point timestamp{};
};

/**
 * Closed-loop background contention generator.
 *
 * The generator owns independent CPU, GPU, DRAM and VRAM stressors. A single
 * monitor thread publishes the latest resource measurements; each stressor
 * independently adjusts its duty cycle toward its requested target.
 *
 * This class intentionally has no FlexOn scheduler/profiler dependencies so
 * it can be used both by standalone experiments and, later, by the offline
 * profiler.
 */
class ContentionGenerator {
public:
    explicit ContentionGenerator(Config config);
    ~ContentionGenerator();

    ContentionGenerator(const ContentionGenerator&) = delete;
    ContentionGenerator& operator=(const ContentionGenerator&) = delete;

    void start();
    void stop();

    bool running() const noexcept;
    bool wait_until_stable(std::chrono::milliseconds timeout) const;

    MonitorState state() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flexon::contention
