#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>

#include <flexon/contention/contention_generator.hpp>

namespace flexon::contention {

struct DramBenchmarkResult {
    double bandwidth_gbps{0.0};
    std::size_t working_set_bytes{0};
    int warmup_iterations{0};
    int measured_iterations{0};
    double elapsed_ms{0.0};
    unsigned workers{0};
};

/**
 * Measure sustainable DRAM bandwidth using the same streaming A+B->C
 * workload used by DramStressor. The benchmark uses randomized buffer
 * contents, a cache-busting working set, multiple independent worker streams,
 * and excludes allocation/initialization/warmup from the timed region.
 */
DramBenchmarkResult benchmark_dram_bandwidth(
    std::size_t requested_total_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL,
    int warmup_iterations = 5,
    int measured_iterations = 50,
    unsigned workers = 0);

class DramStressor {
public:
    DramStressor(double target_percent,
                 double safe_ceiling_gbps,
                 std::chrono::milliseconds control_interval,
                 double deadband_percent,
                 double max_duty_step,
                 std::atomic<double>& published_bandwidth);
    ~DramStressor();

    void start();
    void stop();
    double duty() const noexcept;

private:
    void loop();

    double target_percent_;
    double ceiling_gbps_;
    std::chrono::milliseconds control_interval_;
    double deadband_percent_;
    double max_duty_step_;
    std::atomic<double>& published_bandwidth_;

    std::atomic<bool> stop_requested_{false};
    std::atomic<double> duty_{0.5};
    std::thread thread_;
};

}  // namespace flexon::contention
