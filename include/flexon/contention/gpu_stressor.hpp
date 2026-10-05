#pragma once

#include <atomic>
#include <cstddef>
#include <chrono>
#include <thread>

namespace flexon::contention {

struct VramBenchmarkResult {
    double bandwidth_gbps{0.0};
    std::size_t buffer_bytes{0};
    int warmup_iterations{0};
    int measured_iterations{0};
    double elapsed_ms{0.0};
};

/**
 * Measure sustainable device-to-device VRAM bandwidth using a workload
 * equivalent to a large contiguous tensor copy: one device-memory read and
 * one device-memory write per byte copied. The source buffer is initialized
 * with randomized data before timing.
 */
VramBenchmarkResult benchmark_vram_bandwidth(
    int device_id,
    std::size_t requested_buffer_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL,
    int warmup_iterations = 10,
    int measured_iterations = 100);


class GpuStressor {
public:
    GpuStressor(double target_percent,
                std::chrono::milliseconds control_interval,
                double deadband_percent,
                double max_duty_step,
                const std::atomic<double>& measured_utilization,
                int device_id);
    ~GpuStressor();

    void start();
    void stop();
    double duty() const noexcept;

private:
    void loop();

    double target_percent_;
    std::chrono::milliseconds control_interval_;
    double deadband_percent_;
    double max_duty_step_;
    const std::atomic<double>& measured_utilization_;
    int device_id_;

    std::atomic<bool> stop_requested_{false};
    std::atomic<double> duty_{0.5};
    std::thread thread_;
};

class VramStressor {
public:
    VramStressor(double target_percent,
                 double safe_ceiling_gbps,
                 std::chrono::milliseconds control_interval,
                 double deadband_percent,
                 double max_duty_step,
                 std::atomic<double>& published_bandwidth,
                 int device_id);
    ~VramStressor();

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
    int device_id_;

    std::atomic<bool> stop_requested_{false};
    std::atomic<double> duty_{0.5};
    std::thread thread_;
};

}  // namespace flexon::contention
