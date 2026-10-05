#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <thread>

#include <flexon/contention/contention_generator.hpp>

namespace flexon::contention {

class CpuStressor {
public:
    CpuStressor(unsigned workers,
                double target_percent,
                std::chrono::milliseconds control_interval,
                double deadband_percent,
                double max_duty_step,
                const std::atomic<double>& measured_utilization);
    ~CpuStressor();

    CpuStressor(const CpuStressor&) = delete;
    CpuStressor& operator=(const CpuStressor&) = delete;

    void start();
    void stop();
    double duty() const noexcept;

private:
    void worker_loop(unsigned worker_index);

    unsigned workers_;
    double target_percent_;
    std::chrono::milliseconds control_interval_;
    double deadband_percent_;
    double max_duty_step_;
    const std::atomic<double>& measured_utilization_;

    std::atomic<bool> stop_requested_{false};
    std::atomic<double> duty_{0.5};
    std::vector<std::thread> threads_;
};

}  // namespace flexon::contention
