#pragma once

#include <atomic>
#include <chrono>
#include <thread>

#include <flexon/contention/contention_generator.hpp>

namespace flexon::contention {

class ResourceMonitor {
public:
    ResourceMonitor(std::chrono::milliseconds interval, int cuda_device_id);
    ~ResourceMonitor();

    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;

    void start();
    void stop();

    MonitorState state() const noexcept;

    void publish_bandwidth(Resource resource, double gbps) noexcept;

    const std::atomic<double>& cpu_utilization() const noexcept {
        return cpu_utilization_;
    }
    const std::atomic<double>& gpu_utilization() const noexcept {
        return gpu_utilization_;
    }
    std::atomic<double>& dram_bandwidth() noexcept {
        return dram_bandwidth_;
    }
    std::atomic<double>& vram_bandwidth() noexcept {
        return vram_bandwidth_;
    }

private:
    void loop();

    std::chrono::milliseconds interval_;
    int cuda_device_id_;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;

    std::atomic<double> cpu_utilization_{0.0};
    std::atomic<double> gpu_utilization_{0.0};
    std::atomic<double> dram_bandwidth_{0.0};
    std::atomic<double> vram_bandwidth_{0.0};
    std::atomic<bool> gpu_available_{false};
    std::atomic<bool> gpu_monitor_valid_{false};
};

}  // namespace flexon::contention
