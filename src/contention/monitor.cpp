#include <flexon/contention/monitor.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>

#include <nvml.h>

namespace flexon::contention {

namespace {

// NVML is deliberately kept initialized for the lifetime of the process.
// Repeated nvmlInit_v2()/nvmlShutdown() cycles around CUDA workloads are
// unsafe on some WSL/NVIDIA driver combinations and can corrupt the process
// heap during teardown. The process will release NVML resources on exit.
std::once_flag nvml_init_once;
std::atomic<bool> nvml_process_available{false};

void initialize_nvml_once() noexcept {
    const auto result = nvmlInit_v2();
    nvml_process_available.store(
        result == NVML_SUCCESS, std::memory_order_release);
}

struct CpuTimes {
    std::uint64_t idle{0};
    std::uint64_t total{0};
};

CpuTimes read_cpu_times() {
    std::ifstream in("/proc/stat");
    if (!in) {
        throw std::runtime_error("Failed to open /proc/stat");
    }

    std::string line;
    std::getline(in, line);
    std::istringstream stream(line);

    std::string cpu;
    std::uint64_t user = 0;
    std::uint64_t nice = 0;
    std::uint64_t system = 0;
    std::uint64_t idle = 0;
    std::uint64_t iowait = 0;
    std::uint64_t irq = 0;
    std::uint64_t softirq = 0;
    std::uint64_t steal = 0;

    stream >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    const std::uint64_t idle_all = idle + iowait;
    const std::uint64_t total = user + nice + system + idle + iowait +
                                irq + softirq + steal;
    return {idle_all, total};
}

}  // namespace

ResourceMonitor::ResourceMonitor(std::chrono::milliseconds interval,
                                 int cuda_device_id)
    : interval_(interval), cuda_device_id_(cuda_device_id) {
    if (interval_.count() <= 0) {
        throw std::invalid_argument("Monitor interval must be positive");
    }
}

ResourceMonitor::~ResourceMonitor() {
    stop();
}

void ResourceMonitor::start() {
    if (thread_.joinable()) {
        return;
    }

    stop_requested_.store(false);

    // Initialize NVML at most once per process and keep it alive until process
    // exit. Failure is not hidden: GPU contention requested by the caller
    // will later reject invalid telemetry.
    std::call_once(nvml_init_once, initialize_nvml_once);
    if (nvml_process_available.load(std::memory_order_acquire)) {
        nvmlDevice_t device{};
        if (nvmlDeviceGetHandleByIndex_v2(
                static_cast<unsigned int>(cuda_device_id_), &device) == NVML_SUCCESS) {
            gpu_available_.store(true);
        }
        // Keep NVML initialized while the watcher thread is active.
        gpu_monitor_valid_.store(gpu_available_.load());
    } else {
        gpu_available_.store(false);
        gpu_monitor_valid_.store(false);
    }

    thread_ = std::thread(&ResourceMonitor::loop, this);
}

void ResourceMonitor::stop() {
    stop_requested_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }

    // Do not call nvmlShutdown() here. NVML is process-lifetime state; keeping
    // it alive avoids the CUDA/NVML teardown-order heap corruption observed
    // after offline batch generation under WSL.
    gpu_monitor_valid_.store(false);
    gpu_available_.store(false);
}

void ResourceMonitor::publish_bandwidth(Resource resource, double gbps) noexcept {
    if (resource == Resource::DRAM) {
        dram_bandwidth_.store(std::max(0.0, gbps), std::memory_order_relaxed);
    } else if (resource == Resource::VRAM) {
        vram_bandwidth_.store(std::max(0.0, gbps), std::memory_order_relaxed);
    }
}

MonitorState ResourceMonitor::state() const noexcept {
    MonitorState state;
    state.cpu_utilization_percent = cpu_utilization_.load(std::memory_order_relaxed);
    state.gpu_utilization_percent = gpu_utilization_.load(std::memory_order_relaxed);
    state.dram_bandwidth_gbps = dram_bandwidth_.load(std::memory_order_relaxed);
    state.vram_bandwidth_gbps = vram_bandwidth_.load(std::memory_order_relaxed);
    state.gpu_available = gpu_available_.load(std::memory_order_relaxed);
    state.gpu_monitor_valid = gpu_monitor_valid_.load(std::memory_order_relaxed);
    state.timestamp = std::chrono::steady_clock::now();
    return state;
}

void ResourceMonitor::loop() {
    CpuTimes previous{};
    try {
        previous = read_cpu_times();
    } catch (...) {
        // Keep the watcher alive so GPU telemetry can still be useful.
    }

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(interval_);

        try {
            const CpuTimes current = read_cpu_times();
            const auto total_delta = current.total - previous.total;
            const auto idle_delta = current.idle - previous.idle;
            if (total_delta > 0) {
                const double utilization =
                    100.0 * (1.0 - static_cast<double>(idle_delta) /
                                      static_cast<double>(total_delta));
                cpu_utilization_.store(
                    std::clamp(utilization, 0.0, 100.0),
                    std::memory_order_relaxed);
            }
            previous = current;
        } catch (...) {
            // Retain the last valid CPU measurement.
        }

        if (!gpu_available_.load(std::memory_order_relaxed)) {
            continue;
        }

        nvmlDevice_t device{};
        if (nvmlDeviceGetHandleByIndex_v2(
                static_cast<unsigned int>(cuda_device_id_), &device) != NVML_SUCCESS) {
            gpu_monitor_valid_.store(false, std::memory_order_relaxed);
            continue;
        }

        nvmlUtilization_t utilization{};
        const auto result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result == NVML_SUCCESS) {
            gpu_utilization_.store(
                static_cast<double>(utilization.gpu),
                std::memory_order_relaxed);
            gpu_monitor_valid_.store(true, std::memory_order_relaxed);
        } else {
            gpu_monitor_valid_.store(false, std::memory_order_relaxed);
        }
    }
}

}  // namespace flexon::contention
