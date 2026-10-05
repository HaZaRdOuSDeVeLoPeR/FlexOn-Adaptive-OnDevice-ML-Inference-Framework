#include <flexon/contention/contention_generator.hpp>
#include <flexon/contention/bandwidth_stressor.hpp>
#include <flexon/contention/cpu_stressor.hpp>
#include <flexon/contention/gpu_stressor.hpp>
#include <flexon/contention/monitor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace flexon::contention {

namespace {

void validate_targets(const Config& config) {
    const auto valid = [](double value) {
        return std::isfinite(value) && value >= 0.0 && value <= 100.0;
    };

    if (!valid(config.targets.cpu_percent) ||
        !valid(config.targets.gpu_percent) ||
        !valid(config.targets.dram_percent) ||
        !valid(config.targets.vram_percent)) {
        throw std::invalid_argument("Contention targets must be in [0, 100]");
    }

    if (!std::isfinite(config.safety_factor) ||
        config.safety_factor <= 0.0 || config.safety_factor > 1.0) {
        throw std::invalid_argument("Contention safety_factor must be in (0, 1]");
    }
}

}  // namespace

class ContentionGenerator::Impl {
public:
    explicit Impl(Config config)
        : config(std::move(config)),
          monitor(this->config.monitor_interval, this->config.cuda_device_id) {}

    Config config;
    ResourceMonitor monitor;
    std::unique_ptr<CpuStressor> cpu;
    std::unique_ptr<GpuStressor> gpu;
    std::unique_ptr<DramStressor> dram;
    std::unique_ptr<VramStressor> vram;

    std::atomic<bool> running{false};
    std::thread logger;
};

ContentionGenerator::ContentionGenerator(Config config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    validate_targets(impl_->config);

}

ContentionGenerator::~ContentionGenerator() {
    stop();
}

void ContentionGenerator::start() {
    if (impl_->running.exchange(true)) {
        return;
    }

    // Measure bandwidth ceilings on the actual machine before starting any
    // background stressor. This keeps the calibration independent of the
    // requested contention level and removes machine-specific constants from
    // hardware.yaml.
    try {
        if (impl_->config.targets.dram_percent > 0.0) {
            const auto result = benchmark_dram_bandwidth();
            if (!(result.bandwidth_gbps > 0.0) || !std::isfinite(result.bandwidth_gbps)) {
                throw std::runtime_error("DRAM bandwidth calibration returned an invalid ceiling");
            }
            impl_->config.bandwidth.dram_gbps = result.bandwidth_gbps;

            if (impl_->config.verbose) {
                std::cout << std::fixed << std::setprecision(3)
                          << "[contention] measured DRAM bandwidth ceiling: "
                          << result.bandwidth_gbps << " GB/s"
                          << " (working_set="
                          << static_cast<double>(result.working_set_bytes) /
                                 (1024.0 * 1024.0 * 1024.0)
                          << " GiB, workers=" << result.workers
                          << ", measured_iterations=" << result.measured_iterations
                          << ", elapsed=" << result.elapsed_ms << " ms)\n";
            }
        }

        if (impl_->config.targets.vram_percent > 0.0) {
            const auto result = benchmark_vram_bandwidth(
                impl_->config.cuda_device_id);
            if (!(result.bandwidth_gbps > 0.0) || !std::isfinite(result.bandwidth_gbps)) {
                throw std::runtime_error("VRAM bandwidth calibration returned an invalid ceiling");
            }
            impl_->config.bandwidth.vram_gbps = result.bandwidth_gbps;

            if (impl_->config.verbose) {
                std::cout << std::fixed << std::setprecision(3)
                          << "[contention] measured VRAM bandwidth ceiling: "
                          << result.bandwidth_gbps << " GB/s"
                          << " (buffer="
                          << static_cast<double>(result.buffer_bytes) / (1024.0 * 1024.0)
                          << " MiB, measured_iterations="
                          << result.measured_iterations << ")\n";
            }
        }
    } catch (...) {
        impl_->running.store(false);
        throw;
    }

    impl_->monitor.start();

    if (impl_->config.targets.gpu_percent > 0.0 ||
        impl_->config.targets.vram_percent > 0.0) {
        const auto state = impl_->monitor.state();
        if (!state.gpu_available || !state.gpu_monitor_valid) {
            impl_->running.store(false);
            impl_->monitor.stop();
            throw std::runtime_error(
                "GPU contention requested but NVML GPU utilization monitoring "
                "is unavailable for the selected device");
        }
    }

    if (impl_->config.targets.cpu_percent > 0.0) {
        impl_->cpu = std::make_unique<CpuStressor>(
            impl_->config.cpu_workers,
            impl_->config.targets.cpu_percent,
            impl_->config.control_interval,
            impl_->config.utilization_deadband_percent,
            impl_->config.max_duty_step,
            impl_->monitor.cpu_utilization());
        impl_->cpu->start();
    }

    if (impl_->config.targets.gpu_percent > 0.0) {
        impl_->gpu = std::make_unique<GpuStressor>(
            impl_->config.targets.gpu_percent,
            impl_->config.control_interval,
            impl_->config.utilization_deadband_percent,
            impl_->config.max_duty_step,
            impl_->monitor.gpu_utilization(),
            impl_->config.cuda_device_id);
        impl_->gpu->start();
    }

    if (impl_->config.targets.dram_percent > 0.0) {
        const double ceiling =
            impl_->config.bandwidth.dram_gbps * impl_->config.safety_factor;
        impl_->dram = std::make_unique<DramStressor>(
            impl_->config.targets.dram_percent,
            ceiling,
            impl_->config.control_interval,
            impl_->config.bandwidth_deadband_percent,
            impl_->config.max_duty_step,
            impl_->monitor.dram_bandwidth());
        impl_->dram->start();
    }

    if (impl_->config.targets.vram_percent > 0.0) {
        const double ceiling =
            impl_->config.bandwidth.vram_gbps * impl_->config.safety_factor;
        impl_->vram = std::make_unique<VramStressor>(
            impl_->config.targets.vram_percent,
            ceiling,
            impl_->config.control_interval,
            impl_->config.bandwidth_deadband_percent,
            impl_->config.max_duty_step,
            impl_->monitor.vram_bandwidth(),
            impl_->config.cuda_device_id);
        impl_->vram->start();
    }

    if (impl_->config.verbose) {
        impl_->logger = std::thread([this] {
            while (impl_->running.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (!impl_->running.load(std::memory_order_relaxed)) {
                    break;
                }

                const auto state = impl_->monitor.state();
                const double dram_ceiling =
                    impl_->config.bandwidth.dram_gbps * impl_->config.safety_factor;
                const double vram_ceiling =
                    impl_->config.bandwidth.vram_gbps * impl_->config.safety_factor;
                const double dram_percent =
                    dram_ceiling > 0.0
                        ? 100.0 * state.dram_bandwidth_gbps / dram_ceiling
                        : 0.0;
                const double vram_percent =
                    vram_ceiling > 0.0
                        ? 100.0 * state.vram_bandwidth_gbps / vram_ceiling
                        : 0.0;

                std::cout << std::fixed << std::setprecision(1)
                          << "[contention] cpu=" << state.cpu_utilization_percent
                          << "% gpu=" << state.gpu_utilization_percent
                          << "% dram=" << state.dram_bandwidth_gbps
                          << " GB/s (" << dram_percent << "%) vram="
                          << state.vram_bandwidth_gbps << " GB/s ("
                          << vram_percent << "%)" << std::endl;
            }
        });
    }
}

void ContentionGenerator::stop() {
    if (!impl_->running.exchange(false)) {
        return;
    }

    if (impl_->cpu) {
        impl_->cpu->stop();
    }
    if (impl_->gpu) {
        impl_->gpu->stop();
    }
    if (impl_->dram) {
        impl_->dram->stop();
    }
    if (impl_->vram) {
        impl_->vram->stop();
    }

    if (impl_->logger.joinable()) {
        impl_->logger.join();
    }

    impl_->monitor.stop();

    impl_->cpu.reset();
    impl_->gpu.reset();
    impl_->dram.reset();
    impl_->vram.reset();
}

bool ContentionGenerator::running() const noexcept {
    return impl_->running.load(std::memory_order_relaxed);
}

bool ContentionGenerator::wait_until_stable(
    std::chrono::milliseconds timeout) const {
    const auto start = std::chrono::steady_clock::now();
    auto stable_since = start;

    const auto within = [](double target, double measured, double deadband) {
        return std::abs(target - measured) <= deadband;
    };

    while (std::chrono::steady_clock::now() - start < timeout) {
        const auto state = impl_->monitor.state();
        bool stable = true;

        if (impl_->config.targets.cpu_percent > 0.0) {
            stable &= within(impl_->config.targets.cpu_percent,
                             state.cpu_utilization_percent,
                             impl_->config.utilization_deadband_percent);
        }
        if (impl_->config.targets.gpu_percent > 0.0) {
            stable &= within(impl_->config.targets.gpu_percent,
                             state.gpu_utilization_percent,
                             impl_->config.utilization_deadband_percent);
        }
        if (impl_->config.targets.dram_percent > 0.0) {
            const double ceiling =
                impl_->config.bandwidth.dram_gbps * impl_->config.safety_factor;
            const double measured =
                ceiling > 0.0
                    ? 100.0 * state.dram_bandwidth_gbps / ceiling
                    : 0.0;
            stable &= within(impl_->config.targets.dram_percent,
                             measured,
                             impl_->config.bandwidth_deadband_percent);
        }
        if (impl_->config.targets.vram_percent > 0.0) {
            const double ceiling =
                impl_->config.bandwidth.vram_gbps * impl_->config.safety_factor;
            const double measured =
                ceiling > 0.0
                    ? 100.0 * state.vram_bandwidth_gbps / ceiling
                    : 0.0;
            stable &= within(impl_->config.targets.vram_percent,
                             measured,
                             impl_->config.bandwidth_deadband_percent);
        }

        if (stable) {
            if (std::chrono::steady_clock::now() - stable_since >=
                impl_->config.stabilization_window) {
                return true;
            }
        } else {
            stable_since = std::chrono::steady_clock::now();
        }

        std::this_thread::sleep_for(impl_->config.monitor_interval);
    }

    return false;
}

MonitorState ContentionGenerator::state() const noexcept {
    return impl_->monitor.state();
}

}  // namespace flexon::contention
