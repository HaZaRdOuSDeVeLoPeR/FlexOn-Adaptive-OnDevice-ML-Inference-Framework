#include <flexon/contention/cpu_stressor.hpp>
#include <flexon/contention/controller.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>

namespace flexon::contention {

namespace {

inline double random_value(std::uint64_t& state) noexcept {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(state & 0xFFFFFFFFULL) /
           static_cast<double>(std::numeric_limits<std::uint32_t>::max());
}

void compute_kernel(std::uint64_t& state) noexcept {
    static volatile double sink = 0.0;
    // Randomized inputs prevent the compiler from collapsing the workload to
    // a fixed expression. FMA keeps the workload predominantly arithmetic.
    double a = random_value(state) * 2.0 - 1.0;
    double b = random_value(state) * 2.0 - 1.0;
    double c = random_value(state) * 2.0 - 1.0;

    for (int i = 0; i < 256; ++i) {
        a = std::fma(a, b, c);
        b = std::fma(b, c, a);
        c = std::fma(c, a, b);
        a += random_value(state) * 1.0e-6;
    }

    // Prevent the compiler from treating the computation as dead.
    sink = a + b + c + static_cast<double>(state & 0xFFU);
    (void)sink;
}

}  // namespace

CpuStressor::CpuStressor(unsigned workers,
                         double target_percent,
                         std::chrono::milliseconds control_interval,
                         double deadband_percent,
                         double max_duty_step,
                         const std::atomic<double>& measured_utilization)
    : workers_(workers == 0 ? std::thread::hardware_concurrency() : workers),
      target_percent_(std::clamp(target_percent, 0.0, 100.0)),
      control_interval_(control_interval),
      deadband_percent_(deadband_percent),
      max_duty_step_(max_duty_step),
      measured_utilization_(measured_utilization) {
    if (workers_ == 0) {
        workers_ = 1;
    }
    duty_.store(target_percent_ > 0.0 ? target_percent_ / 100.0 : 0.0);
}

CpuStressor::~CpuStressor() {
    stop();
}

void CpuStressor::start() {
    if (!threads_.empty() || target_percent_ <= 0.0) {
        return;
    }
    stop_requested_.store(false);
    for (unsigned i = 0; i < workers_; ++i) {
        threads_.emplace_back(&CpuStressor::worker_loop, this, i);
    }
}

void CpuStressor::stop() {
    stop_requested_.store(true);
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads_.clear();
}

double CpuStressor::duty() const noexcept {
    return duty_.load(std::memory_order_relaxed);
}

void CpuStressor::worker_loop(unsigned worker_index) {
    std::uint64_t rng =
        0x9E3779B97F4A7C15ULL ^
        (static_cast<std::uint64_t>(worker_index) + 1ULL) * 0xBF58476D1CE4E5B9ULL;

    DutyController controller(deadband_percent_, max_duty_step_, duty());
    auto last_control = std::chrono::steady_clock::now();

    constexpr auto quantum = std::chrono::milliseconds(10);

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        const double duty_cycle = duty_.load(std::memory_order_relaxed);

        if (duty_cycle > 0.0) {
            const auto active = std::chrono::duration_cast<std::chrono::milliseconds>(
                quantum * duty_cycle);
            const auto active_duration = std::max(active, std::chrono::milliseconds(1));
            const auto end = std::chrono::steady_clock::now() + active_duration;
            while (std::chrono::steady_clock::now() < end &&
                   !stop_requested_.load(std::memory_order_relaxed)) {
                compute_kernel(rng);
            }
        }

        const auto inactive = std::chrono::duration_cast<std::chrono::milliseconds>(
            quantum * (1.0 - duty_cycle));
        if (inactive.count() > 0) {
            std::this_thread::sleep_for(inactive);
        }

        const auto now = std::chrono::steady_clock::now();
        if (worker_index == 0 && now - last_control >= control_interval_) {
            const double measured = measured_utilization_.load(std::memory_order_relaxed);
            const double updated = controller.update(target_percent_, measured);
            duty_.store(updated, std::memory_order_relaxed);
            last_control = now;
        }
    }
}

}  // namespace flexon::contention
