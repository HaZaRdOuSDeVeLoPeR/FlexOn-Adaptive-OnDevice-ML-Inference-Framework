#include <flexon/contention/bandwidth_stressor.hpp>
#include <flexon/contention/controller.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <iostream>

namespace flexon::contention {

namespace {

inline std::uint64_t next_random(std::uint64_t& state) noexcept {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

inline float random_float(std::uint64_t& state) noexcept {
    return static_cast<float>(next_random(state) & 0xFFFFU) / 65535.0F;
}

void initialize_buffers(std::vector<float>& a,
                        std::vector<float>& b,
                        std::uint64_t& rng) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = random_float(rng);
        b[i] = random_float(rng);
    }
}

// Tell GCC/Clang that the output buffer escapes the function. This prevents
// the optimizer from deleting the streaming stores while avoiding a volatile
// store for every element, which would unnecessarily distort the bandwidth
// measurement.
inline void compiler_escape(const void* pointer) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(pointer) : "memory");
#else
    (void)pointer;
#endif
}

std::uint64_t streaming_pass(std::vector<float>& a,
                             std::vector<float>& b) {
    const std::size_t bytes = a.size() * sizeof(float);

    std::memcpy(b.data(), a.data(), bytes);

    compiler_escape(b.data());

    // DRAM traffic:
    //   read A + write B
    return static_cast<std::uint64_t>(bytes) * 2ULL;
}

unsigned default_dram_workers() noexcept {
    return std::max(1U, std::thread::hardware_concurrency());
}

void dram_worker(std::atomic<bool>& stop_requested,
                 std::atomic<double>& duty,
                 std::atomic<std::uint64_t>& bytes_processed,
                 unsigned worker_index) {
    // 64 MiB per array => 192 MiB logical working set per worker. The total
    // stressor footprint remains modest while staying well above the L3 cache.
    constexpr std::size_t elements = 16ULL * 1024ULL * 1024ULL;

    std::vector<float> a(elements);
    std::vector<float> b(elements);

    std::uint64_t rng =
        0x9E3779B97F4A7C15ULL ^
        (static_cast<std::uint64_t>(worker_index) + 1ULL) * 0xBF58476D1CE4E5B9ULL;
    initialize_buffers(a, b, rng);

    constexpr auto quantum = std::chrono::milliseconds(100);

    while (!stop_requested.load(std::memory_order_relaxed)) {
        const double duty_cycle = duty.load(std::memory_order_relaxed);

        if (duty_cycle > 0.0) {
            const auto active = std::chrono::duration_cast<std::chrono::milliseconds>(
                quantum * duty_cycle);
            const auto active_duration = std::max(active, std::chrono::milliseconds(1));
            const auto end = std::chrono::steady_clock::now() + active_duration;

            while (std::chrono::steady_clock::now() < end &&
                   !stop_requested.load(std::memory_order_relaxed)) {
                bytes_processed.fetch_add(
                    streaming_pass(a, b),
                    std::memory_order_relaxed);
            }
        }

        const auto inactive = std::chrono::duration_cast<std::chrono::milliseconds>(
            quantum * (1.0 - duty_cycle));
        if (inactive.count() > 0) {
            std::this_thread::sleep_for(inactive);
        }
    }
}

struct BenchmarkBarrier {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned ready{0};
    unsigned warmup_done{0};
    bool warmup_start{false};
    bool measure_start{false};
};

}  // namespace

DramBenchmarkResult benchmark_dram_bandwidth(
    std::size_t requested_total_bytes,
    int warmup_iterations,
    int measured_iterations,
    unsigned workers) {
    if (requested_total_bytes == 0 || warmup_iterations < 0 ||
        measured_iterations <= 0) {
        throw std::invalid_argument("Invalid DRAM benchmark parameters");
    }

    if (workers == 0) {
        workers = default_dram_workers();
    }
    if (workers == 0) {
        workers = 1;
    }

    // The benchmark uses the same streaming A+B->C operation as the runtime
    // DRAM stressor. The default 1 GiB total logical working set is far beyond
    // the target machine's LLC while remaining modest for a normal desktop.
    constexpr std::size_t min_total_bytes = 256ULL * 1024ULL * 1024ULL;
    constexpr std::size_t alignment = 64ULL * 1024ULL * 1024ULL;
    constexpr std::size_t min_per_worker_bytes = 16ULL * 1024ULL * 1024ULL;

    std::size_t total_bytes =
        std::max(requested_total_bytes, min_total_bytes);
    total_bytes = (total_bytes / alignment) * alignment;
    if (total_bytes < min_total_bytes) {
        total_bytes = min_total_bytes;
    }

    std::size_t per_worker_bytes =
        (total_bytes / workers / 2ULL / sizeof(float)) * sizeof(float);

    // If a very high worker count would make each stream too small, reduce the
    // worker count until each worker still has a cache-busting working set.
    while (workers > 1 && per_worker_bytes < min_per_worker_bytes) {
        --workers;
        per_worker_bytes =
            (total_bytes / workers / 2ULL / sizeof(float)) * sizeof(float);
    }

    if (per_worker_bytes < min_per_worker_bytes) {
        throw std::runtime_error(
            "DRAM benchmark working set is too small for the available workers");
    }

    BenchmarkBarrier barrier;
    std::atomic<std::uint64_t> bytes_processed{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (unsigned worker_index = 0; worker_index < workers; ++worker_index) {
        threads.emplace_back([&, worker_index] {
            const std::size_t elements = per_worker_bytes / sizeof(float);

            std::vector<float> a(elements);
            std::vector<float> b(elements);

            std::uint64_t rng =
                0xD1B54A32D192ED03ULL ^
                (static_cast<std::uint64_t>(worker_index) + 1ULL) *
                    0x94D049BB133111EBULL;
            initialize_buffers(a, b, rng);

            {
                std::unique_lock lock(barrier.mutex);
                ++barrier.ready;
                barrier.cv.notify_all();
                barrier.cv.wait(lock, [&] { return barrier.warmup_start; });
            }

            for (int i = 0; i < warmup_iterations; ++i) {
                (void)streaming_pass(a, b);
            }

            {
                std::unique_lock lock(barrier.mutex);
                ++barrier.warmup_done;
                barrier.cv.notify_all();
                barrier.cv.wait(lock, [&] { return barrier.measure_start; });
            }

            for (int i = 0; i < measured_iterations; ++i) {
                bytes_processed.fetch_add(
                    streaming_pass(a, b),
                    std::memory_order_relaxed);
            }
        });
    }

    {
        std::unique_lock lock(barrier.mutex);
        barrier.cv.wait(lock, [&] { return barrier.ready == workers; });
        barrier.warmup_start = true;
        barrier.cv.notify_all();
    }

    {
        std::unique_lock lock(barrier.mutex);
        barrier.cv.wait(lock, [&] { return barrier.warmup_done == workers; });

        // Start the timer immediately before releasing the measured phase.
        // Allocation, initialization and warmup are therefore excluded.
        const auto start = std::chrono::steady_clock::now();
        barrier.measure_start = true;
        barrier.cv.notify_all();
        lock.unlock();

        for (auto& thread : threads) {
            thread.join();
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed_seconds =
            std::chrono::duration<double>(end - start).count();
        const double gbps = elapsed_seconds > 0.0
                                ? static_cast<double>(bytes_processed.load()) /
                                      elapsed_seconds / 1.0e9
                                : 0.0;

        return DramBenchmarkResult{
            gbps,
            per_worker_bytes * workers * 2ULL,
            warmup_iterations,
            measured_iterations,
            elapsed_seconds * 1000.0,
            workers};
    }
}

DramStressor::DramStressor(
    double target_percent,
    double safe_ceiling_gbps,
    std::chrono::milliseconds control_interval,
    double deadband_percent,
    double max_duty_step,
    std::atomic<double>& published_bandwidth)
    : target_percent_(std::clamp(target_percent, 0.0, 100.0)),
      ceiling_gbps_(safe_ceiling_gbps),
      control_interval_(control_interval),
      deadband_percent_(deadband_percent),
      max_duty_step_(max_duty_step),
      published_bandwidth_(published_bandwidth) {
    duty_.store(target_percent_ > 0.0 ? target_percent_ / 100.0 : 0.0);
}

DramStressor::~DramStressor() {
    stop();
}

void DramStressor::start() {
    if (thread_.joinable() || target_percent_ <= 0.0) {
        return;
    }
    stop_requested_.store(false);
    thread_ = std::thread(&DramStressor::loop, this);
}

void DramStressor::stop() {
    stop_requested_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
}

double DramStressor::duty() const noexcept {
    return duty_.load(std::memory_order_relaxed);
}

void DramStressor::loop() {
    const unsigned workers = default_dram_workers();
    std::vector<std::thread> worker_threads;
    worker_threads.reserve(workers);

    std::atomic<std::uint64_t> bytes_processed{0};
    for (unsigned i = 0; i < workers; ++i) {
        worker_threads.emplace_back(
            dram_worker,
            std::ref(stop_requested_),
            std::ref(duty_),
            std::ref(bytes_processed),
            i);
    }

    DutyController controller(deadband_percent_, max_duty_step_, duty());
    auto window_start = std::chrono::steady_clock::now();

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(control_interval_);

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - window_start).count();
        const auto bytes = bytes_processed.exchange(0, std::memory_order_relaxed);

        const double gbps =
            elapsed > 0.0
                ? static_cast<double>(bytes) / elapsed / 1.0e9
                : 0.0;

        published_bandwidth_.store(gbps, std::memory_order_relaxed);

        const double utilization =
            ceiling_gbps_ > 0.0
                ? 100.0 * gbps / ceiling_gbps_
                : 0.0;
        controller.update(target_percent_, utilization);
        duty_.store(controller.duty(), std::memory_order_relaxed);
        window_start = now;
    }

    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

}  // namespace flexon::contention
