#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <flexon/contention/gpu_stressor.hpp>
#include <flexon/contention/controller.hpp>
#include <cuda_runtime.h>


namespace flexon::contention {

namespace {

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ std::uint32_t xorshift32(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ float random_float(uint32_t& state)
{
    state = xorshift32(state);
    return static_cast<float>(state) * (1.0f / 4294967295.0f);
}

__global__ void compute_kernel(float* output,
                               std::uint32_t seed,
                               int iterations) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);

    std::uint32_t state = seed ^ (index * 0x9E3779B9U + 0x85EBCA6BU);
    float a = random_float(state) * 2.0F - 1.0F;
    float b = random_float(state) * 2.0F - 1.0F;
    float c = random_float(state) * 2.0F - 1.0F;

    for (int i = 0; i < iterations; ++i) {
        a = fmaf(a, b, c);
        b = fmaf(b, c, a);
        c = fmaf(c, a, b);

        // Change the arithmetic operands without introducing global-memory
        // traffic into the compute stressor.
        c += random_float(state) * 1.0e-6F;
    }

    output[index] = a + b + c;
}

__global__ void vram_kernel(const float* input_a,
                            const float* input_b,
                            float* output,
                            std::size_t elements,
                            std::uint32_t seed) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= elements) {
        return;
    }

    // The per-thread perturbation prevents a trivially constant access
    // pattern while retaining a bandwidth-dominated kernel.
    std::uint32_t state = seed ^ static_cast<std::uint32_t>(index);
    const float perturbation = random_float(state) * 1.0e-6F;
    output[index] = input_a[index] + input_b[index] + perturbation;
}

}  // namespace

VramBenchmarkResult benchmark_vram_bandwidth(
    int device_id,
    std::size_t requested_buffer_bytes,
    int warmup_iterations,
    int measured_iterations) {
    if (requested_buffer_bytes == 0 || warmup_iterations < 0 || measured_iterations <= 0) {
        throw std::invalid_argument("Invalid VRAM benchmark parameters");
    }

    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    // Keep the benchmark self-contained and safe when it is invoked after a
    // model has already allocated device memory. Two buffers are required.
    // Use at most 45% of currently free VRAM for the benchmark and cap the
    // individual buffer at the requested size.
    constexpr double free_memory_fraction = 0.45;
    constexpr std::size_t min_buffer_bytes = 256ULL * 1024ULL * 1024ULL;
    constexpr std::size_t alignment = 4ULL * 1024ULL * 1024ULL;

    std::size_t buffer_bytes = std::min(
        requested_buffer_bytes,
        static_cast<std::size_t>(
            static_cast<double>(free_bytes) * free_memory_fraction / 2.0));
    buffer_bytes = (buffer_bytes / alignment) * alignment;

    if (buffer_bytes < min_buffer_bytes) {
        throw std::runtime_error(
            "Insufficient free VRAM for the VRAM bandwidth benchmark");
    }

    float* source = nullptr;
    float* destination = nullptr;

    // If allocation fails despite cudaMemGetInfo, progressively reduce the
    // benchmark size rather than making the contention generator unusable.
    while (buffer_bytes >= min_buffer_bytes) {
        const auto source_error = cudaMalloc(&source, buffer_bytes);
        if (source_error == cudaSuccess) {
            const auto destination_error = cudaMalloc(&destination, buffer_bytes);
            if (destination_error == cudaSuccess) {
                break;
            }
            cudaFree(source);
            source = nullptr;
        }

        buffer_bytes /= 2;
        buffer_bytes = (buffer_bytes / alignment) * alignment;
    }

    if (source == nullptr || destination == nullptr) {
        throw std::runtime_error(
            "Unable to allocate VRAM buffers for the bandwidth benchmark");
    }

    try {
        // Randomize the source buffer once. The benchmark itself is a pure
        // device-to-device copy, so random generation is intentionally kept
        // outside the timed region.
        constexpr std::size_t host_chunk_bytes = 16ULL * 1024ULL * 1024ULL;
        const std::size_t host_elements = host_chunk_bytes / sizeof(float);
        std::vector<float> host(host_elements);
        std::mt19937_64 rng(
            static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count()) ^
            static_cast<std::uint64_t>(device_id));
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

        for (float& value : host) {
            value = distribution(rng);
        }

        for (std::size_t offset = 0; offset < buffer_bytes; offset += host_chunk_bytes) {
            const std::size_t float_offset = offset / sizeof(float);
            const std::size_t bytes = std::min(host_chunk_bytes, buffer_bytes - offset);

            check_cuda(cudaMemcpy(source + float_offset,
                                host.data(),
                                bytes,
                                cudaMemcpyHostToDevice),
                    "initialize VRAM benchmark source");
        }
        check_cuda(cudaMemset(destination, 0, buffer_bytes),
                   "initialize VRAM benchmark destination");
        check_cuda(cudaDeviceSynchronize(), "synchronize VRAM benchmark initialization");

        // Warm up the same operation that will be measured. This removes
        // first-use/runtime initialization effects from the bandwidth result.
        for (int i = 0; i < warmup_iterations; ++i) {
            check_cuda(cudaMemcpyAsync(destination,
                                       source,
                                       buffer_bytes,
                                       cudaMemcpyDeviceToDevice),
                       "VRAM benchmark warmup copy");
        }
        check_cuda(cudaDeviceSynchronize(), "synchronize VRAM benchmark warmup");

        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        try {
            check_cuda(cudaEventCreate(&end), "cudaEventCreate end");

            check_cuda(cudaEventRecord(start), "cudaEventRecord start");
            for (int i = 0; i < measured_iterations; ++i) {
                check_cuda(cudaMemcpyAsync(destination,
                                           source,
                                           buffer_bytes,
                                           cudaMemcpyDeviceToDevice),
                           "VRAM benchmark measured copy");
            }
            check_cuda(cudaEventRecord(end), "cudaEventRecord end");
            check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end");

            float elapsed_ms = 0.0F;
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, end),
                       "cudaEventElapsedTime");

            // A device-to-device copy causes one read and one write for each
            // byte copied, matching the 2x accounting used by the Python
            // reference measurement supplied for FlexOn.
            const double bytes_transferred =
                static_cast<double>(buffer_bytes) *
                static_cast<double>(measured_iterations) * 2.0;
            const double elapsed_seconds = static_cast<double>(elapsed_ms) / 1000.0;
            const double bandwidth_gbps =
                elapsed_seconds > 0.0
                    ? bytes_transferred / elapsed_seconds / 1.0e9
                    : 0.0;

            cudaEventDestroy(end);
            cudaEventDestroy(start);
            cudaFree(destination);
            cudaFree(source);

            return VramBenchmarkResult{
                bandwidth_gbps,
                buffer_bytes,
                warmup_iterations,
                measured_iterations,
                static_cast<double>(elapsed_ms)};
        } catch (...) {
            if (end != nullptr) {
                cudaEventDestroy(end);
            }
            if (start != nullptr) {
                cudaEventDestroy(start);
            }
            throw;
        }
    } catch (...) {
        cudaFree(destination);
        cudaFree(source);
        throw;
    }
}

GpuStressor::GpuStressor(
    double target_percent,
    std::chrono::milliseconds control_interval,
    double deadband_percent,
    double max_duty_step,
    const std::atomic<double>& measured_utilization,
    int device_id)
    : target_percent_(std::clamp(target_percent, 0.0, 100.0)),
      control_interval_(control_interval),
      deadband_percent_(deadband_percent),
      max_duty_step_(max_duty_step),
      measured_utilization_(measured_utilization),
      device_id_(device_id) {
    duty_.store(target_percent_ > 0.0 ? target_percent_ / 100.0 : 0.0);
}

GpuStressor::~GpuStressor() {
    stop();
}

void GpuStressor::start() {
    if (thread_.joinable() || target_percent_ <= 0.0) {
        return;
    }
    stop_requested_.store(false);
    thread_ = std::thread([this] {
        try {
            loop();
        } catch (const std::exception& error) {
            std::cerr << "[contention] GPU compute stressor failed: "
                      << error.what() << std::endl;
            stop_requested_.store(true, std::memory_order_relaxed);
        }
    });
}

void GpuStressor::stop() {
    stop_requested_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
}

double GpuStressor::duty() const noexcept {
    return duty_.load(std::memory_order_relaxed);
}

void GpuStressor::loop() {
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_id_),
               "cudaGetDeviceProperties");

    constexpr int threads_per_block = 256;
    const int blocks = std::max(
        properties.multiProcessorCount * 8, 1);
    const std::size_t elements =
        static_cast<std::size_t>(blocks) * threads_per_block;

    float* output = nullptr;
    check_cuda(cudaMalloc(&output, elements * sizeof(float)), "cudaMalloc");

    DutyController controller(deadband_percent_, max_duty_step_, duty());
    auto last_control = std::chrono::steady_clock::now();
    std::uint32_t seed = 0x12345678U;

    // Keep each kernel long enough to exercise the GPU, but short enough that
    // duty-cycle control remains responsive.
    constexpr int kernel_iterations = 50000;

    try {
        constexpr auto duty_quantum = std::chrono::milliseconds(100);

        while (!stop_requested_.load(std::memory_order_relaxed)) {
            const double duty_cycle = controller.duty();
            duty_.store(duty_cycle, std::memory_order_relaxed);

            const auto active_window = std::chrono::duration_cast<std::chrono::milliseconds>(
                duty_quantum * duty_cycle);
            const auto active_end = std::chrono::steady_clock::now() + active_window;

            while (duty_cycle > 0.0 &&
                   std::chrono::steady_clock::now() < active_end &&
                   !stop_requested_.load(std::memory_order_relaxed)) {
                compute_kernel<<<blocks, threads_per_block>>>(
                    output, seed++, kernel_iterations);
                check_cuda(cudaGetLastError(), "compute kernel launch");
                check_cuda(cudaDeviceSynchronize(), "compute kernel synchronize");
            }

            const auto sleep_for = std::chrono::duration_cast<std::chrono::milliseconds>(
                duty_quantum * (1.0 - duty_cycle));
            if (sleep_for.count() > 0) {
                std::this_thread::sleep_for(sleep_for);
            }

            const auto now = std::chrono::steady_clock::now();
            if (now - last_control >= control_interval_) {
                const double measured = measured_utilization_.load(
                    std::memory_order_relaxed);
                controller.update(target_percent_, measured);
            }
        }
    } catch (...) {
        cudaFree(output);
        throw;
    }

    cudaFree(output);
}

VramStressor::VramStressor(
    double target_percent,
    double safe_ceiling_gbps,
    std::chrono::milliseconds control_interval,
    double deadband_percent,
    double max_duty_step,
    std::atomic<double>& published_bandwidth,
    int device_id)
    : target_percent_(std::clamp(target_percent, 0.0, 100.0)),
      ceiling_gbps_(safe_ceiling_gbps),
      control_interval_(control_interval),
      deadband_percent_(deadband_percent),
      max_duty_step_(max_duty_step),
      published_bandwidth_(published_bandwidth),
      device_id_(device_id) {
    duty_.store(target_percent_ > 0.0 ? target_percent_ / 100.0 : 0.0);
}

VramStressor::~VramStressor() {
    stop();
}

void VramStressor::start() {
    if (thread_.joinable() || target_percent_ <= 0.0) {
        return;
    }
    stop_requested_.store(false);
    thread_ = std::thread([this] {
        try {
            loop();
        } catch (const std::exception& error) {
            std::cerr << "[contention] VRAM stressor failed: "
                      << error.what() << std::endl;
            stop_requested_.store(true, std::memory_order_relaxed);
        }
    });
}

void VramStressor::stop() {
    stop_requested_.store(true);
    if (thread_.joinable()) {
        thread_.join();
    }
}

double VramStressor::duty() const noexcept {
    return duty_.load(std::memory_order_relaxed);
}

void VramStressor::loop() {
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");

    // A large working set is required so the workload is not satisfied by
    // small caches. Keep it configurable later; 512 MiB per input is a safe
    // starting point for the RTX 4070-class target used by FlexOn.
    constexpr std::size_t bytes_per_buffer = 512ULL * 1024ULL * 1024ULL;
    const std::size_t elements = bytes_per_buffer / sizeof(float);

    float* input_a = nullptr;
    float* input_b = nullptr;
    float* output = nullptr;

    check_cuda(cudaMalloc(&input_a, bytes_per_buffer), "cudaMalloc input_a");
    check_cuda(cudaMalloc(&input_b, bytes_per_buffer), "cudaMalloc input_b");
    check_cuda(cudaMalloc(&output, bytes_per_buffer), "cudaMalloc output");

    // Initialize once. Runtime stress should measure device-memory traffic,
    // not host-side random-number generation.
    std::vector<float> host(1U << 20);
    for (std::size_t i = 0; i < host.size(); ++i) {
        const auto value = static_cast<float>((i * 2654435761ULL) & 0xFFFF) /
                           65535.0F;
        host[i] = value * 2.0F - 1.0F;
    }

    for (std::size_t offset = 0; offset < elements; offset += host.size()) {
        const std::size_t count = std::min(host.size(), elements - offset);
        check_cuda(cudaMemcpy(input_a + offset,
                              host.data(),
                              count * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "initialize input_a");
        check_cuda(cudaMemcpy(input_b + offset,
                              host.data(),
                              count * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "initialize input_b");
    }

    DutyController controller(deadband_percent_, max_duty_step_, duty());
    auto window_start = std::chrono::steady_clock::now();
    std::uint64_t bytes_processed = 0;
    std::uint32_t seed = 0xA341316CU;

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (elements + threads_per_block - 1) / threads_per_block);

    try {
        constexpr auto duty_quantum = std::chrono::milliseconds(100);

        while (!stop_requested_.load(std::memory_order_relaxed)) {
            const double duty_cycle = controller.duty();
            duty_.store(duty_cycle, std::memory_order_relaxed);

            const auto active_window = std::chrono::duration_cast<std::chrono::milliseconds>(
                duty_quantum * duty_cycle);
            const auto active_end = std::chrono::steady_clock::now() + active_window;

            while (duty_cycle > 0.0 &&
                   std::chrono::steady_clock::now() < active_end &&
                   !stop_requested_.load(std::memory_order_relaxed)) {
                vram_kernel<<<blocks, threads_per_block>>>(
                    input_a, input_b, output, elements, seed++);
                check_cuda(cudaGetLastError(), "VRAM kernel launch");
                check_cuda(cudaDeviceSynchronize(), "VRAM kernel synchronize");

                // 2 reads + 1 write per element.
                bytes_processed += 3ULL * elements * sizeof(float);
            }

            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double>(now - window_start).count();
            if (elapsed >= std::chrono::duration<double>(control_interval_).count()) {
                const double gbps =
                    elapsed > 0.0
                        ? static_cast<double>(bytes_processed) /
                              elapsed / 1.0e9
                        : 0.0;
                published_bandwidth_.store(gbps, std::memory_order_relaxed);

                const double utilization =
                    ceiling_gbps_ > 0.0
                        ? 100.0 * gbps / ceiling_gbps_
                        : 0.0;
                controller.update(target_percent_, utilization);

                bytes_processed = 0;
                window_start = now;
            }

            const auto sleep_for = std::chrono::duration_cast<std::chrono::milliseconds>(
                duty_quantum * (1.0 - duty_cycle));
            if (sleep_for.count() > 0) {
                std::this_thread::sleep_for(sleep_for);
            }
        }
    } catch (...) {
        cudaFree(input_a);
        cudaFree(input_b);
        cudaFree(output);
        throw;
    }

    cudaFree(input_a);
    cudaFree(input_b);
    cudaFree(output);
}

}  // namespace flexon::contention
