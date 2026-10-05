#include <flexon/contention/contention_generator.hpp>
#include <flexon/contention/gpu_stressor.hpp>
#include <flexon/contention/bandwidth_stressor.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

volatile std::sig_atomic_t g_stop = 0;

void handle_signal(int) {
    g_stop = 1;
}

double parse_percent(const char* option, const char* value) {
    const double result = std::stod(value);
    if (result < 0.0 || result > 100.0) {
        throw std::invalid_argument(std::string(option) + " must be in [0, 100]");
    }
    return result;
}

void usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [options]\n\n"
        << "Background contention targets (measured load, not duty cycle):\n"
        << "  --cpu <0..100>       CPU utilization target (%)\n"
        << "  --gpu <0..100>       GPU compute utilization target (%)\n"
        << "  --dram <0..100>      DRAM bandwidth target (% of safe ceiling)\n"
        << "  --vram <0..100>      VRAM bandwidth target (% of safe ceiling)\n\n"
        << "Other options:\n"
        << "  --duration <seconds>       Duration (default: 30)\n"
        << "  --safety-factor <0..1>     Benchmark safety factor (default: 0.85)\n"
        << "  --benchmark-dram           Measure DRAM bandwidth and exit\n"
        << "  --benchmark-vram           Measure VRAM bandwidth and exit\n"
        << "  --benchmark-memory         Measure DRAM and VRAM bandwidth and exit\n"
        << "  --quiet                    Disable periodic telemetry output\n"
        << "  --help, -h                 Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    flexon::contention::Config config;
    int duration_seconds = 30;
    bool benchmark_dram_only = false;
    bool benchmark_vram_only = false;
    bool benchmark_memory_only = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            }
            if (arg == "--quiet") {
                config.verbose = false;
                continue;
            }
            if (arg == "--benchmark-dram") {
                benchmark_dram_only = true;
                continue;
            }
            if (arg == "--benchmark-vram") {
                benchmark_vram_only = true;
                continue;
            }
            if (arg == "--benchmark-memory") {
                benchmark_memory_only = true;
                continue;
            }

            auto require_value = [&](const char* option) -> const char* {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(
                        std::string("Missing value for ") + option);
                }
                return argv[++i];
            };

            if (arg == "--cpu") {
                config.targets.cpu_percent = parse_percent(arg.c_str(), require_value("--cpu"));
            } else if (arg == "--gpu") {
                config.targets.gpu_percent = parse_percent(arg.c_str(), require_value("--gpu"));
            } else if (arg == "--dram") {
                config.targets.dram_percent = parse_percent(arg.c_str(), require_value("--dram"));
            } else if (arg == "--vram") {
                config.targets.vram_percent = parse_percent(arg.c_str(), require_value("--vram"));
            } else if (arg == "--duration") {
                duration_seconds = std::stoi(require_value("--duration"));
                if (duration_seconds <= 0) {
                    throw std::invalid_argument("--duration must be positive");
                }
            } else if (arg == "--safety-factor") {
                config.safety_factor = std::stod(require_value("--safety-factor"));
                if (config.safety_factor <= 0.0 || config.safety_factor > 1.0) {
                    throw std::invalid_argument("--safety-factor must be in (0, 1]");
                }
            } else {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        if (benchmark_dram_only || benchmark_memory_only) {
            const auto result = flexon::contention::benchmark_dram_bandwidth();
            std::cout << std::fixed << std::setprecision(3)
                      << "[contention] measured DRAM bandwidth: "
                      << result.bandwidth_gbps << " GB/s\n"
                      << "[contention] working set: "
                      << static_cast<double>(result.working_set_bytes) /
                             (1024.0 * 1024.0 * 1024.0)
                      << " GiB, workers=" << result.workers
                      << ", warmup=" << result.warmup_iterations
                      << ", iterations=" << result.measured_iterations
                      << ", elapsed=" << result.elapsed_ms << " ms\n";
        }

        if (benchmark_vram_only || benchmark_memory_only) {
            const auto result =
                flexon::contention::benchmark_vram_bandwidth(config.cuda_device_id);
            std::cout << std::fixed << std::setprecision(3)
                      << "[contention] measured VRAM bandwidth: "
                      << result.bandwidth_gbps << " GB/s\n"
                      << "[contention] buffer: "
                      << static_cast<double>(result.buffer_bytes) /
                             (1024.0 * 1024.0)
                      << " MiB, warmup=" << result.warmup_iterations
                      << ", iterations=" << result.measured_iterations
                      << ", elapsed=" << result.elapsed_ms << " ms\n";
        }

        if (benchmark_dram_only || benchmark_vram_only || benchmark_memory_only) {
            return 0;
        }

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        flexon::contention::ContentionGenerator generator(config);
        generator.start();

        std::cout << "[contention] generator started; targets: CPU="
                  << config.targets.cpu_percent << "%, GPU="
                  << config.targets.gpu_percent << "%, DRAM="
                  << config.targets.dram_percent << "%, VRAM="
                  << config.targets.vram_percent << "%\n";

        const auto stabilization_timeout =
            std::chrono::seconds(std::min(30, std::max(5, duration_seconds / 2)));
        if (generator.wait_until_stable(stabilization_timeout)) {
            std::cout << "[contention] stabilized\n";
        } else {
            std::cout << "[contention] warning: target did not stabilize within "
                      << stabilization_timeout.count() << " s\n";
        }

        const auto end = std::chrono::steady_clock::now() +
                         std::chrono::seconds(duration_seconds);
        while (!g_stop && std::chrono::steady_clock::now() < end) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        generator.stop();
        std::cout << "[contention] stopped\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[contention] ERROR: " << error.what() << '\n';
        return 1;
    }
}
