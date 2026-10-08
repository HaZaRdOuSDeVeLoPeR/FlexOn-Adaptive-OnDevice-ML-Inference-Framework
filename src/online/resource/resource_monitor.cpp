#include <string>
#include <algorithm>
#include <fstream>

#include <flexon/online/resource/resource_monitor.hpp>

namespace flexon::online::monitor {

CpuSample read_cpu_sample() {
    std::ifstream input("/proc/stat");
    std::string label;
    std::uint64_t user = 0;
    std::uint64_t nice = 0;
    std::uint64_t system = 0;
    std::uint64_t idle = 0;
    std::uint64_t iowait = 0;
    std::uint64_t irq = 0;
    std::uint64_t softirq = 0;
    std::uint64_t steal = 0;

    if (!(input >> label >> user >> nice >> system >> idle >> iowait
                 >> irq >> softirq >> steal) || label != "cpu") {
        return {};
    }

    const auto idle_all = idle + iowait;
    const auto total = user + nice + system + idle_all +
                       irq + softirq + steal;
    return CpuSample{idle_all, total};
}

double cpu_remaining_capacity(CpuSample& previous) {
    const auto current = read_cpu_sample();

    if (current.total <= previous.total ||
        current.idle < previous.idle) {
        previous = current;
        return 1.0;
    }

    const auto total_delta = current.total - previous.total;
    const auto idle_delta = current.idle - previous.idle;
    previous = current;

    if (total_delta == 0) {
        return 1.0;
    }

    const double remaining =
        static_cast<double>(idle_delta) /
        static_cast<double>(total_delta);

    return std::clamp(remaining, 0.01, 1.0);
}

double query_cuda_remaining_capacity() {
    // GPU utilization is intentionally sampled out-of-band so nvidia-smi
    // startup latency does not become part of the segment scheduling path.
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=utilization.gpu "
        "--format=csv,noheader,nounits -i 0 2>/dev/null",
        "r");
    if (pipe == nullptr) return 1.0;

    char buffer[64]{};
    const bool read_ok = std::fgets(buffer, sizeof(buffer), pipe) != nullptr;
    const int status = pclose(pipe);
    if (!read_ok || status != 0) return 1.0;

    try {
        const double utilization =
            std::stod(std::string(buffer));
        return std::clamp(1.0 - utilization / 100.0, 0.0, 1.0);
    } catch (...) {
        return 1.0;
    }
}

} // namespace flexon::online::resource