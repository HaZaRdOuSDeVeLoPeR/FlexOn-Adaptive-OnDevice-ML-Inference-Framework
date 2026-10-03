#pragma once

#include <cstdint>
#include <cmath>

namespace flexon::online::monitor {

struct MonitorState {
    double cpu_remaining{1.0};
    double cuda_remaining{1.0};
};

struct CpuSample {
    std::uint64_t idle{0};
    std::uint64_t total{0};
};

CpuSample read_cpu_sample();
double cpu_remaining_capacity(CpuSample& previous);
double query_cuda_remaining_capacity();

} // namespace flexon::online::monitor