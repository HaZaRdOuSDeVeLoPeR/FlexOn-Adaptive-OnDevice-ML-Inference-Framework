#include <cassert>
#include <chrono>
#include <cmath>
#include <thread>

#include <flexon/online/resource/resource_monitor.hpp>

int main() {
    using namespace flexon::online::monitor;

    const auto sample = read_cpu_sample();
    assert(sample.total >= sample.idle);
    assert(sample.total > 0);

    // The production sampler should produce a bounded finite capacity when
    // given a real previous sample. A short delay ensures non-zero deltas on
    // normal Linux /proc/stat implementations.
    CpuSample previous = sample;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const double remaining = cpu_remaining_capacity(previous);

    assert(std::isfinite(remaining));
    assert(remaining >= 0.0);
    assert(remaining <= 1.0);
    assert(previous.total >= sample.total);
    assert(previous.idle >= sample.idle);

    return 0;
}
