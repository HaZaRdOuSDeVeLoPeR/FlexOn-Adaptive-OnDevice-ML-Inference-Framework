#include <cassert>
#include <cmath>
#include <stdexcept>

#include <flexon/online/resource/resource_runtime.hpp>

namespace {

flexon::online::runtime::SegmentRuntime make_segment(
    bool cpu_supported,
    bool cuda_supported) {
    flexon::online::runtime::SegmentRuntime segment;
    segment.manifest.cpu_supported = cpu_supported;
    segment.manifest.cuda_supported = cuda_supported;
    return segment;
}

}  // namespace

int main() {
    using flexon::core::Resource;
    using flexon::online::runtime::choose_concrete_resource;
    using flexon::online::runtime::choose_resource;

    const auto both = make_segment(true, true);
    const auto cpu_only = make_segment(true, false);
    const auto cuda_only = make_segment(false, true);

    assert(choose_resource(both, Resource::CPU) == Resource::CPU);
    assert(choose_resource(both, Resource::CUDA) == Resource::CUDA);
    assert(choose_resource(cpu_only, Resource::Auto) == Resource::CPU);
    assert(choose_resource(cuda_only, Resource::Auto) == Resource::CUDA);

    assert(choose_concrete_resource(both, Resource::CPU) == Resource::CPU);
    assert(choose_concrete_resource(both, Resource::CUDA) == Resource::CUDA);

    // choose_concrete_resource is intentionally only for already-resolved
    // resources; Auto must be rejected at this boundary.
    bool threw = false;
    try {
        (void)choose_concrete_resource(both, Resource::Auto);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Unsupported explicit requests must not silently become another resource.
    threw = false;
    try {
        (void)choose_resource(cpu_only, Resource::CUDA);
    } catch (...) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        (void)choose_resource(cuda_only, Resource::CPU);
    } catch (...) {
        threw = true;
    }
    assert(threw);

    return 0;
}
