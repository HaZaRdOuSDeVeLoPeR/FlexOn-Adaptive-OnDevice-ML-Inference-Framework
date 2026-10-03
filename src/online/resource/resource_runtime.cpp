#include <flexon/online/execution/helper.hpp>
#include <flexon/online/execution/executor.hpp>
#include <flexon/online/resource/resource_runtime.hpp>

namespace flexon::online::runtime {

core::Resource choose_resource(const SegmentRuntime& segment,
                                core::Resource requested) {
    if (requested == core::Resource::CPU) {
        if (!segment.manifest.cpu_supported) {
            throw std::runtime_error(
                "CPU execution requested for a CPU-unsupported segment " +
                std::to_string(segment.manifest.id));
        }
        return core::Resource::CPU;
    }
    if (requested == core::Resource::CUDA) {
        if (!segment.manifest.cuda_supported) {
            throw std::runtime_error(
                "CUDA execution requested for a CUDA-unsupported segment " +
                std::to_string(segment.manifest.id));
        }
        return core::Resource::CUDA;
    }

    if (segment.manifest.cpu_supported && segment.manifest.cuda_supported) {
        return segment.manifest.cuda_mean_ms < segment.manifest.cpu_mean_ms
                   ? core::Resource::CUDA
                   : core::Resource::CPU;
    }
    return segment.manifest.cpu_supported ? core::Resource::CPU
                                          : core::Resource::CUDA;
}

core::Resource choose_concrete_resource(
    const SegmentRuntime& segment,
    core::Resource requested) {
    if (requested == core::Resource::Auto) {
        throw std::invalid_argument(
            "Auto resource must be resolved by OnlineScheduler before "
            "segment execution");
    }
    return choose_resource(segment, requested);
}

core::Resource tensor_resource(const Ort::Value& value) {
    const auto memory_info = value.GetTensorMemoryInfo();
    auto name = memory_info.GetAllocatorName();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (name.find("cuda") != std::string::npos) {
        return core::Resource::CUDA;
    }
    if (name.find("cpu") != std::string::npos) {
        return core::Resource::CPU;
    }

    throw std::runtime_error(
        "Unsupported tensor memory allocator: " + name);
}

} // namespace flexon::online::resource