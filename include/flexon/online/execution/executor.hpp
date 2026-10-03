#pragma once

#include <vector>
#include <memory>
#include <future>

#include <flexon/core/types.hpp>
#include <flexon/online/resource/resource_runtime.hpp>
#include <onnxruntime_cxx_api.h>

namespace flexon::online::executor {

struct SegmentExecutionStats {
    core::Resource resource{core::Resource::CPU};
    double elapsed_ms{0.0};
    double boundary_copy_ms{0.0};
};

struct SegmentExecutionResult {
    SegmentExecutionStats stats;
    std::chrono::steady_clock::time_point completed_at{};
    std::vector<std::pair<std::string, runtime::RuntimeTensor>> outputs;
};

SegmentExecutionResult execute_segment_once(
    runtime::SegmentRuntime& segment,
    core::Resource requested_resource,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    arena::BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::shared_ptr<std::promise<void>>& inputs_bound = nullptr);

void commit_segment_outputs(
    SegmentExecutionResult& result,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store);

SegmentExecutionStats execute_segment(
    runtime::SegmentRuntime& segment,
    core::Resource requested_resource,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    arena::BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator);

} // namespace flexon::online::executor