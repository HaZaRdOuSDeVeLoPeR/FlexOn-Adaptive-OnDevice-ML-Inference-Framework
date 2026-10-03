#pragma once

#include <flexon/online/execution/executor.hpp>
#include <flexon/online/scheduler/scheduler.hpp>

namespace flexon::online::recovery {

executor::SegmentExecutionStats execute_segment_with_recovery(
    runtime::SegmentRuntime& segment,
    core::Resource primary_resource,
    const scheduler::SchedulerSegmentCosts& costs,
    scheduler::OnlineScheduler& scheduler,
    const scheduler::SchedulerConfig& scheduler_config,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    arena::BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::future<executor::SegmentExecutionResult>>& background_recoveries);

} // namespace flexon::online::recovery