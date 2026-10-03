#pragma once

#include <cuda_runtime_api.h>
#include <flexon/core/types.hpp>
#include <flexon/online/artifact/manifest_loader.hpp>
#include <flexon/online/execution/executor.hpp>
#include <flexon/online/execution/buffer_arena.hpp>

namespace flexon::online::helper {

void check_cuda(cudaError_t status, const char* operation);

void print_segment_stats(
    const manifest::SegmentManifest& manifest,
    const executor::SegmentExecutionStats& stats,
    const arena::BufferArena& arena);

std::vector<std::int64_t> concrete_shape(std::vector<std::int64_t> shape);
const char* resource_name(core::Resource resource);

std::unique_ptr<Ort::Session> create_session(
    Ort::Env& env,
    const std::filesystem::path& model_path,
    core::Resource resource);

runtime::PreparedSession* find_session(
    runtime::SegmentRuntime& segment,
    core::Resource resource);

Ort::Value make_input_value(
    Ort::AllocatorWithDefaultOptions& allocator,
    const Ort::TypeInfo& type_info,
    std::uint32_t seed);

Ort::MemoryInfo memory_info_for_resource(core::Resource resource);

void add_initial_inputs(
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    const std::vector<std::string>& input_names,
    Ort::Session& first_session,
    Ort::AllocatorWithDefaultOptions& allocator);

} // namespace::flexon::online::helper