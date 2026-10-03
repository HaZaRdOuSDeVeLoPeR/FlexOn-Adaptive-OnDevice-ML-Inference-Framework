#include <random>

#include <flexon/online/execution/helper.hpp>
#include <flexon/online/resource/resource_runtime.hpp>
#include <flexon/online/execution/executor.hpp>

namespace flexon::online::executor {

SegmentExecutionResult execute_segment_once(
    runtime::SegmentRuntime& segment,
    core::Resource requested_resource,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    arena::BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::shared_ptr<std::promise<void>>& inputs_bound
) {

    const auto resource = runtime::choose_concrete_resource(segment, requested_resource);
    auto* prepared = helper::find_session(segment, resource);
    if (!prepared || !prepared->session) {
        throw std::runtime_error(
            "Prepared session missing for " + std::string(helper::resource_name(resource)) +
            " segment " + std::to_string(segment.manifest.id));
    }

    Ort::Session& session = *prepared->session;
    const auto input_count = session.GetInputCount();
    const auto output_count = session.GetOutputCount();

    std::vector<std::string> input_name_storage;
    input_name_storage.reserve(input_count);
    std::vector<std::string> output_name_storage;
    output_name_storage.reserve(output_count);
    double boundary_copy_ms = 0.0;

    std::vector<std::shared_ptr<arena::BufferLease>> transferred_inputs;
    transferred_inputs.reserve(input_count);
    std::vector<std::shared_ptr<arena::BufferLease>> output_leases;
    output_leases.reserve(output_count);
    Ort::IoBinding io_binding(session);

    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_name_storage.emplace_back(name.get());
        const auto& input_name = input_name_storage.back();
        const auto it = tensor_store.find(input_name);

        if (it == tensor_store.end()) {
            throw std::runtime_error(
                "Missing tensor required by segment " +
                std::to_string(segment.manifest.id) + ": " + input_name);
        }

        const auto source_resource = runtime::tensor_resource(it->second.value);
        if (source_resource == resource) {
            io_binding.BindInput(input_name.c_str(), it->second.value);
        } else {
            const auto type_info = it->second.value.GetTensorTypeAndShapeInfo();
            const auto shape = helper::concrete_shape(type_info.GetShape());
            const auto type = type_info.GetElementType();
            auto lease = arena.acquire(arena::buffer_key(resource, shape, type));
            const auto bytes = lease->block->bytes;

            const auto memory_info = helper::memory_info_for_resource(resource);
            Ort::Value transferred = Ort::Value::CreateTensor(
                memory_info,
                lease->block->data,
                bytes,
                shape.data(),
                shape.size(),
                type);

            const auto start = std::chrono::steady_clock::now();
            if (resource == core::Resource::CUDA) {
                helper::check_cuda(
                    cudaMemcpy(
                        lease->block->data,
                        it->second.value.GetTensorRawData(),
                        bytes,
                        cudaMemcpyHostToDevice),
                    "cudaMemcpyHostToDevice");
            } else {
                helper::check_cuda(
                    cudaMemcpy(
                        lease->block->data,
                        it->second.value.GetTensorRawData(),
                        bytes,
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpyDeviceToHost");
            }
            const auto end = std::chrono::steady_clock::now();
            boundary_copy_ms += std::chrono::duration<double, std::milli>(
                end - start).count();

            io_binding.BindInput(input_name.c_str(), transferred);
            transferred_inputs.push_back(std::move(lease));
        }
    }

    // Recovery executions may continue after the winning execution returns.
    // Signal once this worker has finished reading the shared tensor store so
    // the caller can safely update tensor liveness/output ownership.
    if (inputs_bound) {
        inputs_bound->set_value();
    }

    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_name_storage.emplace_back(name.get());
    }

    const auto memory_info = helper::memory_info_for_resource(resource);

    for (std::size_t i = 0; i < output_count; ++i) {
        auto type_info = session.GetOutputTypeInfo(i);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        const auto raw_shape = tensor_info.GetShape();
        bool concrete = true;
        for (const auto dim : raw_shape) {
            if (dim < 0) {
                concrete = false;
                break;
            }
        }

        if (!concrete) {
            io_binding.BindOutput(output_name_storage[i].c_str(), memory_info);
            output_leases.push_back(nullptr);
            continue;
        }

        const auto shape = helper::concrete_shape(raw_shape);
        const auto type = tensor_info.GetElementType();
        auto lease = arena.acquire(arena::buffer_key(resource, shape, type));
        const auto bytes = lease->block->bytes;
        Ort::Value output = Ort::Value::CreateTensor(
            memory_info,
            lease->block->data,
            bytes,
            shape.data(),
            shape.size(),
            type);
        io_binding.BindOutput(output_name_storage[i].c_str(), output);
        output_leases.push_back(std::move(lease));
    }

    const auto start = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr}, io_binding);
    io_binding.SynchronizeOutputs();
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    auto outputs = io_binding.GetOutputValues();
    if (outputs.size() != output_name_storage.size()) {
        throw std::runtime_error(
            "Segment " + std::to_string(segment.manifest.id) +
            " returned an unexpected number of outputs");
    }

    SegmentExecutionResult result;
    result.stats = SegmentExecutionStats{
        resource,
        elapsed_ms,
        boundary_copy_ms};
    result.completed_at = end;
    result.outputs.reserve(outputs.size());

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        runtime::RuntimeTensor tensor;
        tensor.lease = std::move(output_leases[i]);
        tensor.value = std::move(outputs[i]);
        result.outputs.emplace_back(
            output_name_storage[i],
            std::move(tensor));
    }

    io_binding.ClearBoundOutputs();
    io_binding.ClearBoundInputs();

    if (resource == core::Resource::CUDA) {
        helper::check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize before releasing transferred inputs");
    }

    return result;
}

void commit_segment_outputs(
    SegmentExecutionResult& result,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store) {

    for (auto& output : result.outputs) {
        auto existing = tensor_store.find(output.first);
        if (existing != tensor_store.end()) {
            existing->second.value = Ort::Value{nullptr};
            existing->second.lease.reset();
            existing->second = std::move(output.second);
        } else {
            tensor_store.emplace(
                output.first,
                std::move(output.second));
        }
    }
}

SegmentExecutionStats execute_segment(
    runtime::SegmentRuntime& segment,
    core::Resource requested_resource,
    std::unordered_map<std::string, runtime::RuntimeTensor>& tensor_store,
    arena::BufferArena& arena,
    Ort::AllocatorWithDefaultOptions& allocator) {

    auto result = execute_segment_once(
        segment,
        requested_resource,
        tensor_store,
        arena,
        allocator);
    const auto stats = result.stats;
    commit_segment_outputs(result, tensor_store);
    helper::print_segment_stats(segment.manifest, stats, arena);
    return stats;
}

} // namespace::flexon::online::execution