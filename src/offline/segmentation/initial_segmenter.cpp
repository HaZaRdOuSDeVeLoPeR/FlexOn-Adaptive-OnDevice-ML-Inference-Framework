#include <flexon/offline/segmentation/initial_segmenter.hpp>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace flexon::offline::segmentation {

namespace {

bool is_fallback(const std::string& op,
                 const config::OfflineConfig& config) {
    return std::find(config.cuda_fallback_ops.begin(),
                     config.cuda_fallback_ops.end(), op) !=
           config.cuda_fallback_ops.end();
}

const core::OperatorCost* find_cost(
    const core::OperatorProfile& profile,
    const core::Resource resource) {
    for (const auto& cost : profile.costs) {
        if (cost.resource == resource) {
            return &cost;
        }
    }
    return nullptr;
}

bool supported(const core::OperatorProfile& profile,
               const core::Resource resource) {
    const auto* cost = find_cost(profile, resource);
    return cost != nullptr &&
           cost->status == core::ResourceSupportStatus::Supported;
}

core::SegmentInfo make_segment(
    const onnx::GraphProto& graph,
    const core::GraphInfo& info,
    std::uint32_t level,
    std::uint32_t id,
    const std::vector<std::uint32_t>& indices) {

    core::SegmentInfo segment;
    segment.level = level;
    segment.id = id;
    segment.operator_indices = indices;

    std::unordered_set<std::uint32_t> members(indices.begin(), indices.end());

    for (const auto index : indices) {
        const auto& op = info.operators.at(index);
        segment.operator_names.push_back(op.name);

        for (const auto& input : op.inputs) {
            const auto producer = info.producer_operator.find(input);
            const bool produced_inside =
                producer != info.producer_operator.end() &&
                members.count(producer->second) != 0;

            if (!produced_inside &&
                std::find(segment.input_tensors.begin(),
                          segment.input_tensors.end(), input) ==
                    segment.input_tensors.end()) {
                segment.input_tensors.push_back(input);
            }
        }

        for (const auto& output : op.outputs) {
            const auto consumers = info.consumer_operators.find(output);
            bool consumed_outside = false;

            if (consumers != info.consumer_operators.end()) {
                for (const auto consumer : consumers->second) {
                    if (members.count(consumer) == 0) {
                        consumed_outside = true;
                        break;
                    }
                }
            }

            const bool is_graph_output =
                std::find(info.graph_outputs.begin(),
                          info.graph_outputs.end(), output) !=
                info.graph_outputs.end();

            if ((consumed_outside || is_graph_output) &&
                std::find(segment.output_tensors.begin(),
                          segment.output_tensors.end(), output) ==
                    segment.output_tensors.end()) {
                segment.output_tensors.push_back(output);
            }
        }
    }

    return segment;
}

}  // namespace

std::vector<core::SegmentInfo> InitialSegmenter::create(
    const onnx::ModelProto& model,
    const core::GraphInfo& graph,
    const config::OfflineConfig& config,
    const core::OperatorProfileMap& operator_profiles) {

    if (config.initial_partition != "fallback_operator") {
        throw std::invalid_argument(
            "Unsupported initial partition policy: " +
            config.initial_partition);
    }

    std::vector<core::SegmentInfo> segments;
    std::vector<std::uint32_t> current;

    const auto flush = [&]() {
        if (current.empty()) {
            return;
        }

        segments.push_back(make_segment(
            model.graph(), graph, 0,
            static_cast<std::uint32_t>(segments.size()), current));
        current.clear();
    };

    for (const auto& op : graph.operators) {
        const auto profile_it = operator_profiles.find(op.graph_index);
        if (profile_it == operator_profiles.end()) {
            throw std::runtime_error(
                "Missing operator profile for operator '" + op.name + "'");
        }

        for (const auto& cost : profile_it->second.costs) {
            if (cost.status == core::ResourceSupportStatus::ProfilingFailed) {
                throw std::runtime_error(
                    "Profiling failed for operator '" + op.name +
                    "' on resource '" +
                    (cost.resource == core::Resource::CPU ? "cpu" : "cuda") +
                    "': " + cost.reason);
            }
        }

        const bool cpu_supported =
            config.cpu_enabled &&
            supported(profile_it->second, core::Resource::CPU);
        const bool cuda_supported =
            config.cuda_enabled &&
            supported(profile_it->second, core::Resource::CUDA);

        const bool supported_on_enabled_resource =
            (config.cpu_enabled && cpu_supported) ||
            (config.cuda_enabled && cuda_supported);

        if (!supported_on_enabled_resource) {
            throw std::runtime_error(
                "Operator '" + op.name +
                "' is unsupported on all enabled resources");
        }

        const bool resource_boundary =
            config.cpu_enabled && config.cuda_enabled &&
            (cpu_supported != cuda_supported);

        // An explicit fallback remains a reproducible override. Otherwise,
        // any operator supported by exactly one enabled resource becomes an
        // initial resource-capability boundary. Operators supported by both
        // resources remain in the surrounding segment.
        const bool fallback =
            is_fallback(op.op_type, config) || resource_boundary;

        if (fallback) {
            flush();
            segments.push_back(make_segment(
                model.graph(), graph, 0,
                static_cast<std::uint32_t>(segments.size()),
                {op.graph_index}));
        } else {
            current.push_back(op.graph_index);
        }
    }

    flush();

    return segments;
}

}  // namespace flexon::offline::segmentation
