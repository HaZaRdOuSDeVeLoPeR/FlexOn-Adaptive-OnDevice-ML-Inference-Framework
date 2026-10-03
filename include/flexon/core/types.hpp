#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>

namespace flexon::core {

/**
 * Identifies a hardware execution resource used by FlexOn.
 *
 * This is deliberately a data-only type. Policy belongs to the online
 * scheduler, not to core data structures.
 */
enum class Resource : std::uint8_t {
    CPU,
    CUDA,
    Auto
};

/**
 * Describes whether a resource can execute a profiled graph object.
 *
 * Unsupported means the resource cannot execute the object without relying
 * on another execution provider. ProfilingFailed is reserved for cases
 * where the capability probe succeeded far enough to create a session but
 * measurement itself failed.
 */
enum class ResourceSupportStatus : std::uint8_t {
    Supported,
    Unsupported,
    ProfilingFailed
};

/**
 * Identifies one operator in the source ONNX graph.
 */
struct OperatorInfo {
    // Stable position of the node in the source ONNX graph. This remains
    // meaningful even when the ONNX node has no name.
    std::uint32_t graph_index{0};
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

/**
 * Structural representation of the top-level ONNX graph used by offline
 * analysis. The type intentionally contains no ONNX/ORT-specific objects.
 */
struct TensorInfo {
    std::string name;
    int data_type{0};
    std::vector<std::int64_t> shape;
    bool shape_known{false};
};

struct GraphInfo {
    std::string name;
    std::vector<OperatorInfo> operators;
    std::vector<std::string> graph_inputs;
    std::vector<std::string> graph_outputs;
    std::vector<std::string> initializers;

    // Tensor producer: tensor -> source operator graph index.
    std::unordered_map<std::string, std::uint32_t> producer_operator;

    // Tensor consumers: tensor -> source operator graph indices.
    std::unordered_map<std::string, std::vector<std::uint32_t>> consumer_operators;

    // Best-effort tensor metadata obtained from ONNX value information and
    // shape inference. It is used by the offline segment compiler/profiler.
    std::unordered_map<std::string, TensorInfo> tensors;
};

/**
 * Describes a segment as a contiguous ordered sequence of source operators.
 */
struct SegmentInfo {
    std::uint32_t level{0};
    std::uint32_t id{0};
    std::vector<std::string> operator_names;
    std::vector<std::string> input_tensors;
    std::vector<std::string> output_tensors;
    std::vector<std::uint32_t> operator_indices;
};

/**
 * One measured execution cost for a segment/resource pair.
 */
struct SegmentCost {
    Resource resource{Resource::CPU};
    double mean_ms{0.0};
    double percentile_ms{0.0};
    ResourceSupportStatus status{ResourceSupportStatus::Supported};
    std::string reason;
};

/**
 * One measured execution cost for an operator/resource pair.
 *
 * Operator-level costs are used only by the offline segmentation policy
 * (Equation (1) in the FlexOn paper). Segment-level costs remain the
 * authoritative measurements for the generated artifact.
 */
struct OperatorCost {
    Resource resource{Resource::CPU};
    double mean_ms{0.0};
    double percentile_ms{0.0};
    ResourceSupportStatus status{ResourceSupportStatus::Supported};
    std::string reason;
};

/**
 * Complete cost profile for one operator.
 */
struct OperatorProfile {
    OperatorInfo op;
    std::vector<OperatorCost> costs;
};

using OperatorProfileMap =
    std::unordered_map<std::uint32_t, OperatorProfile>;

/**
 * Complete cost profile for one segment.
 */
struct SegmentProfile {
    SegmentInfo segment;
    std::vector<SegmentCost> costs;
};

/**
 * Diagnostics explaining why a generated segmentation level stopped.
 */
struct LevelStatistics {
    std::uint32_t level{0};
    std::uint32_t segment_count{0};
    double average_cost_ms{0.0};
};

/**
 * One observed online segment execution.
 */
struct ExecutionRecord {
    std::uint32_t level{0};
    std::uint32_t segment_id{0};
    Resource resource{Resource::CPU};
    double expected_ms{0.0};
    double measured_ms{0.0};
    double boundary_copy_ms{0.0};
    double scheduler_overhead_us{0.0};
};

/**
 * Runtime resource snapshot.
 */
struct ResourceSnapshot {
    Resource resource{Resource::CPU};
    double utilization_percent{0.0};
    double remaining_capacity{1.0};
};

}  // namespace flexon::core
