#include <flexon/offline/segmentation/multi_level_segmenter.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace flexon::offline::segmentation {

namespace {

double segment_cost(const core::SegmentProfile& profile) {
    // FlexOn defines E_i^l as the minimum profiled execution time across
    // available resources. Use the mean of the profiling trials as the
    // execution-time statistic, matching the paper's averaged profiling
    // measurements. The configured percentile remains available in the
    // artifact for diagnostics.
    if (profile.costs.empty()) {
        throw std::runtime_error(
            "Segment has no successful resource profiling result");
    }

    double cost = std::numeric_limits<double>::infinity();
    for (const auto& entry : profile.costs) {
        if (entry.status == core::ResourceSupportStatus::Supported &&
            std::isfinite(entry.mean_ms)) {
            cost = std::min(cost, entry.mean_ms);
        }
    }
    if (!std::isfinite(cost)) {
        throw std::runtime_error(
            "Segment has no usable resource profiling result");
    }
    return cost;
}

double operator_cost(
    const core::OperatorProfile& profile,
    const core::Resource resource) {

    for (const auto& entry : profile.costs) {
        if (entry.resource == resource &&
            entry.status == core::ResourceSupportStatus::Supported) {
            return entry.mean_ms;
        }
    }

    return std::numeric_limits<double>::infinity();
}

std::pair<core::SegmentInfo, core::SegmentInfo> balanced_split(
    const core::SegmentInfo& segment,
    const std::unordered_map<std::uint32_t, core::OperatorProfile>&
        operator_profiles,
    std::uint32_t next_id) {

    const auto count = segment.operator_indices.size();
    if (count < 2) {
        throw std::invalid_argument(
            "Cannot split a segment containing fewer than two operators");
    }

    // Equation (1):
    //
    // min_P max_{r1,r2} |
    //   sum_{j=0..P}     C_r1(op_j)
    // - sum_{j=P+1..n-1} C_r2(op_j)
    // |
    //
    // P is a boundary between operators, hence P ranges from 0 to n-2.
    // We evaluate every available resource pair exactly rather than using a
    // midpoint heuristic.
    std::vector<core::Resource> resources;
    resources.reserve(2);

    const auto first_profile_it =
        operator_profiles.find(segment.operator_indices.front());
    if (first_profile_it == operator_profiles.end()) {
        throw std::runtime_error(
            "Missing operator profile required for balanced split");
    }

    // Equation (1) is evaluated over resource pairs for which every
    // operator in the candidate segment has a measured cost. If a resource
    // failed profiling for one operator, it is not a valid pair for this
    // candidate split.
    for (const auto& first_cost : first_profile_it->second.costs) {
        bool available_for_all = true;

        for (const auto index : segment.operator_indices) {
            const auto it = operator_profiles.find(index);
            if (it == operator_profiles.end()) {
                throw std::runtime_error(
                    "Missing operator profile required for balanced split");
            }

            const bool found =
                std::any_of(
                    it->second.costs.begin(),
                    it->second.costs.end(),
                    [&](const core::OperatorCost& cost) {
                        return cost.resource == first_cost.resource &&
                               cost.status ==
                                   core::ResourceSupportStatus::Supported;
                    });

            if (!found) {
                available_for_all = false;
                break;
            }
        }

        if (available_for_all) {
            resources.push_back(first_cost.resource);
        }
    }

    if (resources.empty()) {
        throw std::runtime_error(
            "No common profiled resource is available for balanced split");
    }

    // Prefix sums for each resource.
    std::unordered_map<std::uint8_t, std::vector<double>> prefix;
    for (const auto resource : resources) {
        std::vector<double> sums(count, 0.0);

        for (std::size_t i = 0; i < count; ++i) {
            const auto it =
                operator_profiles.find(segment.operator_indices[i]);

            if (it == operator_profiles.end()) {
                throw std::runtime_error(
                    "Missing operator profile required for balanced split");
            }

            const double cost = operator_cost(it->second, resource);
            if (!std::isfinite(cost)) {
                throw std::runtime_error(
                    "Operator does not have a profiling result for every "
                    "resource used by the balanced split");
            }

            sums[i] = cost + (i == 0 ? 0.0 : sums[i - 1]);
        }

        prefix.emplace(
            static_cast<std::uint8_t>(resource), std::move(sums));
    }

    double best_objective = std::numeric_limits<double>::infinity();
    std::size_t best_boundary = 0;

    for (std::size_t boundary = 0; boundary + 1 < count; ++boundary) {
        double objective = 0.0;

        for (const auto left_resource : resources) {
            const auto& left_prefix =
                prefix.at(static_cast<std::uint8_t>(left_resource));
            const double left_cost = left_prefix[boundary];

            for (const auto right_resource : resources) {
                const auto& right_prefix =
                    prefix.at(static_cast<std::uint8_t>(right_resource));

                const double total_right =
                    right_prefix.back() - right_prefix[boundary];

                objective = std::max(
                    objective, std::abs(left_cost - total_right));
            }
        }

        // Strict comparison gives deterministic first-boundary tie breaking.
        if (objective < best_objective) {
            best_objective = objective;
            best_boundary = boundary;
        }
    }

    auto make_child = [&](std::size_t begin, std::size_t end,
                          std::uint32_t id) {
        core::SegmentInfo child;
        child.level = segment.level + 1;
        child.id = id;

        child.operator_indices.insert(
            child.operator_indices.end(),
            segment.operator_indices.begin() +
                static_cast<std::ptrdiff_t>(begin),
            segment.operator_indices.begin() +
                static_cast<std::ptrdiff_t>(end));

        return child;
    };

    return {
        make_child(0, best_boundary + 1, next_id),
        make_child(best_boundary + 1, count, next_id + 1)
    };
}

void populate_segment_metadata(
    core::SegmentInfo& segment,
    const core::GraphInfo& graph,
    std::uint32_t level,
    std::uint32_t id) {

    segment.level = level;
    segment.id = id;
    segment.input_tensors.clear();
    segment.output_tensors.clear();
    segment.operator_names.clear();

    for (const auto index : segment.operator_indices) {
        const auto& op = graph.operators.at(index);

        segment.operator_names.push_back(op.name);

        for (const auto& input : op.inputs) {
            const auto producer = graph.producer_operator.find(input);
            const bool internal =
                producer != graph.producer_operator.end() &&
                std::find(
                    segment.operator_indices.begin(),
                    segment.operator_indices.end(),
                    producer->second) !=
                    segment.operator_indices.end();

            if (!internal &&
                std::find(segment.input_tensors.begin(),
                          segment.input_tensors.end(), input) ==
                    segment.input_tensors.end()) {
                segment.input_tensors.push_back(input);
            }
        }

        for (const auto& output : op.outputs) {
            const auto consumers =
                graph.consumer_operators.find(output);

            bool outside = false;
            if (consumers != graph.consumer_operators.end()) {
                for (const auto consumer : consumers->second) {
                    if (std::find(
                            segment.operator_indices.begin(),
                            segment.operator_indices.end(),
                            consumer) ==
                        segment.operator_indices.end()) {
                        outside = true;
                        break;
                    }
                }
            }

            const bool graph_output =
                std::find(graph.graph_outputs.begin(),
                          graph.graph_outputs.end(), output) !=
                graph.graph_outputs.end();

            if ((outside || graph_output) &&
                std::find(segment.output_tensors.begin(),
                          segment.output_tensors.end(), output) ==
                    segment.output_tensors.end()) {
                segment.output_tensors.push_back(output);
            }
        }
    }
}

}  // namespace

SegmentationResult MultiLevelSegmenter::build(
    const onnx::ModelProto& model,
    const core::GraphInfo& graph,
    const std::vector<core::SegmentInfo>& initial,
    const core::OperatorProfileMap& operator_profiles,
    profiling::SegmentProfiler& profiler,
    const config::OfflineConfig& config) {

    SegmentationResult result;
    std::vector<core::SegmentInfo> current = initial;

    // Operator-level costs are profiled once before initial segmentation and
    // reused for every level. This is important because the first partition
    // must already respect resource capability boundaries.
    if (operator_profiles.size() != graph.operators.size()) {
        throw std::runtime_error(
            "Operator profile map is incomplete");
    }

    for (std::uint32_t level = 0;
         level < config.max_levels && !current.empty();
         ++level) {

        // Assign level-local IDs and regenerate boundary metadata.
        for (std::uint32_t i = 0;
             i < current.size(); ++i) {
            populate_segment_metadata(current[i], graph, level, i);
        }

        std::vector<core::SegmentProfile> profiles;
        profiles.reserve(current.size());

        double total_cost = 0.0;
        for (const auto& segment : current) {
            auto profile =
                profiler.profile(model, graph, segment);

            total_cost += segment_cost(profile);
            profiles.push_back(std::move(profile));
        }

        result.levels.push_back(profiles);

        // Algorithm 1 continues while the average execution time at the
        // current level exceeds T. The paper's pseudocode prints '<' here,
        // but its prose and the intended algorithm require E_bar_l > T.
        const double average_cost =
            total_cost / static_cast<double>(profiles.size());

        result.level_statistics.push_back({
            level,
            static_cast<std::uint32_t>(profiles.size()),
            average_cost});

        if (average_cost <=
            config.average_segment_cost_threshold_ms) {
            break;
        }

        // No next level may be created after reaching the configured limit.
        if (level + 1 >= config.max_levels) {
            break;
        }

        std::vector<core::SegmentInfo> next;
        next.reserve(current.size() * 2);
        std::uint32_t next_id = 0;

        for (std::size_t i = 0; i < current.size(); ++i) {
            const auto& segment = current[i];
            const double current_cost = segment_cost(profiles[i]);

            // A one-operator segment cannot be split further.
            if (segment.operator_indices.size() < 2) {
                auto retained = segment;
                retained.level = level + 1;
                retained.id = next_id++;
                next.push_back(std::move(retained));
                continue;
            }

            bool should_split = false;

            if (current.size() == 1) {
                // Algorithm 1's "average of other segments" is undefined
                // for the initial single-segment case. To allow segmentation
                // to bootstrap, the only segment is split whenever E_bar_l>T.
                should_split = true;
            } else {
                const double other_average =
                    (total_cost - current_cost) /
                    static_cast<double>(current.size() - 1);

                should_split = current_cost > other_average;
            }

            if (should_split) {
                auto children =
                    balanced_split(
                        segment, operator_profiles, next_id);

                next.push_back(std::move(children.first));
                next.push_back(std::move(children.second));
                next_id += 2;
            } else {
                // Carry unchanged, as specified by Algorithm 1.
                auto retained = segment;
                retained.level = level + 1;
                retained.id = next_id++;
                next.push_back(std::move(retained));
            }
        }

        // Prevent an infinite loop if all segments are indivisible or no
        // segment satisfies the strict "greater than other average" rule.
        if (next.size() == current.size()) {
            break;
        }

        current = std::move(next);
    }

    return result;
}

}  // namespace flexon::offline::segmentation
