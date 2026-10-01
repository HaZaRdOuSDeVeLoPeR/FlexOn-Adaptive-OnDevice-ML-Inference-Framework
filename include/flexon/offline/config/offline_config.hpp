#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace flexon::offline::config {

struct OfflineConfig {
    std::uint32_t warmup_iterations{10};
    std::uint32_t measurement_iterations{50};
    double percentile{95.0};

    std::uint32_t max_levels{5};
    double average_segment_cost_threshold_ms{0.1};
    std::string split_policy{"max_min_balance"};
    std::string initial_partition{"fallback_operator"};

    double numerical_tolerance{1.0e-4};

    bool cpu_enabled{true};
    bool cuda_enabled{true};
    int cuda_device_id{0};
    std::string artifact_root{"artifacts"};

    // Explicit CUDA fallback overrides for the first resource-aware
    // partition. In normal operation, measured CUDA capability is used
    // automatically; this list is retained for reproducible overrides.
    std::vector<std::string> cuda_fallback_ops;
};

OfflineConfig load(const std::filesystem::path& path);

}  // namespace flexon::offline::config
