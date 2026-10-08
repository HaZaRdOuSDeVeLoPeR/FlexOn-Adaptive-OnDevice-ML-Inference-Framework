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

    // Standardized maximum-contention pass used to record per-segment
    // contended measurements are converted to per-segment/resource maximum
    // degradation ratios;
    // the runtime degradation ratio is intentionally calculated later.
    bool degradation_profiling_enabled{true};
    double degradation_cpu_percent{90.0};
    double degradation_gpu_percent{90.0};
    double degradation_dram_percent{90.0};
    double degradation_vram_percent{90.0};
    double degradation_safety_factor{0.85};
    std::uint32_t degradation_stabilization_seconds{10};

    // Explicit CUDA fallback overrides for the first resource-aware
    // partition. In normal operation, measured CUDA capability is used
    // automatically; this list is retained for reproducible overrides.
    std::vector<std::string> cuda_fallback_ops;
};

OfflineConfig load(const std::filesystem::path& path);

}  // namespace flexon::offline::config
