#pragma once

#include <filesystem>

namespace flexon::offline {

/**
 * Public entry point for the offline phase.
 *
 * The engine processes one model at a time. Batch orchestration is handled
 * by the flexon_offline application.
 *
 * Pipeline:
 * graph analysis -> profiling -> initial segmentation ->
 * multi-level segmentation -> executable artifact generation -> validation.
 */
class OfflineEngine {
public:
    explicit OfflineEngine(
        std::filesystem::path config_path = "config/offline.yaml",
        std::filesystem::path models_config_path =
            "config/models.yaml");

    /**
     * Run the complete offline pipeline for one model.
     *
     * Runtime/input-shape configuration is loaded from models.yaml.
     */
    void run(
        const std::filesystem::path& model_path,
        const std::filesystem::path& artifact_directory);

private:
    std::filesystem::path config_path_;
    std::filesystem::path models_config_path_;
};

}  // namespace flexon::offline
