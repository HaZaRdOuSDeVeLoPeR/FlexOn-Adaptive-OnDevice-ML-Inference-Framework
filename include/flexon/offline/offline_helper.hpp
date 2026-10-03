#pragma once

#include <filesystem>
#include <vector>

#include <flexon/offline/offline_engine.hpp>

namespace flexon::offline::cli {

std::vector<std::filesystem::path> find_artifacts(
    const std::filesystem::path& root);

bool validate_artifact(
    const std::filesystem::path& artifact);

void validate_artifact_or_throw(
    const std::filesystem::path& artifact);

void validate_all_or_throw(
    const std::filesystem::path& root);

void usage(const char* program);
bool artifact_is_valid(const std::filesystem::path& artifact_directory);

void generate_model(
    const std::filesystem::path& model_path,
    const std::filesystem::path& artifact_directory,
    const std::filesystem::path& config_path,
    const std::filesystem::path& models_config_path);

}  // namespace flexon::offline::cli
