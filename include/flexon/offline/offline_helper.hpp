#pragma once

#include <filesystem>
#include <vector>

namespace flexon::offline::cli {

std::vector<std::filesystem::path> find_artifacts(
    const std::filesystem::path& root);

bool validate_artifact(
    const std::filesystem::path& artifact);

void validate_artifact_or_throw(
    const std::filesystem::path& artifact);

void validate_all_or_throw(
    const std::filesystem::path& root);

}  // namespace flexon::offline::cli
