#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <flexon/offline/offline_helper.hpp>
#include <flexon/offline/artifact/artifact_validator.hpp>

namespace flexon::offline::cli {

std::vector<std::filesystem::path> find_artifacts(
    const std::filesystem::path& root) {

    if (!std::filesystem::exists(root)) {
        std::cout << "Artifact root does not exist. Creating directory: " << root.string() << std::endl;
        
        std::error_code ec;
        // create_directories creates the folder and any missing parent paths
        if (!std::filesystem::create_directories(root, ec)) {
            // Double-check if it failed because it suddenly exists (race condition) or if it truly failed
            if (!std::filesystem::exists(root)) {
                throw std::runtime_error(
                    "Failed to create artifact root: " + root.string() + " (Reason: " + ec.message() + ")");
            }
        }
    }

    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error(
            "Artifact root is not a directory: " + root.string());
    }

    std::vector<std::filesystem::path> artifacts;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }
        if (entry.path().filename() == ".work") {
            continue;
        }
        artifacts.push_back(entry.path());
    }

    std::sort(artifacts.begin(), artifacts.end());
    return artifacts;
}

bool validate_artifact(const std::filesystem::path& artifact) {
    try {
        flexon::offline::artifact::ArtifactValidator::validate(artifact);
        std::cout << "[offline] validation PASS: " << artifact << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cerr << "[offline] validation FAIL: " << artifact
                  << " -> " << error.what() << '\n';
        return false;
    }
}

void validate_artifact_or_throw(const std::filesystem::path& artifact) {
    try {
        flexon::offline::artifact::ArtifactValidator::validate(artifact);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Artifact validation failed for '" + artifact.string() + "': " +
            error.what());
    }
}

void validate_all_or_throw(const std::filesystem::path& root) {
    const auto artifacts = find_artifacts(root);
    if (artifacts.empty()) {
        throw std::runtime_error(
            "No artifacts found under '" + root.string() + "'. "
            "Run artifact generation first.");
    }

    std::size_t failed = 0;
    for (const auto& artifact : artifacts) {
        if (!validate_artifact(artifact)) {
            ++failed;
        }
    }

    if (failed != 0) {
        throw std::runtime_error(
            std::to_string(failed) + " artifact(s) failed validation");
    }
}

}  // namespace flexon::offline::cli
