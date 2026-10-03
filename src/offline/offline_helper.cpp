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

void usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [--all | --model <model.onnx>]\n"
        << "  " << program << " --validate [--artifact <directory> | --all]\n\n"
        << "Options:\n"
        << "  --all             Process every model in models.yaml.\n"
        << "                    Default when no generation selector is given.\n"
        << "  --model           Process one ONNX model.\n"
        << "  --validate        Validate existing artifacts only; never generate.\n"
        << "  --artifact        Explicit artifact directory for --model, or the\n"
        << "                    artifact to validate with --validate.\n"
        << "  --artifact-root   Root directory for generated/validated artifacts.\n"
        << "                    Default: artifacts\n"
        << "  --config          Offline profiling configuration.\n"
        << "                    Default: config/offline.yaml\n"
        << "  --models-config   Model catalog/runtime configuration.\n"
        << "                    Default: config/models.yaml\n"
        << "  --help, -h        Show this help message.\n";
}

bool artifact_is_valid(const std::filesystem::path& artifact_directory) {
    if (!std::filesystem::exists(artifact_directory) ||
        !std::filesystem::is_directory(artifact_directory)) {
        return false;
    }
    return flexon::offline::cli::validate_artifact(artifact_directory);
}

void generate_model(
    const std::filesystem::path& model_path,
    const std::filesystem::path& artifact_directory,
    const std::filesystem::path& config_path,
    const std::filesystem::path& models_config_path) {

    std::cout
        << "\n[offline] ========================================\n"
        << "[offline] model: " << model_path << '\n'
        << "[offline] artifact: " << artifact_directory << '\n'
        << "[offline] ========================================\n";

    if (artifact_is_valid(artifact_directory)) {
        std::cout << "[offline] artifact already exists and is valid; skipping\n";
        return;
    }

    flexon::offline::OfflineEngine engine(config_path, models_config_path);
    engine.run(model_path, artifact_directory);

    // Every fresh generation is validated before the process reports success.
    flexon::offline::cli::validate_artifact_or_throw(artifact_directory);
    std::cout << "[offline] generated artifact validated successfully\n";
}

}  // namespace flexon::offline::cli
