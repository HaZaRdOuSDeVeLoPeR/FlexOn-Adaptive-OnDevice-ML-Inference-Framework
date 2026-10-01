#include <flexon/offline/artifact/artifact_validator.hpp>
#include <flexon/offline/config/model_config.hpp>
#include <flexon/offline/offline_engine.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program
        << " --model <model.onnx>"
        << " [--artifact <directory>]"
        << " [--artifact-root <directory>]"
        << " [--config <config/offline.yaml>]"
        << " [--models-config <config/models.yaml>]\n\n"

        << "  " << program
        << " --all"
        << " [--artifact-root <directory>]"
        << " [--config <config/offline.yaml>]"
        << " [--models-config <config/models.yaml>]\n\n"

        << "Options:\n"
        << "  --model           Process one ONNX model.\n"
        << "  --all             Process every model in models.yaml.\n"
        << "  --artifact        Explicit artifact directory for --model.\n"
        << "  --artifact-root   Root directory for generated artifacts.\n"
        << "                    Default: artifacts\n"
        << "  --config          Offline profiling configuration.\n"
        << "                    Default: config/offline.yaml\n"
        << "  --models-config   Model catalog/runtime configuration.\n"
        << "                    Default: config/models.yaml\n"
        << "  --help, -h        Show this help message.\n";
}

bool artifact_is_valid(
    const std::filesystem::path& artifact_directory) {

    if (!std::filesystem::exists(artifact_directory) ||
        !std::filesystem::is_directory(artifact_directory)) {
        return false;
    }

    try {
        flexon::offline::artifact::ArtifactValidator::validate(
            artifact_directory);

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void process_model(
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
        std::cout
            << "[offline] artifact already exists and is valid; "
            << "skipping\n";

        return;
    }

    flexon::offline::OfflineEngine engine(
        config_path,
        models_config_path);

    engine.run(
        model_path,
        artifact_directory);
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model;
    std::filesystem::path artifact;
    std::filesystem::path artifact_root = "artifacts";
    std::filesystem::path config = "config/offline.yaml";
    std::filesystem::path models_config = "config/models.yaml";

    bool all = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            }

            if (arg == "--all") {
                all = true;
                continue;
            }

            if (arg == "--model" ||
                arg == "--artifact" ||
                arg == "--artifact-root" ||
                arg == "--config" ||
                arg == "--models-config") {

                if (i + 1 >= argc) {
                    throw std::invalid_argument(
                        "Missing value for " + arg);
                }

                const std::filesystem::path value = argv[++i];

                if (arg == "--model") {
                    model = value;
                } else if (arg == "--artifact") {
                    artifact = value;
                } else if (arg == "--artifact-root") {
                    artifact_root = value;
                } else if (arg == "--config") {
                    config = value;
                } else {
                    models_config = value;
                }

                continue;
            }

            throw std::invalid_argument(
                "Unknown argument: " + arg);
        }

        if (all && !model.empty()) {
            throw std::invalid_argument(
                "--all and --model cannot be used together");
        }

        if (all && !artifact.empty()) {
            throw std::invalid_argument(
                "--artifact cannot be used with --all; "
                "use --artifact-root instead");
        }

        if (!all && model.empty()) {
            usage(argv[0]);
            return 2;
        }

        // ---------------------------------------------------------------
        // Single-model mode
        // ---------------------------------------------------------------

        if (!all) {
            std::filesystem::path artifact_directory = artifact;

            if (artifact_directory.empty()) {
                const auto model_name =
                    model.stem().string();

                artifact_directory =
                    artifact_root / model_name;
            }

            process_model(
                model,
                artifact_directory,
                config,
                models_config);

            return 0;
        }

        // ---------------------------------------------------------------
        // Batch mode
        // ---------------------------------------------------------------

        const auto models_config_data =
            flexon::offline::config::load_models_config(
                models_config);

        std::size_t generated = 0;
        std::size_t skipped = 0;
        std::size_t failed = 0;

        for (const auto& model_spec :
             models_config_data.models) {

            const auto artifact_directory =
                artifact_root / model_spec.name;

            try {
                if (artifact_is_valid(artifact_directory)) {
                    std::cout
                        << "\n[offline] "
                        << model_spec.name
                        << ": valid artifact exists; skipping\n";

                    ++skipped;
                    continue;
                }

                process_model(
                    model_spec.path,
                    artifact_directory,
                    config,
                    models_config);

                ++generated;

            } catch (const std::exception& error) {
                ++failed;

                std::cerr
                    << "\n[offline] ERROR processing model '"
                    << model_spec.name
                    << "': "
                    << error.what()
                    << '\n';
            }
        }

        std::cout
            << "\n[offline] ========================================\n"
            << "[offline] batch summary\n"
            << "[offline] generated: " << generated << '\n'
            << "[offline] skipped:   " << skipped << '\n'
            << "[offline] failed:    " << failed << '\n'
            << "[offline] ========================================\n";

        return failed == 0 ? 0 : 1;

    } catch (const std::exception& error) {
        std::cerr
            << "[offline] ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}
