#include <flexon/offline/config/model_config.hpp>
#include <flexon/offline/offline_engine.hpp>
#include <flexon/offline/offline_helper.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model;
    std::filesystem::path artifact;
    std::filesystem::path artifact_root = "artifacts";
    std::filesystem::path config = "config/offline.yaml";
    std::filesystem::path models_config = "config/models.yaml";

    bool all = false;
    bool validate_only = false;

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

            if (arg == "--validate") {
                validate_only = true;
                continue;
            }

            if (arg == "--model" ||
                arg == "--artifact" ||
                arg == "--artifact-root" ||
                arg == "--config" ||
                arg == "--models-config") {

                if (i + 1 >= argc) {
                    throw std::invalid_argument("Missing value for " + arg);
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

            throw std::invalid_argument("Unknown argument: " + arg);
        }

        if (validate_only) {
            if (!model.empty()) {
                throw std::invalid_argument(
                    "--validate cannot be combined with --model or generation options");
            }
            if (all && !artifact.empty()) {
                throw std::invalid_argument(
                    "--all and --artifact cannot be used together");
            }

            if (!artifact.empty()) {
                flexon::offline::cli::validate_artifact_or_throw(artifact);
            } else {
                // --validate alone means validate every existing artifact.
                flexon::offline::cli::validate_all_or_throw(artifact_root);
            }
            return 0;
        }

        if (all && !model.empty()) {
            throw std::invalid_argument(
                "--all and --model cannot be used together");
        }

        if (all && !artifact.empty()) {
            throw std::invalid_argument(
                "--artifact cannot be used with --all; use --artifact-root instead");
        }

        // No generation selector means --all.
        if (!all && model.empty()) {
            all = true;
        }

        if (!all) {
            std::filesystem::path artifact_directory = artifact;
            if (artifact_directory.empty()) {
                artifact_directory = artifact_root / model.stem().string();
            }

            generate_model(
                model,
                artifact_directory,
                config,
                models_config);
            return 0;
        }

        const auto models_config_data =
            flexon::offline::config::load_models_config(models_config);

        std::size_t generated = 0;
        std::size_t skipped = 0;
        std::size_t failed = 0;

        for (const auto& model_spec : models_config_data.models) {
            const auto artifact_directory = artifact_root / model_spec.name;

            try {
                if (artifact_is_valid(artifact_directory)) {
                    std::cout << "\n[offline] " << model_spec.name
                              << ": valid artifact exists; skipping\n";
                    ++skipped;
                    continue;
                }

                generate_model(
                    model_spec.path,
                    artifact_directory,
                    config,
                    models_config);
                ++generated;
            } catch (const std::exception& error) {
                ++failed;
                std::cerr << "\n[offline] ERROR processing model '"
                          << model_spec.name << "': " << error.what() << '\n';
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
        std::cerr << "[offline] ERROR: " << error.what() << '\n';
        return 1;
    }
}
