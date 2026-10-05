#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <flexon/offline/config/model_config.hpp>
#include <flexon/offline/offline_engine.hpp>
#include <flexon/offline/offline_helper.hpp>

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
                flexon::offline::cli::usage(argv[0]);
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

            flexon::offline::cli::generate_model(
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
                if (flexon::offline::cli::artifact_is_valid(artifact_directory)) {
                    std::cout << "\n[offline] " << model_spec.name
                              << ": valid artifact exists; skipping\n";
                    ++skipped;
                    continue;
                }

                flexon::offline::cli::generate_model(
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
