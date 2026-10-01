#include <flexon/offline/artifact/artifact_validator.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace {

void usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << "\n"
        << "  " << program << " --all"
        << " [--artifact-root <directory>]\n"
        << "  " << program << " --artifact <directory>\n\n"

        << "Options:\n"
        << "  --all             Validate all artifact directories.\n"
        << "  --artifact        Validate one artifact directory.\n"
        << "  --artifact-root   Root containing artifact directories.\n"
        << "                    Default: artifacts\n"
        << "  --help, -h        Show this help message.\n";
}

std::vector<std::filesystem::path> find_artifacts(
    const std::filesystem::path& root) {

    if (!std::filesystem::exists(root)) {
        throw std::runtime_error(
            "Artifact root does not exist: " +
            root.string());
    }

    if (!std::filesystem::is_directory(root)) {
        throw std::runtime_error(
            "Artifact root is not a directory: " +
            root.string());
    }

    std::vector<std::filesystem::path> artifacts;

    for (const auto& entry :
         std::filesystem::directory_iterator(root)) {

        if (!entry.is_directory()) {
            continue;
        }

        // .work is an internal temporary directory and is not
        // an offline artifact.
        if (entry.path().filename() == ".work") {
            continue;
        }

        artifacts.push_back(entry.path());
    }

    std::sort(
        artifacts.begin(),
        artifacts.end());

    return artifacts;
}

bool validate_one(
    const std::filesystem::path& artifact) {

    try {
        flexon::offline::artifact::ArtifactValidator::validate(
            artifact);

        std::cout
            << "[tester] PASS: "
            << artifact
            << '\n';

        return true;

    } catch (const std::exception& error) {
        std::cerr
            << "[tester] FAIL: "
            << artifact
            << " -> "
            << error.what()
            << '\n';

        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path artifact;
    std::filesystem::path artifact_root = "artifacts";

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

            if (arg == "--artifact" ||
                arg == "--artifact-root") {

                if (i + 1 >= argc) {
                    throw std::invalid_argument(
                        "Missing value for " + arg);
                }

                const std::filesystem::path value =
                    argv[++i];

                if (arg == "--artifact") {
                    artifact = value;
                } else {
                    artifact_root = value;
                }

                continue;
            }

            throw std::invalid_argument(
                "Unknown argument: " + arg);
        }

        if (!artifact.empty() && all) {
            throw std::invalid_argument(
                "--artifact and --all cannot be used together");
        }

        // ---------------------------------------------------------------
        // Single artifact mode
        // ---------------------------------------------------------------

        if (!artifact.empty()) {
            if (validate_one(artifact)) {
                std::cout
                    << "Offline artifact validation passed.\n";

                return 0;
            }

            return 1;
        }

        // ---------------------------------------------------------------
        // Batch mode
        //
        // No explicit artifact means validate the artifact root.
        // --all is therefore optional and primarily improves readability
        // in scripts.
        // ---------------------------------------------------------------

        const auto artifacts =
            find_artifacts(artifact_root);

        if (artifacts.empty()) {
            std::cerr
                << "[tester] No artifact directories found under "
                << artifact_root
                << '\n';

            return 1;
        }

        std::size_t passed = 0;
        std::size_t failed = 0;

        for (const auto& artifact_directory :
             artifacts) {

            if (validate_one(artifact_directory)) {
                ++passed;
            } else {
                ++failed;
            }
        }

        std::cout
            << "\n[tester] ========================================\n"
            << "[tester] validation summary\n"
            << "[tester] passed: " << passed << '\n'
            << "[tester] failed: " << failed << '\n'
            << "[tester] ========================================\n";

        if (failed == 0) {
            std::cout
                << "Offline artifact validation passed.\n";

            return 0;
        }

        std::cerr
            << "Offline artifact validation failed.\n";

        return 1;

    } catch (const std::exception& error) {
        std::cerr
            << "[tester] ERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}
