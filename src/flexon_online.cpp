#include <flexon/online/runtime_engine.hpp>
#include <flexon/online/runtime_helper.hpp>

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

template <typename T>
T require_value(int& index, int argc, char** argv, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + option);
    }
    return T(argv[++index]);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact;
        flexon::online::RunOptions options;
        options.iterations = 100;
        options.resource = flexon::core::Resource::Auto;
        options.adaptive_level = false;

        std::optional<double> alpha;
        std::optional<double> beta;
        std::optional<double> gamma;
        bool with_recovery = false;
        bool level_explicit = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                flexon::online::cli::print_usage(argv[0]);
                return 0;
            }

            if (arg == "--artifact") {
                artifact = require_value<std::string>(i, argc, argv, arg);
            } else if (arg == "--iterations") {
                options.iterations = static_cast<std::uint32_t>(
                    std::stoul(require_value<std::string>(i, argc, argv, arg)));
            } else if (arg == "--level") {
                options.level = static_cast<std::uint32_t>(
                    std::stoul(require_value<std::string>(i, argc, argv, arg)));
                level_explicit = true;
            } else if (arg == "--resource") {
                options.resource = flexon::online::cli::parse_resource(
                    require_value<std::string>(i, argc, argv, arg));
            } else if (arg == "--resource-plan") {
                options.resource_plan = flexon::online::cli::parse_resource_plan(
                    require_value<std::string>(i, argc, argv, arg));
            } else if (arg == "--adaptive-level") {
                options.adaptive_level = true;
            } else if (arg == "--alpha") {
                alpha = std::stod(
                    require_value<std::string>(i, argc, argv, arg));
            } else if (arg == "--beta") {
                beta = std::stod(
                    require_value<std::string>(i, argc, argv, arg));
            } else if (arg == "--gamma") {
                gamma = std::stod(
                    require_value<std::string>(i, argc, argv, arg));
            } else if (arg == "--with-recovery") {
                with_recovery = true;
            } else if (arg == "--scheduler-config") {
                options.scheduler_config_path = require_value<std::string>(
                    i, argc, argv, arg);
            } else {
                flexon::online::cli::print_usage(argv[0]);
                return 2;
            }
        }

        if (artifact.empty()) {
            flexon::online::cli::print_usage(argv[0]);
            return 2;
        }

        if (options.adaptive_level && !options.resource_plan.empty()) {
            throw std::invalid_argument(
                "--adaptive-level cannot be combined with a fixed resource plan");
        }

        options.scheduler_alpha_override = alpha;
        options.scheduler_beta_override = beta;
        options.scheduler_gamma_override = gamma;
        options.with_recovery = with_recovery;

        flexon::online::FlexOnRuntime runtime;
        runtime.load(artifact);

        if (runtime.level_count() == 0) {
            throw std::runtime_error("Artifact contains zero segmentation levels");
        }

        if (!level_explicit) {
            // The artifact levels are zero-indexed; max_level / 2 is therefore
            // represented by (level_count - 1) / 2.
            options.level = (runtime.level_count() - 1U) / 2U;
        }
        if (options.level >= runtime.level_count()) {
            throw std::runtime_error("Requested initial level does not exist");
        }

        runtime.run(options);

        std::cout << "[online] execution completed successfully\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[online] ERROR: " << ex.what() << '\n';
        return 1;
    }
}
