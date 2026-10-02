#include <flexon/online/runtime_engine.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage(const char* program) {
    std::cout << "Usage: " << program
              << " --artifact <dir> [--level N] [--resource cpu|cuda|auto]"
                 " [--iterations N] [--resource-plan cpu,cuda,...] [--adaptive-level] [--scheduler-config path]\n";
}

std::vector<flexon::online::RuntimeResource> resource_plan_from_string(
    const std::string& value) {
    std::vector<flexon::online::RuntimeResource> plan;
    std::size_t start = 0;
    while (start < value.size()) {
        const auto comma = value.find(',', start);
        const auto token = value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (token == "cpu") plan.push_back(flexon::online::RuntimeResource::CPU);
        else if (token == "cuda") plan.push_back(flexon::online::RuntimeResource::CUDA);
        else throw std::invalid_argument(
            "resource-plan entries must be cpu or cuda: " + token);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return plan;
}

flexon::online::RuntimeResource resource_from_string(const std::string& value) {
    if (value == "cpu") return flexon::online::RuntimeResource::CPU;
    if (value == "cuda") return flexon::online::RuntimeResource::CUDA;
    if (value == "auto") return flexon::online::RuntimeResource::Auto;
    throw std::invalid_argument("Unknown resource: " + value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact;
        flexon::online::RunOptions options;
        options.iterations = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--artifact" && i + 1 < argc) {
                artifact = argv[++i];
            } else if (arg == "--level" && i + 1 < argc) {
                options.level =
                    static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--resource" && i + 1 < argc) {
                options.resource = resource_from_string(argv[++i]);
            } else if (arg == "--iterations" && i + 1 < argc) {
                options.iterations =
                    static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--resource-plan" && i + 1 < argc) {
                options.resource_plan =
                    resource_plan_from_string(argv[++i]);
            } else if (arg == "--adaptive-level") {
                options.adaptive_level = true;
            } else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                usage(argv[0]);
                return 2;
            }
        }

        if (artifact.empty()) {
            usage(argv[0]);
            return 2;
        }

        if (options.adaptive_level && !options.resource_plan.empty()) {
            throw std::invalid_argument(
                "--adaptive-level cannot be combined with a fixed resource plan");
        }

        flexon::online::FlexOnRuntime runtime;
        runtime.load(artifact);

        if (!runtime.loaded()) {
            throw std::runtime_error(
                "Runtime reports not-loaded after load()");
        }

        if (runtime.level_count() == 0) {
            throw std::runtime_error(
                "Runtime reports zero levels");
        }

        if (options.level >= runtime.level_count()) {
            throw std::runtime_error(
                "Requested test level does not exist");
        }

        runtime.run(options);

        std::cout << "[tester] PASS: artifact loaded and level "
                  << options.level << " executed successfully\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "[tester] FAIL: " << ex.what() << '\n';
        return 1;
    }
}
