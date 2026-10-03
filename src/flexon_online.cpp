#include <flexon/online/runtime_engine.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --artifact <dir> [--level N] [--resource cpu|cuda|auto]"
           " [--iterations N] [--resource-plan cpu,cuda,...]\n";
}

std::vector<flexon::core::Resource> parse_resource_plan(
    const std::string& value) {
    std::vector<flexon::core::Resource> plan;
    std::size_t start = 0;
    while (start < value.size()) {
        const auto comma = value.find(',', start);
        const auto token = value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (token.empty()) {
            throw std::invalid_argument("Empty resource-plan entry");
        }
        if (token == "cpu") plan.push_back(flexon::core::Resource::CPU);
        else if (token == "cuda") plan.push_back(flexon::core::Resource::CUDA);
        else if (token == "auto") plan.push_back(flexon::core::Resource::Auto);
        else throw std::invalid_argument("Unknown resource in plan: " + token);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return plan;
}

flexon::core::Resource parse_resource(const std::string& value) {
    if (value == "cpu") return flexon::core::Resource::CPU;
    if (value == "cuda") return flexon::core::Resource::CUDA;
    if (value == "auto") return flexon::core::Resource::Auto;
    throw std::invalid_argument("Unknown resource: " + value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string artifact;
        flexon::online::RunOptions options;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--artifact" && i + 1 < argc) {
                artifact = argv[++i];
            } else if (arg == "--level" && i + 1 < argc) {
                options.level = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--resource" && i + 1 < argc) {
                options.resource = parse_resource(argv[++i]);
            } else if (arg == "--iterations" && i + 1 < argc) {
                options.iterations = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--resource-plan" && i + 1 < argc) {
                options.resource_plan = parse_resource_plan(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                print_usage(argv[0]);
                return 2;
            }
        }

        if (artifact.empty()) {
            print_usage(argv[0]);
            return 2;
        }

        flexon::online::FlexOnRuntime runtime;
        runtime.load(artifact);
        runtime.run(options);

        std::cout << "[online] execution completed successfully\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[online] ERROR: " << ex.what() << '\n';
        return 1;
    }
}
