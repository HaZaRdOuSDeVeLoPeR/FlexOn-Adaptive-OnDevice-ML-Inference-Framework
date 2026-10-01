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
           " [--iterations N]\n";
}

flexon::online::RuntimeResource parse_resource(const std::string& value) {
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
