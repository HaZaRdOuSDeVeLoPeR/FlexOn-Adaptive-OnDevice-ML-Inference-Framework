#include <flexon/online/runtime_engine.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* program) {
    std::cout << "Usage: " << program
              << " --artifact <dir> [--level N] [--resource cpu|cuda|auto]"
                 " [--iterations N]\n";
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
                options.level = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--resource" && i + 1 < argc) {
                options.resource = resource_from_string(argv[++i]);
            } else if (arg == "--iterations" && i + 1 < argc) {
                options.iterations = static_cast<std::uint32_t>(std::stoul(argv[++i]));
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

        flexon::online::FlexOnRuntime runtime;
        runtime.load(artifact);

        if (!runtime.loaded()) {
            throw std::runtime_error("Runtime reports not-loaded after load()");
        }
        if (runtime.level_count() == 0) {
            throw std::runtime_error("Runtime reports zero levels");
        }
        if (options.level >= runtime.level_count()) {
            throw std::runtime_error("Requested test level does not exist");
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
