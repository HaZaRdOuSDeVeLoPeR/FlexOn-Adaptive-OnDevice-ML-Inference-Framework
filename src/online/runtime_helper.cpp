#include <iostream>
#include <stdexcept>

#include <flexon/online/runtime_helper.hpp>

namespace flexon::online::cli {

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --artifact <dir> [options]\n\n"
        << "Required:\n"
        << "  --artifact <dir>                  Offline artifact to execute.\n\n"
        << "Runtime options:\n"
        << "  --iterations <N>                  Number of inference iterations.\n"
        << "                                    Default: 100\n"
        << "  --level <N>                       Initial segmentation level.\n"
        << "                                    Default: max_level / 2\n"
        << "  --resource <cpu|cuda|auto>        Resource selection mode.\n"
        << "                                    Default: auto\n"
        << "  --resource-plan <plan>            Fixed comma-separated cpu/cuda plan, or auto.\n"
        << "                                    Default: auto\n"
        << "  --adaptive-level                  Enable adaptive segmentation-level selection.\n"
        << "                                    Default: false\n"
        << "  --alpha <value>                   Override scheduler alpha.\n"
        << "                                    Default: 1.2\n"
        << "  --beta <value>                    Override scheduler beta.\n"
        << "                                    Default: 1.6\n"
        << "  --gamma <value>                   Override scheduler gamma.\n"
        << "                                    Default: 1.3\n"
        << "  --with-recovery                   Enable speculative recovery.\n"
        << "                                    Default: false\n"
        << "  --priority-isolation              Scheduler/control thread uses maximum Linux real-time priority\n"
        << "                                    inference stays normal.\n"
        << "  --scheduler-config <p>            Scheduler configuration file.\n"
        << "                                    Default: config/scheduler.yaml\n"
        << "  --help, -h                        Show this help message.\n";
}

std::vector<core::Resource> parse_resource_plan(const std::string& value) {
    if (value == "auto") {
        return {};
    }

    std::vector<core::Resource> plan;
    std::size_t start = 0;
    while (start < value.size()) {
        const auto comma = value.find(',', start);
        const auto token = value.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start);

        if (token == "cpu") {
            plan.push_back(core::Resource::CPU);
        } else if (token == "cuda") {
            plan.push_back(core::Resource::CUDA);
        } else if (token == "auto") {
            throw std::invalid_argument(
                "resource-plan entries must be cpu or cuda; use 'auto' for automatic selection");
        } else if (token.empty()) {
            throw std::invalid_argument("Empty resource-plan entry");
        } else {
            throw std::invalid_argument(
                "Unknown resource-plan entry: " + token);
        }

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;

        if (start == value.size()) {
            throw std::invalid_argument("Empty resource-plan entry");
        }
    }

    if (plan.empty()) {
        throw std::invalid_argument("Resource plan cannot be empty");
    }
    return plan;
}

core::Resource parse_resource(const std::string& value) {
    if (value == "cpu") return core::Resource::CPU;
    if (value == "cuda") return core::Resource::CUDA;
    if (value == "auto") return core::Resource::Auto;
    throw std::invalid_argument("Unknown resource: " + value);
}

}  // namespace flexon::online::cli
