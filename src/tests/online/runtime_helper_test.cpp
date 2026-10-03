#include <cassert>
#include <stdexcept>

#include <flexon/core/types.hpp>
#include <flexon/online/runtime_helper.hpp>

int main() {
    using flexon::core::Resource;
    using flexon::online::cli::parse_resource;
    using flexon::online::cli::parse_resource_plan;

    assert(parse_resource("cpu") == Resource::CPU);
    assert(parse_resource("cuda") == Resource::CUDA);
    assert(parse_resource("auto") == Resource::Auto);

    for (const char* value : {"gpu", "CPU", "", "cuda,cpu"}) {
        bool threw = false;
        try {
            (void)parse_resource(value);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    assert(parse_resource_plan("auto").empty());

    const auto cpu_cuda = parse_resource_plan("cpu,cuda");
    assert((cpu_cuda == std::vector<Resource>{Resource::CPU, Resource::CUDA}));

    const auto cuda_cpu_cpu = parse_resource_plan("cuda,cpu,cpu");
    assert((cuda_cpu_cpu == std::vector<Resource>{
        Resource::CUDA, Resource::CPU, Resource::CPU}));

    for (const char* value : {
             "", ",cpu", "cpu,", "cpu,,cuda", "cpu,auto", "gpu"}) {
        bool threw = false;
        try {
            (void)parse_resource_plan(value);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    return 0;
}
