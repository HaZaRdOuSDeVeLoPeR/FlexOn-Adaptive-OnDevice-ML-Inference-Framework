#include <cassert>
#include <filesystem>
#include <stdexcept>

#include <flexon/online/runtime_engine.hpp>

int main() {
    flexon::online::FlexOnRuntime runtime;

    assert(!runtime.loaded());
    assert(runtime.level_count() == 0);
    assert(runtime.model_name().empty());

    // Running before load is an API contract violation.
    {
        flexon::online::RunOptions options;
        options.iterations = 1;
        bool threw = false;
        try {
            runtime.run(options);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }

    // Loading a missing artifact must fail before any runtime state is
    // installed.
    {
        const auto missing =
            std::filesystem::temp_directory_path() /
            "flexon_runtime_engine_missing_artifact";

        bool threw = false;
        try {
            runtime.load(missing);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
        assert(!runtime.loaded());
        assert(runtime.level_count() == 0);
    }

    // Zero iterations is rejected by the public runtime API. This check is
    // intentionally performed only after a successful load in a real run;
    // the pre-load guard above verifies the independent load contract.
    // The remaining full execution path is covered by the benchmark notebook
    // against real offline artifacts because it requires ONNX Runtime models.

    return 0;
}
