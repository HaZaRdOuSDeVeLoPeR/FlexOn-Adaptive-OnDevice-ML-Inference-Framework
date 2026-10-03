#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <flexon/online/artifact/manifest_loader.hpp>

namespace {

struct TempArtifact {
    std::filesystem::path root;

    TempArtifact()
        : root(std::filesystem::temp_directory_path() /
               ("flexon_artifact_handler_test_" +
                std::to_string(std::rand()))) {
        std::filesystem::create_directories(root);
    }

    ~TempArtifact() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
};

void write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream file(path);
    assert(file);
    file << text;
}

void expect_load_failure(const std::filesystem::path& root) {
    bool threw = false;
    try {
        (void)flexon::online::manifest::load_manifest(root);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    // Missing manifest.
    {
        TempArtifact artifact;
        expect_load_failure(artifact.root);
    }

    // Unsupported format.
    {
        TempArtifact artifact;
        write(artifact.root / "manifest.yaml",
              "format_version: 1\n"
              "model:\n  name: test\n"
              "levels: []\n");
        expect_load_failure(artifact.root);
    }

    // Missing model/levels.
    {
        TempArtifact artifact;
        write(artifact.root / "manifest.yaml", "format_version: 2\n");
        expect_load_failure(artifact.root);
    }

    // Valid minimal artifact. Segment model existence is checked by the
    // loader, so a placeholder file is sufficient for this unit test.
    {
        TempArtifact artifact;
        std::filesystem::create_directories(artifact.root / "segments");
        write(artifact.root / "segments" / "segment_0.onnx", "placeholder");
        write(artifact.root / "manifest.yaml",
              "format_version: 2\n"
              "model:\n"
              "  name: unit_test_model\n"
              "levels:\n"
              "  - level: 2\n"
              "    segments:\n"
              "      - id: 7\n"
              "        model: segments/segment_0.onnx\n"
              "        inputs: [input]\n"
              "        outputs: [output]\n"
              "        costs:\n"
              "          - resource: cpu\n"
              "            status: supported\n"
              "            mean_ms: 1.5\n"
              "          - resource: cuda\n"
              "            status: supported\n"
              "            mean_ms: 0.5\n");

        const auto manifest =
            flexon::online::manifest::load_manifest(artifact.root);

        assert(manifest.directory == artifact.root);
        assert(manifest.model_name == "unit_test_model");
        assert(manifest.levels.size() == 1);
        assert(manifest.levels[0].level == 2);
        assert(manifest.levels[0].segments.size() == 1);

        const auto& segment = manifest.levels[0].segments[0];
        assert(segment.id == 7);
        assert(segment.input_names.size() == 1);
        assert(segment.input_names[0] == "input");
        assert(segment.output_names.size() == 1);
        assert(segment.output_names[0] == "output");
        assert(segment.cpu_supported);
        assert(segment.cuda_supported);
        assert(segment.cpu_mean_ms == 1.5);
        assert(segment.cuda_mean_ms == 0.5);
    }

    // A segment with no supported resources must be rejected.
    {
        TempArtifact artifact;
        std::filesystem::create_directories(artifact.root / "segments");
        write(artifact.root / "segments" / "segment_0.onnx", "placeholder");
        write(artifact.root / "manifest.yaml",
              "format_version: 2\n"
              "model:\n  name: unsupported\n"
              "levels:\n"
              "  - level: 0\n"
              "    segments:\n"
              "      - id: 0\n"
              "        model: segments/segment_0.onnx\n"
              "        costs:\n"
              "          - resource: cuda\n"
              "            status: unsupported\n");
        expect_load_failure(artifact.root);
    }

    return 0;
}
