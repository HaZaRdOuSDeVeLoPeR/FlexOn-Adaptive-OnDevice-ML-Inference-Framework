#include <flexon/offline/model/onnx_model_loader.hpp>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

std::filesystem::path make_test_model() {
    const auto path =
        std::filesystem::temp_directory_path() /
        "flexon_onnx_model_loader_test.onnx";

    onnx::ModelProto model;
    model.set_ir_version(7);
    model.set_producer_name("FlexOnTest");

    auto* graph = model.mutable_graph();
    graph->set_name("loader_test_graph");

    auto* input = graph->add_input();
    input->set_name("X");

    auto* output = graph->add_output();
    output->set_name("Y");

    auto* node = graph->add_node();
    node->set_name("Identity_0");
    node->set_op_type("Identity");
    node->add_input("X");
    node->add_output("Y");

    std::ofstream out(path, std::ios::binary);

    if (!out) {
        throw std::runtime_error(
            "Failed to create temporary ONNX test model");
    }

    if (!model.SerializeToOstream(&out)) {
        throw std::runtime_error(
            "Failed to serialize temporary ONNX test model");
    }

    return path;
}

}  // namespace

int main() {
    const auto model_path = make_test_model();

    // ---------------------------------------------------------------------
    // Valid model
    // ---------------------------------------------------------------------

    {
        auto model =
            flexon::offline::model::OnnxModelLoader::load(model_path);

        assert(model != nullptr);
        assert(model->has_graph());
        assert(model->graph().name() == "loader_test_graph");
        assert(model->graph().node_size() == 1);
        assert(model->graph().node(0).name() == "Identity_0");
        assert(model->graph().node(0).op_type() == "Identity");
        assert(model->graph().node(0).input(0) == "X");
        assert(model->graph().node(0).output(0) == "Y");
    }

    // ---------------------------------------------------------------------
    // Missing model
    // ---------------------------------------------------------------------

    {
        bool threw = false;

        try {
            flexon::offline::model::OnnxModelLoader::load(
                model_path.string() + ".missing");
        } catch (const std::runtime_error&) {
            threw = true;
        }

        assert(threw);
    }

    // ---------------------------------------------------------------------
    // Malformed model
    // ---------------------------------------------------------------------

    {
        const auto malformed_path =
            std::filesystem::temp_directory_path() /
            "flexon_malformed.onnx";

        {
            std::ofstream out(malformed_path, std::ios::binary);
            out << "this is not an ONNX model";
        }

        bool threw = false;

        try {
            flexon::offline::model::OnnxModelLoader::load(
                malformed_path);
        } catch (const std::runtime_error&) {
            threw = true;
        }

        assert(threw);

        std::filesystem::remove(malformed_path);
    }

    std::filesystem::remove(model_path);

    std::cout << "ONNX model loader test passed.\n";
    return 0;
}