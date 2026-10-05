#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <flexon/offline/config/model_config.hpp>

namespace {

onnx::ModelProto make_model() {
    onnx::ModelProto model;
    model.set_ir_version(7);

    auto* graph = model.mutable_graph();
    graph->set_name("shape_test");

    auto* input = graph->add_input();
    input->set_name("input");

    input->mutable_type()
        ->mutable_tensor_type()
        ->set_elem_type(
            onnx::TensorProto_DataType_FLOAT);

    auto* shape =
        input->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape();

    shape->add_dim()->set_dim_param("batch");
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(224);
    shape->add_dim()->set_dim_value(224);

    auto* node = graph->add_node();
    node->set_op_type("Identity");
    node->add_input("input");
    node->add_output("output");

    auto* output = graph->add_output();
    output->set_name("output");

    output->mutable_type()
        ->mutable_tensor_type()
        ->set_elem_type(
            onnx::TensorProto_DataType_FLOAT);

    return model;
}

}  // namespace

int main() {
    const auto model = make_model();

    // ---------------------------------------------------------------------
    // Batch-size resolution
    // ---------------------------------------------------------------------

    flexon::offline::config::ModelConfig config;
    config.batch_size = 4;
    config.static_shapes_only = true;

    const auto resolved =
        flexon::offline::config::resolve_profiling_shapes(
            model,
            config);

    const auto& dims =
        resolved.graph()
            .input(0)
            .type()
            .tensor_type()
            .shape()
            .dim();

    assert(dims.size() == 4);
    assert(dims[0].dim_value() == 4);
    assert(dims[1].dim_value() == 3);
    assert(dims[2].dim_value() == 224);
    assert(dims[3].dim_value() == 224);

    // ---------------------------------------------------------------------
    // Explicit shape resolution
    // ---------------------------------------------------------------------

    flexon::offline::config::ModelConfig explicit_config;
    explicit_config.batch_size = 1;
    explicit_config.input_shapes["input"] =
        {2, 5, 16, 16};

    const auto explicitly_resolved =
        flexon::offline::config::resolve_profiling_shapes(
            model,
            explicit_config);

    const auto& explicit_dims =
        explicitly_resolved.graph()
            .input(0)
            .type()
            .tensor_type()
            .shape()
            .dim();

    assert(explicit_dims[0].dim_value() == 2);
    assert(explicit_dims[1].dim_value() == 5);
    assert(explicit_dims[2].dim_value() == 16);
    assert(explicit_dims[3].dim_value() == 16);

    // ---------------------------------------------------------------------
    // models.yaml catalog loading
    // ---------------------------------------------------------------------

    const auto temp_path =
        std::filesystem::temp_directory_path() /
        "flexon_models_config_test.yaml";

    {
        std::ofstream file(temp_path);

        file
            << "models:\n"
            << "  - name: resnet18\n"
            << "    path: models/resnet18-v1-7.onnx\n"
            << "  - name: efficientnet_lite4\n"
            << "    path: models/efficientnet-lite4-11.onnx\n"
            << "runtime:\n"
            << "  batch_size: 2\n"
            << "  static_shapes_only: true\n"
            << "  dynamic_dimension_default: 0\n"
            << "  input_shapes: {}\n";
    }

    const auto models_config =
        flexon::offline::config::load_models_config(
            temp_path);

    assert(models_config.models.size() == 2);

    assert(models_config.models[0].name == "resnet18");
    assert(
        models_config.models[0].path ==
        std::filesystem::path(
            "models/resnet18-v1-7.onnx"));

    assert(
        models_config.models[1].name ==
        "efficientnet_lite4");

    assert(
        models_config.models[1].path ==
        std::filesystem::path(
            "models/efficientnet-lite4-11.onnx"));

    assert(models_config.runtime.batch_size == 2);
    assert(
        models_config.runtime.static_shapes_only);

    std::filesystem::remove(temp_path);

    std::cout
        << "Model config test passed.\n";

    return 0;
}
