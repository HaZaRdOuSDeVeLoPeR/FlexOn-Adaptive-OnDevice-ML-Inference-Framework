#include <flexon/offline/graph/graph_analyzer.hpp>
#include <flexon/offline/shapes/runtime_shape_resolver.hpp>

#include <cassert>
#include <iostream>

int main() {
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("runtime_shape_test");

    auto* input = graph->add_input();
    input->set_name("X");
    input->mutable_type()->mutable_tensor_type()->set_elem_type(
        onnx::TensorProto_DataType_FLOAT);
    auto* shape = input->mutable_type()->mutable_tensor_type()->mutable_shape();
    shape->add_dim()->set_dim_value(1);
    shape->add_dim()->set_dim_value(3);
    shape->add_dim()->set_dim_value(8);
    shape->add_dim()->set_dim_value(8);

    auto* node = graph->add_node();
    node->set_op_type("Identity");
    node->add_input("X");
    node->add_output("H");

    auto* output = graph->add_output();
    output->set_name("H");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(
        onnx::TensorProto_DataType_FLOAT);

    auto info = flexon::offline::graph::GraphAnalyzer::analyze(model);

    // Force the exact condition this resolver handles: a tensor whose shape
    // metadata is unavailable after static analysis. The runtime probe must
    // recover the actual execution shape without model-specific assumptions.
    info.tensors.at("H").shape.clear();
    info.tensors.at("H").shape_known = false;

    flexon::offline::shapes::RuntimeShapeResolver::resolve(model, info);

    const auto& resolved = info.tensors.at("H");
    assert(resolved.shape_known);
    assert((resolved.shape == std::vector<std::int64_t>{1, 3, 8, 8}));

    std::cout << "Runtime shape resolver test passed.\n";
    return 0;
}
