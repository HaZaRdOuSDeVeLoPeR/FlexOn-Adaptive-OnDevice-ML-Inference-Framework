#include <flexon/offline/graph/graph_analyzer.hpp>

#include <cassert>
#include <iostream>
#include <stdexcept>

namespace {

onnx::ModelProto make_test_model() {
    onnx::ModelProto model;
    model.set_ir_version(7);
    model.set_producer_name("FlexOnTest");

    auto* graph = model.mutable_graph();
    graph->set_name("graph_analyzer_test");

    auto* input = graph->add_input();
    input->set_name("X");

    auto* weight = graph->add_initializer();
    weight->set_name("W");

    auto* hidden = graph->add_node();
    hidden->set_name("Hidden");
    hidden->set_op_type("Add");
    hidden->add_input("X");
    hidden->add_input("W");
    hidden->add_output("H");

    // Deliberately unnamed node.
    auto* output_node = graph->add_node();
    output_node->set_op_type("Relu");
    output_node->add_input("H");
    output_node->add_output("Y");

    auto* output = graph->add_output();
    output->set_name("Y");

    return model;
}

}  // namespace

int main() {
    const auto model = make_test_model();

    const auto graph =
        flexon::offline::graph::GraphAnalyzer::analyze(model);

    // ---------------------------------------------------------------------
    // Basic graph information
    // ---------------------------------------------------------------------

    assert(graph.name == "graph_analyzer_test");

    assert(graph.graph_inputs.size() == 1);
    assert(graph.graph_inputs[0] == "X");

    assert(graph.graph_outputs.size() == 1);
    assert(graph.graph_outputs[0] == "Y");

    assert(graph.initializers.size() == 1);
    assert(graph.initializers[0] == "W");

    // ---------------------------------------------------------------------
    // Operator ordering
    // ---------------------------------------------------------------------

    assert(graph.operators.size() == 2);

    assert(graph.operators[0].graph_index == 0);
    assert(graph.operators[0].name == "Hidden");
    assert(graph.operators[0].op_type == "Add");

    assert(graph.operators[1].graph_index == 1);
    assert(graph.operators[1].name == "node_1");
    assert(graph.operators[1].op_type == "Relu");

    // ---------------------------------------------------------------------
    // Producer map
    // ---------------------------------------------------------------------

    assert(graph.producer_operator.at("H") == 0);
    assert(graph.producer_operator.at("Y") == 1);

    // ---------------------------------------------------------------------
    // Consumer map
    // ---------------------------------------------------------------------

    assert(graph.consumer_operators.at("X").size() == 1);
    assert(graph.consumer_operators.at("X")[0] == 0);

    assert(graph.consumer_operators.at("W").size() == 1);
    assert(graph.consumer_operators.at("W")[0] == 0);

    assert(graph.consumer_operators.at("H").size() == 1);
    assert(graph.consumer_operators.at("H")[0] == 1);

    // ---------------------------------------------------------------------
    // Duplicate producer rejection
    // ---------------------------------------------------------------------

    {
        onnx::ModelProto invalid_model;
        invalid_model.set_ir_version(7);

        auto* invalid_graph = invalid_model.mutable_graph();

        auto* node_a = invalid_graph->add_node();
        node_a->set_name("A");
        node_a->set_op_type("Identity");
        node_a->add_output("duplicate");

        auto* node_b = invalid_graph->add_node();
        node_b->set_name("B");
        node_b->set_op_type("Identity");
        node_b->add_output("duplicate");

        bool threw = false;

        try {
            (void)flexon::offline::graph::GraphAnalyzer::analyze(
                invalid_model);
        } catch (const std::runtime_error&) {
            threw = true;
        }

        assert(threw);
    }

    std::cout << "Graph analyzer test passed.\n";
    return 0;
}