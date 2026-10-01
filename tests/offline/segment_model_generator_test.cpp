#include <flexon/offline/graph/graph_analyzer.hpp>
#include <flexon/offline/model/segment_model_generator.hpp>
#include <flexon/offline/segmentation/initial_segmenter.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <utility>

int main() {
    onnx::ModelProto model;
    model.set_ir_version(7);

    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("segment_test");

    auto* input = graph->add_input();
    input->set_name("X");
    input->mutable_type()->mutable_tensor_type()->set_elem_type(
        onnx::TensorProto_DataType_FLOAT);
    input->mutable_type()->mutable_tensor_type()->mutable_shape()
        ->add_dim()->set_dim_value(1);

    auto* a = graph->add_node();
    a->set_name("A");
    a->set_op_type("Identity");
    a->add_input("X");
    a->add_output("H");

    auto* b = graph->add_node();
    b->set_name("B");
    b->set_op_type("Relu");
    b->add_input("H");
    b->add_output("Y");

    auto* output = graph->add_output();
    output->set_name("Y");
    output->mutable_type()->mutable_tensor_type()->set_elem_type(
        onnx::TensorProto_DataType_FLOAT);

    const auto info =
        flexon::offline::graph::GraphAnalyzer::analyze(model);

    flexon::offline::config::OfflineConfig config;

    flexon::core::OperatorProfileMap profiles;
    for (const auto& op : info.operators) {
        flexon::core::OperatorProfile profile;
        profile.op = op;
        profile.costs.push_back({
            flexon::core::Resource::CPU, 1.0, 1.0});
        profile.costs.push_back({
            flexon::core::Resource::CUDA, 1.0, 1.0});
        profiles.emplace(op.graph_index, std::move(profile));
    }

    const auto segments =
        flexon::offline::segmentation::InitialSegmenter::create(
            model, info, config, profiles);

    assert(segments.size() == 1);
    assert(segments[0].operator_indices.size() == 2);

    const auto segment_model =
        flexon::offline::model::SegmentModelGenerator::build(
            model, info, segments[0]);

    assert(segment_model.has_graph());
    assert(segment_model.graph().node_size() == 2);
    assert(segment_model.graph().input_size() == 1);
    assert(segment_model.graph().output_size() == 1);
    assert(segment_model.graph().input(0).name() == "X");
    assert(segment_model.graph().output(0).name() == "Y");

    // A missing CUDA cost marks an operator as a resource fallback and
    // creates a boundary around it.
    {
        auto fallback_profiles = profiles;
        fallback_profiles.at(1).costs.erase(
            std::remove_if(
                fallback_profiles.at(1).costs.begin(),
                fallback_profiles.at(1).costs.end(),
                [](const auto& cost) {
                    return cost.resource == flexon::core::Resource::CUDA;
                }),
            fallback_profiles.at(1).costs.end());

        const auto fallback_segments =
            flexon::offline::segmentation::InitialSegmenter::create(
                model, info, config, fallback_profiles);

        assert(fallback_segments.size() == 2);
        assert(fallback_segments[0].operator_indices.size() == 1);
        assert(fallback_segments[1].operator_indices.size() == 1);
        assert(fallback_segments[1].operator_indices[0] == 1);
    }

    // Explicit unsupported status is equivalent to an absent resource cost
    // for initial partitioning, while the status itself remains available to
    // the manifest generator.
    {
        auto explicit_profiles = profiles;
        explicit_profiles.at(0).costs[1].status =
            flexon::core::ResourceSupportStatus::Unsupported;
        explicit_profiles.at(0).costs[1].mean_ms =
            std::numeric_limits<double>::infinity();
        explicit_profiles.at(0).costs[1].percentile_ms =
            std::numeric_limits<double>::infinity();

        const auto explicit_segments =
            flexon::offline::segmentation::InitialSegmenter::create(
                model, info, config, explicit_profiles);

        assert(explicit_segments.size() == 2);
        assert(explicit_segments[0].operator_indices.size() == 1);
        assert(explicit_segments[0].operator_indices[0] == 0);
        assert(explicit_segments[1].operator_indices.size() == 1);
        assert(explicit_segments[1].operator_indices[0] == 1);
    }

    std::cout << "Segment model generator test passed.\n";
    return 0;
}
