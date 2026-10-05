#include <fstream>

#include <flexon/offline/model/onnx_model_loader.hpp>

namespace flexon::offline::model {

std::unique_ptr<onnx::ModelProto> OnnxModelLoader::load(
    const std::filesystem::path& model_path) {

    if (model_path.empty()) {
        throw std::invalid_argument(
            "ONNX model path must not be empty");
    }

    std::ifstream input(model_path, std::ios::binary);

    if (!input) {
        throw std::runtime_error(
            "Failed to open ONNX model: " + model_path.string());
    }

    auto model = std::make_unique<onnx::ModelProto>();

    if (!model->ParseFromIstream(&input)) {
        throw std::runtime_error(
            "Failed to parse ONNX ModelProto: " + model_path.string());
    }

    if (!model->has_graph()) {
        throw std::runtime_error(
            "ONNX model does not contain a graph: " + model_path.string());
    }

    return model;
}

}  // namespace flexon::offline::model