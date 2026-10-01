#include <flexon/offline/config/model_config.hpp>

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

namespace flexon::offline::config {

namespace {

template <typename T>
T optional(const YAML::Node& node, const char* key, T value) {
    if (node[key]) {
        return node[key].as<T>();
    }

    return value;
}

std::vector<std::int64_t> read_shape(
    const YAML::Node& node,
    const std::string& input_name) {

    if (!node.IsSequence() || node.size() == 0) {
        throw std::invalid_argument(
            "Input shape for '" + input_name +
            "' must be a non-empty sequence");
    }

    std::vector<std::int64_t> shape;
    shape.reserve(node.size());

    for (const auto& dim : node) {
        const auto value = dim.as<std::int64_t>();

        if (value <= 0) {
            throw std::invalid_argument(
                "Input shape for '" + input_name +
                "' contains a non-positive dimension");
        }

        shape.push_back(value);
    }

    return shape;
}

bool is_dynamic(const onnx::TensorShapeProto_Dimension& dim) {
    return !dim.has_dim_value() || dim.dim_value() <= 0;
}

ModelConfig load_runtime_config(const YAML::Node& root) {
    ModelConfig config;

    const auto runtime = root["runtime"];

    if (runtime) {
        config.batch_size =
            optional(runtime, "batch_size", config.batch_size);

        config.static_shapes_only =
            optional(
                runtime,
                "static_shapes_only",
                config.static_shapes_only);

        config.dynamic_dimension_default =
            optional(
                runtime,
                "dynamic_dimension_default",
                config.dynamic_dimension_default);

        if (runtime["input_shapes"]) {
            const auto shapes = runtime["input_shapes"];

            if (!shapes.IsMap()) {
                throw std::invalid_argument(
                    "runtime.input_shapes must be a map");
            }

            for (const auto& entry : shapes) {
                const auto input_name =
                    entry.first.as<std::string>();

                config.input_shapes[input_name] =
                    read_shape(entry.second, input_name);
            }
        }
    }

    if (config.batch_size == 0) {
        throw std::invalid_argument(
            "runtime.batch_size must be greater than zero");
    }

    if (config.dynamic_dimension_default < 0) {
        throw std::invalid_argument(
            "runtime.dynamic_dimension_default must be zero or greater");
    }

    return config;
}

std::vector<ModelSpec> load_models(const YAML::Node& root) {
    std::vector<ModelSpec> models;

    const auto models_node = root["models"];

    if (!models_node) {
        throw std::invalid_argument(
            "models.yaml must contain a 'models' sequence");
    }

    if (!models_node.IsSequence()) {
        throw std::invalid_argument(
            "'models' in models.yaml must be a sequence");
    }

    models.reserve(models_node.size());

    for (const auto& entry : models_node) {
        if (!entry.IsMap()) {
            throw std::invalid_argument(
                "Each entry under 'models' must be a map");
        }

        if (!entry["name"]) {
            throw std::invalid_argument(
                "Every model entry must contain a 'name'");
        }

        if (!entry["path"]) {
            throw std::invalid_argument(
                "Every model entry must contain a 'path'");
        }

        ModelSpec spec;
        spec.name = entry["name"].as<std::string>();
        spec.path = entry["path"].as<std::string>();

        if (spec.name.empty()) {
            throw std::invalid_argument(
                "Model name must not be empty");
        }

        if (spec.path.empty()) {
            throw std::invalid_argument(
                "Model path for '" + spec.name +
                "' must not be empty");
        }

        models.push_back(std::move(spec));
    }

    if (models.empty()) {
        throw std::invalid_argument(
            "models.yaml contains no models");
    }

    return models;
}

}  // namespace

ModelsConfig load_models_config(
    const std::filesystem::path& path) {

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "Models config not found: " + path.string());
    }

    const auto root = YAML::LoadFile(path.string());

    ModelsConfig config;
    config.models = load_models(root);
    config.runtime = load_runtime_config(root);

    return config;
}

onnx::ModelProto resolve_profiling_shapes(
    const onnx::ModelProto& source,
    const ModelConfig& config) {

    if (!source.has_graph()) {
        throw std::invalid_argument(
            "Cannot resolve profiling shapes for an ONNX model "
            "without a graph");
    }

    onnx::ModelProto resolved = source;
    auto* graph = resolved.mutable_graph();

    for (int i = 0; i < graph->input_size(); ++i) {
        auto* input = graph->mutable_input(i);

        if (!input->has_type() ||
            !input->type().has_tensor_type()) {
            continue;
        }

        auto* tensor =
            input->mutable_type()->mutable_tensor_type();

        if (!tensor->has_shape()) {
            throw std::runtime_error(
                "Model input '" + input->name() +
                "' has no tensor shape; provide "
                "runtime.input_shapes for profiling");
        }

        const auto explicit_shape =
            config.input_shapes.find(input->name());

        const bool has_explicit_shape =
            explicit_shape != config.input_shapes.end();

        if (has_explicit_shape) {
            const auto& shape = explicit_shape->second;

            if (shape.size() !=
                static_cast<std::size_t>(
                    tensor->shape().dim_size())) {

                throw std::invalid_argument(
                    "Configured shape rank for input '" +
                    input->name() +
                    "' does not match the ONNX model");
            }

            for (int d = 0;
                 d < tensor->shape().dim_size();
                 ++d) {

                tensor->mutable_shape()
                    ->mutable_dim(d)
                    ->set_dim_value(shape[d]);
            }

            continue;
        }

        for (int d = 0;
             d < tensor->shape().dim_size();
             ++d) {

            auto* dimension =
                tensor->mutable_shape()->mutable_dim(d);

            if (!is_dynamic(*dimension)) {
                continue;
            }

            // The first dynamic dimension is treated as the
            // batch dimension. This is a runtime policy rather
            // than a model-specific assumption.
            if (d == 0) {
                dimension->set_dim_value(
                    static_cast<std::int64_t>(
                        config.batch_size));
                continue;
            }

            if (config.dynamic_dimension_default > 0) {
                dimension->set_dim_value(
                    config.dynamic_dimension_default);
                continue;
            }

            if (config.static_shapes_only) {
                throw std::runtime_error(
                    "Dynamic dimension " +
                    std::to_string(d) +
                    " of model input '" +
                    input->name() +
                    "' cannot be resolved. Provide "
                    "runtime.input_shapes for that input "
                    "or set runtime.dynamic_dimension_default.");
            }

            throw std::runtime_error(
                "Dynamic dimension " +
                std::to_string(d) +
                " of model input '" +
                input->name() +
                "' still has no concrete profiling value");
        }
    }

    return resolved;
}

}  // namespace flexon::offline::config
