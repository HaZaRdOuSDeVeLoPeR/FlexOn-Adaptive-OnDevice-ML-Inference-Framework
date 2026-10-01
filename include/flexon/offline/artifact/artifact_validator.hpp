#pragma once

#include <flexon/offline/config/offline_config.hpp>

#include <filesystem>

namespace flexon::offline::artifact {

class ArtifactValidator {
public:
    static void validate(const std::filesystem::path& directory);
};

}  // namespace flexon::offline::artifact
