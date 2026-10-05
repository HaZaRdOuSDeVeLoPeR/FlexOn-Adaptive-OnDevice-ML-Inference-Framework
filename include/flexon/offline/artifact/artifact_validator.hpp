#pragma once

#include <filesystem>
#include <flexon/offline/config/offline_config.hpp>

namespace flexon::offline::artifact {

class ArtifactValidator {
public:
    static void validate(const std::filesystem::path& directory);
};

}  // namespace flexon::offline::artifact
