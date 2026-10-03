#pragma once

#include <memory>

#include <flexon/core/types.hpp>
#include <flexon/online/artifact/manifest_loader.hpp>
#include <flexon/online/execution/buffer_arena.hpp>

namespace flexon::online::runtime {

struct PreparedSession {
    core::Resource resource{core::Resource::CPU};
    std::unique_ptr<Ort::Session> session;
};

struct SegmentRuntime {
    manifest::SegmentManifest manifest;
    std::vector<PreparedSession> sessions;
};

struct RuntimeState {
    manifest::ArtifactManifest artifact;

    Ort::Env env{
        ORT_LOGGING_LEVEL_WARNING,
        "FlexOnOnline"
    };

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::vector<SegmentRuntime>> levels;
};

struct RuntimeTensor {
    // Declared before value so destruction is value first, lease second.
    // This guarantees that an externally-backed Ort::Value is gone before
    // its arena allocation becomes reusable.
    std::shared_ptr<arena::BufferLease> lease;
    Ort::Value value{nullptr};
};

core::Resource choose_concrete_resource(
    const SegmentRuntime& segment,
    core::Resource requested);

core::Resource choose_resource(const SegmentRuntime& segment,
                                core::Resource requested);

core::Resource tensor_resource(const Ort::Value& value);
                                
} // namespace flexon::online::runtime