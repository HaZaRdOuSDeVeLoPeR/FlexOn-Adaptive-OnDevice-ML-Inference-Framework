#pragma once

#include <cstdint>
#include <unordered_set>
#include <mutex>

#include <onnxruntime_cxx_api.h>
#include <flexon/core/types.hpp>

namespace flexon::online::arena {

struct BufferKey {
    core::Resource resource{core::Resource::CPU};
    std::size_t bytes{0};
    ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    std::vector<std::int64_t> shape;

    bool operator==(const BufferKey& other) const noexcept {
        return resource == other.resource &&
               bytes == other.bytes &&
               type == other.type &&
               shape == other.shape;
    }
};

struct BufferKeyHash {
    std::size_t operator()(const BufferKey& key) const noexcept {
        std::size_t hash = static_cast<std::size_t>(key.resource);
        hash ^= key.bytes + static_cast<std::size_t>(0x9e3779b9) +
                (hash << 6) + (hash >> 2);
        hash ^= static_cast<std::size_t>(key.type) +
                static_cast<std::size_t>(0x9e3779b9) +
                (hash << 6) + (hash >> 2);
        for (const auto dim : key.shape) {
            const auto value = static_cast<std::size_t>(dim);
            hash ^= value + static_cast<std::size_t>(0x9e3779b9) +
                    (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct BufferBlock {
    core::Resource resource{core::Resource::CPU};
    void* data{nullptr};
    std::size_t bytes{0};
    BufferKey key;
};


class BufferArena;

struct BufferLease {
    BufferArena* arena{nullptr};
    BufferBlock* block{nullptr};
    BufferLease(const BufferLease&) = delete;
    BufferLease& operator=(const BufferLease&) = delete;
    ~BufferLease();
};

class BufferArena {
public:
    ~BufferArena();

    std::shared_ptr<BufferLease> acquire(const BufferKey& key);

    std::size_t allocation_count() const noexcept;
    std::size_t reuse_count() const noexcept;
    std::size_t allocated_bytes() const noexcept;
    std::size_t peak_live_bytes() const noexcept;
    std::size_t live_bytes() const noexcept;

private:
    mutable std::mutex mutex_;
    friend struct BufferLease;

    std::size_t live_bytes_unlocked() const noexcept;
    void release(BufferBlock* block);
    void update_peak_live_bytes() noexcept;

    std::vector<std::unique_ptr<BufferBlock>> blocks_;
    std::unordered_map<BufferKey, std::vector<BufferBlock*>, BufferKeyHash> free_blocks_;
    std::unordered_set<BufferBlock*> in_use_;
    std::size_t allocation_count_{0};
    std::size_t reuse_count_{0};
    std::size_t allocated_bytes_{0};
    std::size_t peak_live_bytes_{0};
};

std::size_t element_count(const std::vector<std::int64_t>& shape);

BufferKey buffer_key(
    core::Resource resource,
    const std::vector<std::int64_t>& shape,
    ONNXTensorElementDataType type);

} // namespace flexon::online::arena