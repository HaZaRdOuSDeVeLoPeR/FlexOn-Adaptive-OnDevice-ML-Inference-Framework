#include <flexon/online/execution/buffer_arena.hpp>
#include <flexon/online/execution/helper.hpp>

namespace flexon::online::arena {
    
BufferLease::~BufferLease() {
    if (arena != nullptr && block != nullptr) {
        arena->release(block);
    }
};

BufferArena::~BufferArena() {
    // All RuntimeTensor leases are destroyed before the arena because the
    // arena is declared before the per-run tensor store.
    for (auto& block : blocks_) {
        if (block->data == nullptr) continue;
        if (block->resource == core::Resource::CUDA) {
            cudaFree(block->data);
        } else {
            std::free(block->data);
        }
    }
}

std::shared_ptr<BufferLease> BufferArena::acquire(const BufferKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& free_list = free_blocks_[key];
    if (!free_list.empty()) {
        auto* block = free_list.back();
        free_list.pop_back();
        in_use_.insert(block);
        ++reuse_count_;
        update_peak_live_bytes();
        return std::shared_ptr<BufferLease>(
            new BufferLease{this, block});
    }

    auto block = std::make_unique<BufferBlock>();
    block->resource = key.resource;
    block->bytes = key.bytes;
    block->key = key;

    if (key.bytes != 0) {
        if (key.resource == core::Resource::CUDA) {
            helper::check_cuda(cudaMalloc(&block->data, key.bytes), "cudaMalloc");
        } else {
            block->data = std::malloc(key.bytes);
            if (block->data == nullptr) {
                throw std::bad_alloc();
            }
        }
    }

    auto* raw = block.get();
    blocks_.push_back(std::move(block));
    in_use_.insert(raw);
    ++allocation_count_;
    allocated_bytes_ += key.bytes;
    update_peak_live_bytes();
    return std::shared_ptr<BufferLease>(new BufferLease{this, raw});
}

std::size_t BufferArena::allocation_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_count_;
}
std::size_t BufferArena::reuse_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return reuse_count_;
}
std::size_t BufferArena::allocated_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_bytes_;
}
std::size_t BufferArena::peak_live_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_live_bytes_;
}

std::size_t BufferArena::live_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_bytes_unlocked();
}

std::size_t BufferArena::live_bytes_unlocked() const noexcept {
    std::size_t total = 0;
    for (const auto& block : blocks_) {
        if (in_use_.find(block.get()) != in_use_.end()) {
            total += block->bytes;
        }
    }
    return total;
}


void BufferArena::release(BufferBlock* block) {
    if (block == nullptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_[block->key].push_back(block);
    in_use_.erase(block);
}

void BufferArena::update_peak_live_bytes() noexcept {
    peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_unlocked());
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("Negative tensor dimension");
        }
        if (dim != 0 && count > std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(dim)) {
            throw std::runtime_error("Tensor is too large for host allocation");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

BufferKey buffer_key(
    core::Resource resource,
    const std::vector<std::int64_t>& shape,
    ONNXTensorElementDataType type) {
    const auto count = element_count(shape);
    const auto checked_bytes = [count](std::size_t element_bytes) {
        if (count != 0 &&
            element_bytes > std::numeric_limits<std::size_t>::max() / count) {
            throw std::runtime_error("Tensor byte size overflows size_t");
        }
        return count * element_bytes;
    };
    std::size_t bytes = 0;
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: bytes = checked_bytes(sizeof(float)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: bytes = checked_bytes(sizeof(double)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: bytes = checked_bytes(sizeof(std::int64_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: bytes = checked_bytes(sizeof(std::int32_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: bytes = checked_bytes(sizeof(std::uint8_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: bytes = checked_bytes(sizeof(std::int8_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: bytes = checked_bytes(sizeof(std::uint16_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: bytes = checked_bytes(sizeof(std::int16_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: bytes = checked_bytes(sizeof(std::uint32_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: bytes = checked_bytes(sizeof(std::uint64_t)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: bytes = checked_bytes(sizeof(bool)); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: bytes = checked_bytes(2); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: bytes = checked_bytes(2); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: bytes = checked_bytes(8); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: bytes = checked_bytes(16); break;
        default:
            throw std::runtime_error(
                "Unsupported tensor type for buffer arena: " +
                std::to_string(static_cast<int>(type)));
    }
    return BufferKey{resource, bytes, type, shape};
}

}; // namespace flexon::online::execution