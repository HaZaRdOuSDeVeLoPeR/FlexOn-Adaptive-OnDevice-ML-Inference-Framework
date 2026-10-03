#include <cassert>
#include <cstdint>
#include <stdexcept>

#include <flexon/online/execution/buffer_arena.hpp>

int main() {
    using namespace flexon;
    using online::arena::BufferArena;
    using online::arena::buffer_key;
    using online::arena::element_count;

    assert(element_count({}) == 1);
    assert(element_count({1, 3, 4}) == 12);
    assert(element_count({0, 100}) == 0);

    bool threw = false;
    try {
        (void)element_count({1, -1, 4});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    const auto key = buffer_key(
        core::Resource::CPU,
        {2, 4},
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    assert(key.bytes == 2U * 4U * sizeof(float));
    assert(key.resource == core::Resource::CPU);
    assert(key.shape == std::vector<std::int64_t>({2, 4}));

    BufferArena arena;
    assert(arena.allocation_count() == 0);
    assert(arena.reuse_count() == 0);
    assert(arena.live_bytes() == 0);

    {
        auto first = arena.acquire(key);
        assert(first);
        assert(first->block != nullptr);
        assert(arena.allocation_count() == 1);
        assert(arena.reuse_count() == 0);
        assert(arena.allocated_bytes() == key.bytes);
        assert(arena.live_bytes() == key.bytes);
        assert(arena.peak_live_bytes() == key.bytes);
    }

    assert(arena.live_bytes() == 0);

    {
        auto second = arena.acquire(key);
        assert(second);
        assert(arena.allocation_count() == 1);
        assert(arena.reuse_count() == 1);
        assert(arena.live_bytes() == key.bytes);
    }

    assert(arena.live_bytes() == 0);

    const auto different_shape = buffer_key(
        core::Resource::CPU,
        {4, 2},
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    {
        auto a = arena.acquire(key);
        auto b = arena.acquire(different_shape);
        assert(arena.allocation_count() == 2);
        assert(arena.live_bytes() == key.bytes + different_shape.bytes);
        assert(arena.peak_live_bytes() == key.bytes + different_shape.bytes);
    }

    assert(arena.live_bytes() == 0);
    assert(arena.peak_live_bytes() == key.bytes + different_shape.bytes);

    return 0;
}
