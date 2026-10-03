#include <cassert>

#include <flexon/core/types.hpp>

int main() {
    flexon::core::SegmentInfo segment;
    segment.level = 0;
    segment.id = 0;

    assert(segment.level == 0);
    assert(segment.id == 0);

    return 0;
}
