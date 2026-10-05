#pragma once

#include <string>

namespace flexon::online::priority {

enum class Level {
    Normal,
    Maximum,
};

// Applies the requested scheduling policy to the calling thread. Maximum is
// SCHED_FIFO at the highest Linux real-time priority and requires CAP_SYS_NICE.
void set_current_thread(Level level);

// Promote the calling scheduler thread through the narrowly-scoped privileged
// helper. The helper validates that the target TID belongs to this process.
void promote_current_thread();

long current_thread_id();
bool is_maximum_priority();
std::string describe_current_thread();

class ScopedLevel {
public:
    explicit ScopedLevel(Level level);
    ~ScopedLevel();

    ScopedLevel(const ScopedLevel&) = delete;
    ScopedLevel& operator=(const ScopedLevel&) = delete;

private:
    int previous_policy_;
    int previous_priority_;
    bool active_{false};
};

} // namespace flexon::online::priority
