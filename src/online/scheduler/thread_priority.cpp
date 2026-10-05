#include <flexon/online/scheduler/thread_priority.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace flexon::online::priority {
namespace {

constexpr const char* kDefaultHelper =
    "/usr/local/libexec/flexon_scheduler_helper";

void apply(int policy, int priority) {
    sched_param param{};
    param.sched_priority = priority;
    const int rc = pthread_setschedparam(pthread_self(), policy, &param);
    if (rc != 0) {
        throw std::runtime_error(
            "pthread_setschedparam failed: " + std::string(std::strerror(rc)));
    }
}

void read_current(int& policy, int& priority) {
    sched_param param{};
    const int rc = pthread_getschedparam(pthread_self(), &policy, &param);
    if (rc != 0) {
        throw std::runtime_error(
            "pthread_getschedparam failed: " + std::string(std::strerror(rc)));
    }
    priority = param.sched_priority;
}

std::string helper_path() {
    if (const char* configured = std::getenv("FLEXON_PRIORITY_HELPER")) {
        if (*configured != '\0') return configured;
    }
    return kDefaultHelper;
}

} // namespace

long current_thread_id() {
    return static_cast<long>(::syscall(SYS_gettid));
}

void set_current_thread(Level level) {
    if (level == Level::Maximum) {
        apply(SCHED_FIFO, sched_get_priority_max(SCHED_FIFO));
    } else {
        apply(SCHED_OTHER, 0);
    }
}

void promote_current_thread() {
    const auto helper = helper_path();
    const auto tid = current_thread_id();
    if (tid <= 0) throw std::runtime_error("failed to obtain scheduler thread TID");

    const pid_t child = ::fork();
    if (child < 0) {
        throw std::runtime_error(
            "fork failed while starting priority helper: " +
            std::string(std::strerror(errno)));
    }

    if (child == 0) {
        const std::string tid_arg = std::to_string(tid);
        ::execl(helper.c_str(), helper.c_str(), tid_arg.c_str(), static_cast<char*>(nullptr));
        _exit(errno == ENOENT ? 127 : 126);
    }

    int status = 0;
    while (::waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(
            "waitpid failed for priority helper: " +
            std::string(std::strerror(errno)));
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
            throw std::runtime_error(
                "priority helper not found: " + helper +
                "; run scripts/setup_priority_helper.sh");
        }
        throw std::runtime_error(
            "priority helper failed for scheduler thread TID " +
            std::to_string(tid) +
            "; ensure the helper has CAP_SYS_NICE and is installed securely");
    }

    if (!is_maximum_priority()) {
        throw std::runtime_error(
            "priority helper reported success but scheduler thread is not "
            "SCHED_FIFO at maximum priority");
    }
}

bool is_maximum_priority() {
    int policy = 0;
    int priority = 0;
    read_current(policy, priority);
    return policy == SCHED_FIFO &&
           priority == sched_get_priority_max(SCHED_FIFO);
}

std::string describe_current_thread() {
    int policy = 0;
    int priority = 0;
    read_current(policy, priority);

    const char* name = "unknown";
    if (policy == SCHED_OTHER) name = "SCHED_OTHER";
    else if (policy == SCHED_FIFO) name = "SCHED_FIFO";
    else if (policy == SCHED_RR) name = "SCHED_RR";

    std::ostringstream out;
    out << "policy=" << name << " priority=" << priority;
    return out.str();
}

ScopedLevel::ScopedLevel(Level level) {
    read_current(previous_policy_, previous_priority_);
    set_current_thread(level);
    active_ = true;
}

ScopedLevel::~ScopedLevel() {
    if (!active_) return;
    try {
        apply(previous_policy_, previous_priority_);
    } catch (...) {
        // Destructors must not throw.
    }
}

} // namespace flexon::online::priority
