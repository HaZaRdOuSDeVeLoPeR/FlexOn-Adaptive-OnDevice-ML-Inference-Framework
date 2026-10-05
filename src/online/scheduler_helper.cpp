#include <cerrno>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <sched.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>

namespace {

long parse_positive(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return value;
}

long read_tgid(pid_t tid) {
    std::ifstream status("/proc/" + std::to_string(tid) + "/status");
    if (!status) throw std::runtime_error("target TID does not exist");

    std::string line;
    while (std::getline(status, line)) {
        constexpr const char* prefix = "Tgid:";
        if (line.rfind(prefix, 0) != 0) continue;
        const auto first = line.find_first_not_of(" \t", 5);
        if (first == std::string::npos) break;
        return std::stol(line.substr(first));
    }
    throw std::runtime_error("could not read target thread group ID");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: " << argv[0] << " <target-tid>\n";
            return 2;
        }

        const pid_t target_tid = static_cast<pid_t>(
            parse_positive(argv[1], "target TID"));
        const pid_t owner_pid = ::getppid();

        // This helper has CAP_SYS_NICE. Never allow that capability to become
        // a generic process scheduler control interface: the target must be a
        // thread in the process that directly invoked this helper.
        if (read_tgid(target_tid) != owner_pid) {
            throw std::runtime_error(
                "target TID does not belong to the invoking FlexOn process");
        }

        sched_param param{};
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
        if (::sched_setscheduler(target_tid, SCHED_FIFO, &param) != 0) {
            throw std::runtime_error(
                "sched_setscheduler failed: " +
                std::string(std::strerror(errno)));
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[priority-helper] ERROR: " << ex.what() << '\n';
        return 1;
    }
}
