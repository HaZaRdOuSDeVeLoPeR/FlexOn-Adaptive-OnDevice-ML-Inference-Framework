#pragma once

#include <algorithm>
#include <cmath>

namespace flexon::contention {

class DutyController {
public:
    DutyController(double deadband_percent,
                   double max_step,
                   double initial_duty = 0.5)
        : deadband_(deadband_percent),
          max_step_(max_step),
          duty_(std::clamp(initial_duty, 0.0, 1.0)) {}

    double update(double target_percent, double measured_percent) {
        const double error = target_percent - measured_percent;
        if (std::abs(error) <= deadband_) {
            return duty_;
        }

        // A proportional update expressed directly in percentage points.
        // 10 percentage points of tracking error moves duty by at most 10%.
        const double delta = std::clamp(error / 100.0,
                                        -max_step_,
                                        max_step_);
        duty_ = std::clamp(duty_ + delta, 0.0, 1.0);
        return duty_;
    }

    double duty() const noexcept { return duty_; }
    void set_duty(double duty) noexcept { duty_ = std::clamp(duty, 0.0, 1.0); }

private:
    double deadband_;
    double max_step_;
    double duty_;
};

}  // namespace flexon::contention
