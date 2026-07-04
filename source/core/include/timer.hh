#pragma once

#include <chrono>

namespace core {
class Timer {
public:
        using Clock = std::chrono::steady_clock;

        void start() noexcept { time_start = Clock::now(); }

        void end() noexcept { time_end = Clock::now(); }

        void stop() noexcept { end(); }

        [[nodiscard]] double elapsed() const noexcept {
                return std::chrono::duration<double>(time_end - time_start)
                    .count();
        }

private:
        Clock::time_point time_start{};
        Clock::time_point time_end{};
};
}  // namespace core
