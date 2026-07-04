#include "timer.hh"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {
bool expect(const bool condition, const char* message) {
        if (condition) {
                return true;
        }

        std::cerr << message << '\n';
        return false;
}
}  // namespace

int main() {
        auto ok = true;

        core::Timer timer;
        timer.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{2});
        timer.end();

        const auto first_elapsed = timer.elapsed();
        ok &= expect(first_elapsed > 0.0, "timer should measure elapsed seconds");
        ok &= expect(
            first_elapsed < 1.0, "timer should report seconds, not milliseconds");

        timer.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        timer.stop();

        const auto second_elapsed = timer.elapsed();
        ok &= expect(second_elapsed > 0.0, "stop should record the end time");
        ok &= expect(
            second_elapsed < first_elapsed,
            "starting again should reset the measured interval");

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
