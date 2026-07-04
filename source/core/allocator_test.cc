#include "allocator.hh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace {
struct Counted {
        inline static int default_constructed = 0;
        inline static int destroyed = 0;

        Counted() { ++default_constructed; }
        ~Counted() { ++destroyed; }
};

struct alignas(65536) OverAligned {
        int value = 0;
};

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

        using int_vector =
            std::vector<int, core::allocator_without_initialization<int>>;
        const int_vector filled(4, 7);
        ok &= expect(filled.size() == 4, "allocator should support vector size");
        ok &= expect(filled.front() == 7 && filled.back() == 7,
                     "allocator should preserve explicit construction arguments");

        {
                using counted_allocator =
                    core::allocator_without_initialization<Counted>;
                using counted_vector = std::vector<Counted, counted_allocator>;

                Counted::default_constructed = 0;
                Counted::destroyed = 0;

                counted_vector items;
                items.resize(3);

                ok &= expect(Counted::default_constructed == 3,
                             "resize should default construct non-trivial items");
                ok &= expect(items.get_allocator() == counted_allocator{},
                             "allocator instances should compare equal");
        }

        ok &= expect(Counted::destroyed == 3,
                     "allocator should destroy constructed non-trivial items");

        using rebound = typename std::allocator_traits<
            core::allocator_without_initialization<int>>::template rebind_alloc<double>;
        ok &= expect(
            std::is_same_v<rebound, core::allocator_without_initialization<double>>,
            "allocator_traits should be able to rebind the allocator");

        core::allocator_without_initialization<OverAligned> over_aligned_allocator;
        auto* over_aligned = over_aligned_allocator.allocate(1);
        ok &= expect(reinterpret_cast<std::uintptr_t>(over_aligned) %
                         alignof(OverAligned) ==
                         0,
                     "allocator should honor over-aligned types");
        over_aligned_allocator.deallocate(over_aligned, 1);

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
