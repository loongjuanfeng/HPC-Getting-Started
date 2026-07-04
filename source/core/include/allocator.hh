#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace core {
[[nodiscard]] void* allocate_bytes_without_initialization(
    std::size_t bytes, std::size_t alignment);
void deallocate_bytes_without_initialization(
    void* pointer, std::size_t bytes, std::size_t alignment) noexcept;

template <typename Type>
class allocator_without_initialization {
public:
        using value_type = Type;
        using propagate_on_container_move_assignment = std::true_type;
        using propagate_on_container_swap = std::true_type;
        using is_always_equal = std::true_type;

        constexpr allocator_without_initialization() noexcept = default;

        template <typename Other>
        constexpr allocator_without_initialization(
            const allocator_without_initialization<Other>&) noexcept {}

        [[nodiscard]] Type* allocate(const std::size_t count) {
                if (count > max_size()) {
                        throw std::bad_array_new_length();
                }

                return static_cast<Type*>(allocate_bytes_without_initialization(
                    count * sizeof(Type), alignof(Type)));
        }

        void deallocate(Type* pointer, const std::size_t count) noexcept {
                deallocate_bytes_without_initialization(
                    pointer, count * sizeof(Type), alignof(Type));
        }

        [[nodiscard]] constexpr std::size_t max_size() const noexcept {
                return std::numeric_limits<std::size_t>::max() / sizeof(Type);
        }

        template <typename Item>
        void construct(Item* pointer)
            noexcept(std::is_nothrow_default_constructible_v<Item>) {
                ::new (static_cast<void*>(pointer)) Item;
        }

        template <typename Item, typename... Arguments>
            requires(sizeof...(Arguments) > 0)
        void construct(Item* pointer, Arguments&&... arguments)
            noexcept(std::is_nothrow_constructible_v<Item, Arguments...>) {
                ::new (static_cast<void*>(pointer))
                    Item(std::forward<Arguments>(arguments)...);
        }

        template <typename Item>
        void destroy(Item* pointer) noexcept {
                std::destroy_at(pointer);
        }

        template <typename Other>
        [[nodiscard]] constexpr bool operator==(
            const allocator_without_initialization<Other>&) const noexcept {
                return true;
        }
};
}  // namespace core
