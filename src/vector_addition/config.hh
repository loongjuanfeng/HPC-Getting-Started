#include <cstddef>

namespace config {
constexpr bool use_no_initialization_allocator = true;
constexpr std::size_t default_vector_size = 1'900'000'000;
constexpr int default_block_size = 256;
} // namespace config
