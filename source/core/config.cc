#include "config.hh"

namespace core {

Table read(const std::filesystem::path& path) {
        return toml::parse_file(path.string());
}

namespace {
void merge_into(Table& result, const Table& custom) {
        for (const auto& [key, custom_value] : custom) {
                if (const auto* custom_table = custom_value.as_table()) {
                        if (auto* base_table = result[key].as_table()) {
                                merge_into(*base_table, *custom_table);
                        } else {
                                result.insert_or_assign(key, *custom_table);
                        }
                } else {
                        result.insert_or_assign(key, custom_value);
                }
        }
}
}  // namespace

Table merge(const Table& base, const Table& custom) {
        auto result = base;
        merge_into(result, custom);
        return result;
}

}  // namespace core
