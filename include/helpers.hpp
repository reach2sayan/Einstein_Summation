

#include <ranges>
#include <algorithm>
#include <vector>
#include <string_view>
#include <string>
#include <tuple>
#include <unordered_map>

/*
template<typename T, size_t rank, size_t... dimensions>
struct ArrayWrap {
    T data[(dimensions * ...)];
    std::mdspan < T, std::extents<size_t, dimensions...> span;

    ArrayWrap(const T* arr) : data{ arr }, span{ data.data() } {}
};*/

consteval std::pair<std::string_view, std::string_view>
split_arrow(std::string_view str) {
    auto dash_pos = str.find('-');
    auto arrow_pos = str.find('>');
    auto lview = str.substr(0, dash_pos);
    auto rview = str.substr(arrow_pos + 1);
    return { lview, rview };
}

constexpr auto
split_comma(std::string_view str) {
    auto old_pos = 0;
    auto pos = str.find(',');
    
    std::vector<std::string_view> inputs;
    inputs.reserve(str.size());
    
    while (pos != std::string::npos) {
        auto news = str.substr(old_pos, pos);
        inputs.emplace_back(str);
        str = str.substr(pos + 1);
        old_pos = pos;
        pos = str.find(',');
    }
    inputs.shrink_to_fit();
    return inputs;
}

auto std::unordered_map<char, size_t> make_set(const std::vector<std::string) {

}

