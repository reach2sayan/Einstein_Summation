#include <tuple>
#include <string_view>
#include <vector>
#include <ranges>

constexpr std::pair<std::string_view, std::string_view>
split_arrow(std::string_view str) {
  auto dash_pos = str.find('-');
  auto arrow_pos = str.find('>');
  auto lview = str.substr(0, dash_pos);
  auto rview = str.substr(arrow_pos + 1);
  return {lview, rview};
}

constexpr auto split_comma(std::string_view str) {
  auto old_pos = 0;
  auto pos = str.find(',');

  std::vector<std::string_view> input;
  input.reserve(str.size());

  while (pos != std::string_view::npos) {
    auto news = str.substr(old_pos, pos);
    input.emplace_back(news);
    str = str.substr(pos + 1);
    old_pos = pos;
    pos = str.find(',');
  }
  input.emplace_back(str);
  input.shrink_to_fit();
  return input;
}

constexpr auto make_label_axis_map(std::string_view str) {
  std::vector<std::pair<char, size_t>> retmap;
  auto i = 0;
  for (auto [index, c] : std::ranges::enumerate_view(str)) {
    retmap.emplace_back(c, index);
  }
  return retmap;
}