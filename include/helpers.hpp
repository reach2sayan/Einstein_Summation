#include <algorithm>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cassert>

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

  std::vector<std::string_view> inputs;
  inputs.reserve(str.size());

  while (pos != std::string::npos) {
    auto news = str.substr(old_pos, pos);
    inputs.emplace_back(news);
    str = str.substr(pos + 1);
    old_pos = pos;
    pos = str.find(',');
  }
  inputs.emplace_back(str);
  inputs.shrink_to_fit();
  return inputs;
}

template <typename Tuple>
void fill_label_to_dim_map(const std::vector<std::string_view> &labels,
                           const Tuple &inputs_tuple,
                           std::unordered_map<char, size_t> &label_to_dim) {
  constexpr size_t N = std::tuple_size_v<Tuple>;
  assert(labels.size() == N);

  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (([&] {
      std::string_view label = labels[Is];
      const auto &current_span = std::get<Is>(inputs_tuple);
      auto shp = current_span.get_shape();
      assert(shp.size() == label.size());
      for (auto &&[lchar, dim] : std::ranges::views::zip(label, shp)) {
        if (!label_to_dim.contains(lchar))
          label_to_dim[lchar] = dim;
        else
          assert(label_to_dim[lchar] == dim);
      }
    }()), ...);
  }(std::make_index_sequence<N>{});
}

template <typename... Spans>
constexpr auto make_label_and_extents_map(const std::vector<std::string_view> &labels,
                                          Spans&&... inputs) {
  std::unordered_map<char, size_t> label_to_dim{};
  auto temp_tuple = std::make_tuple(std::forward<Spans>(inputs)...);
  fill_label_to_dim_map(labels, temp_tuple, label_to_dim);
  return label_to_dim;
}

constexpr auto make_iotas(const std::unordered_map<char, size_t>& lmap) {
  std::vector<std::ranges::iota_view<size_t, size_t>> iotas;
  iotas.reserve(lmap.size());
  for (auto [key, index] : lmap) {
    iotas.push_back(std::ranges::iota_view<size_t, size_t>(0, index));
  }
  return iotas;
}

