#ifndef HELPER_HPP
#define HELPER_HPP

#include <cassert>
#include <functional>
#include <map>
#include <numeric>
#include <ranges>
#include <string_view>
#include <tuple>
#include <vector>

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
  std::map<char, size_t> retmap;
  for (auto [index, c] : std::ranges::enumerate_view(str)) {
    retmap.insert({c, index});
  }
  return retmap;
}

constexpr std::multimap<char, size_t>
make_missing_axis(const std::vector<std::string_view> &inputs,
                  std::string_view out_str) {
  std::multimap<char, size_t> missing;
  for (auto &&[index, input] : std::ranges::enumerate_view(inputs)) {
    for (char c : input) {
      if (out_str.find(c) == std::string_view::npos) {
        missing.insert({c, index});
      }
    }
  }
  return missing;
}

std::vector<char> constexpr find_common_characters(
    const std::vector<std::string_view> &strings) {
  if (strings.empty()) {
    return {};
  }

  std::unordered_map<char, int> charCount;
  for (char c : strings[0]) {
    charCount[c]++;
  }

  for (size_t i = 1; i < strings.size(); i++) {
    std::unordered_map<char, int> currentCount;
    for (char c : strings[i]) {
      if (charCount.find(c) != charCount.end()) {
        currentCount[c]++;
      }
    }
    for (auto &[c, count] : charCount) {
      auto it = currentCount.find(c);
      if (it == currentCount.end()) {
        charCount[c] = 0;
      } else {
        charCount[c] = std::min(charCount[c], it->second);
      }
    }
  }
  std::vector<char> result;
  for (const auto &[c, count] : charCount) {
    for (int i = 0; i < count; i++) {
      result.push_back(c);
    }
  }
  return result;
}

constexpr auto make_iotas(const std::unordered_map<char, size_t> &lmap) {
  std::vector<std::vector<size_t>> iotas;
  iotas.reserve(lmap.size());
  for (auto &[key, index] : lmap) {
    std::vector<size_t> tmp(index);
    std::iota(std::begin(tmp), std::end(tmp), 0);
    iotas.emplace_back(std::move(tmp));
  }
  return iotas;
}

#ifdef HIDE

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
       auto shp = current_span.extents();
       // assert(shp.size() == label.size());
       for (auto &&[lchar, dim] : std::ranges::views::zip(label, shp)) {
         if (!label_to_dim.contains(lchar))
           label_to_dim[lchar] = dim;
         else
           assert(label_to_dim[lchar] == dim);
       }
     }()),
     ...);
  }(std::make_index_sequence<N>{});
}

constexpr auto
make_label_and_extents_map(const std::vector<std::string_view> &labels,
                           auto &&...inputs) {
  std::unordered_map<char, size_t> label_to_dim{};
  auto temp_tuple = std::make_tuple(std::forward<decltype(inputs)>(inputs)...);
  fill_label_to_dim_map(labels, temp_tuple, label_to_dim);
  return label_to_dim;
}

constexpr void
cartesian_product(const std::vector<std::vector<size_t>> &vectors,
                  std::vector<std::vector<size_t>> &result) {
  if (vectors.empty())
    return;

  std::vector<size_t> current;
  size_t depth = 0;

  std::function<void(size_t)> backtrack = [&](size_t index) {
    if (index == vectors.size()) {
      result.push_back(current);
      return;
    }
    for (int val : vectors[index]) {
      current.push_back(val);
      backtrack(index + 1);
      current.pop_back();
    }
  };
  backtrack(0);
}

#endif

#endif // HELPER_HPP