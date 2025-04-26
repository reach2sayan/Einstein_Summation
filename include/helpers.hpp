#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
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

auto get_map_keys(const std::unordered_map<char, size_t> &lmap) {
  std::vector<char> keys;
  keys.reserve(lmap.size());
  for (auto &[key, index] : lmap) {
    keys.emplace_back(key);
  }
  return keys;
}

void cartesian_product(const std::vector<std::vector<size_t>> &vectors,
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

auto index_repeater(const std::vector<char> unique_labels,
                    const std::vector<std::vector<size_t>> &products) {
  assert(unique_labels.size() == products.front().size());
  std::vector<std::unordered_map<char, size_t>> annotated;
  for (const std::vector<size_t> &product : products) {
    assert(unique_labels.size() == product.size());
    std::unordered_map<char, size_t> tmp_map{};
    for (auto [label, product] :
         std::ranges::zip_view(unique_labels, product)) {
      tmp_map.insert({label, product});
    }
    annotated.push_back(std::move(tmp_map));
  }
  return annotated;
}

auto get_result_indices(
    const std::vector<std::unordered_map<char, size_t>> &repeater,
    std::string_view res_expr) {
  std::vector<std::vector<std::vector<size_t>>> result_indices;
  for (const auto& repeat : repeater) {
    std::vector<std::vector<size_t>> tmp;
    for (auto c : res_expr) {
      tmp.push_back(std::vector{repeat.at(c)});
    }
    result_indices.push_back(std::vector{std::vector{tmp}});
  }
  return result_indices;
}

/*
for indices in domain:
    vals = {k: v for v, k in zip(indices, to_key)}
print(vals)
res_ind = tuple(zip([vals[key] for key in res_expr]))
inputs_ind = [tuple(zip([vals[key] for key in expr])) for expr in inputs_expr]
res[res_ind] += reduce_mult([M[i] for M, i in zip(inputs, inputs_ind)])
*/

void func() {}
