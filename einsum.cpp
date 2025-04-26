//
// Created by sayan on 4/25/25.
//

#include "einsum.hpp"
#include <algorithm>
#include <numeric>

constexpr size_t EinsumLabels::find_axis(char c) {
  auto iter = std::ranges::find_if(label_axis_map,
                                   [c](auto &p) { return p.first == c; });
  if (iter != label_axis_map.end())
    return iter->second;
}

template <typename... Ts> auto Einsum<Ts...>::make_result_indices() {

  auto helper = []<size_t... Is>(auto &&inputs, auto &&mat,
                                 std::index_sequence<Is...>) {
    return make_label_and_extents_map(inputs, std::get<Is>(mat)...);
  };
  auto label_map = helper(labels.inputs, matrices,
                          std::make_index_sequence<sizeof...(Ts)>{});
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
