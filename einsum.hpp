//
// Created by sayan on 4/25/25.
//

#ifndef EINSUM_HPP
#define EINSUM_HPP
#include "helper.hpp"
#include <array>
#include <experimental/mdspan>
#include <iostream>
#include <tuple>
#include <ostream>



class EinsumLabels {
  std::string_view out_str;
  std::vector<std::string_view> inputs;
  std::vector<std::pair<char, size_t>> label_axis_map;

public:
  constexpr EinsumLabels(std::string_view str)
      : out_str{split_arrow(str).second},
        inputs{split_comma(split_arrow(str).second)},
        label_axis_map{make_label_axis_map(out_str)} {}

  constexpr size_t find_axis(char c);

  friend std::ostream &operator<<(std::ostream &out,
                                  const EinsumLabels &labels) {
    out << "Output: " << labels.out_str << "\n";
    for (auto input : labels.inputs) {
      out << "Input: " << input << "\n";
    }
    out << "Axis Map:\n";
    for (auto [c, i] : labels.label_axis_map) {
      out << c << " -> " << i << "\n";
    }
    out << std::endl;
    return out;
  }

  auto num_inputs() const { return inputs.size(); }
  std::string_view common_axis() const;
  auto get_map() const { return label_axis_map; }
};

template <typename... Ts> class MatrixHolder {
  std::tuple<Ts...> matrices;

public:
  MatrixHolder(Ts... matrices) : matrices{matrices...} {}
  constexpr size_t num_matrices() { return sizeof...(Ts); }
  decltype(auto) operator[](size_t i) { return std::get<i>(matrices); }
};

template <typename... Ts> class Einsum {
  MatrixHolder<Ts...> matrices;
  EinsumLabels labels;

public:
  Einsum(std::string_view str, Ts... mats) : matrices{mats...}, labels{str} {}
  friend std::ostream& operator<<(std::ostream& out, const Einsum& einsum) {
    out << "\n";
    out << einsum.labels;
    return out;
  }
};


constexpr size_t EinsumLabels::find_axis(char c) {
  auto iter = std::ranges::find_if(label_axis_map,
                                   [c](auto &p) { return p.first == c; });
  if (iter != label_axis_map.end())
    return iter->second;
}
#endif // EINSUM_HPP
