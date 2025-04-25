#pragma once

#include "helpers.hpp"

#if !defined(ARRAY_T_HPP)
#include "array_t.hpp"
#endif

template <typename T> class Einsum {
public:
  explicit Einsum(std::string_view str, auto... arrays);

private:
  std::string_view einsum_str;
  std::vector<std::string_view> input_labels;
  const size_t num_inputs;
  std::unordered_map<char, size_t> label_to_dim;
};

template <typename T>
Einsum<T>::Einsum(std::string_view str, auto... arrays)
    : einsum_str{str}, input_labels{split_comma(split_arrow(str).first)},
      num_inputs{sizeof...(arrays)},
      label_to_dim{make_label_and_extents_map(input_labels, arrays...)} {}
