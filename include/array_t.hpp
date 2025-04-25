//
// Created by sayan on 4/25/25.
//

#ifndef ARRAY_T_HPP
#define ARRAY_T_HPP

#include <array>
#include <experimental/mdspan>

template <typename T, size_t... Dimensions> class ArrayT {
  std::vector<T> data;
  std::array<size_t, sizeof...(Dimensions)> shape;

public:
  explicit ArrayT(std::vector<T> d) : data{d}, shape{Dimensions...} {}
  auto get_shape() const { return shape; }
  auto get_span() const {
    std::mdspan span{data, shape};
    return span;
  }
};

#endif // ARRAY_T_HPP
