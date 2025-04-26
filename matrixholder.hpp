//
// Created by sayan on 4/26/25.
//
#ifndef MATRIXHOLDER_HPP
#define MATRIXHOLDER_HPP

#include <tuple>

template <typename... Ts> class MatrixHolder {
  std::tuple<Ts...> matrices;

public:
  MatrixHolder(Ts... matrices) : matrices{matrices...} {}
  constexpr size_t num_matrices() { return sizeof...(Ts); }
  decltype(auto) operator[](size_t i) { return std::get<i>(matrices); }
};


#endif //MATRIXHOLDER_HPP
