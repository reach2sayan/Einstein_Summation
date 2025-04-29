//
// Created by sayan on 4/26/25.
//
#ifndef MATRIXHOLDER_HPP
#define MATRIXHOLDER_HPP

#include <tuple>

template <typename... Ts> class MatrixHolder {
  std::tuple<Ts...> matrices;
  template <typename... Us> friend class Einsum;
public:
  MatrixHolder(Ts... matrices) : matrices{matrices...} {}
  constexpr size_t num_matrices() { return sizeof...(Ts); }
};

// TODO : add get interface


#endif //MATRIXHOLDER_HPP
