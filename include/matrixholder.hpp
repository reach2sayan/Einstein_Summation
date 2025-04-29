//
// Created by sayan on 4/26/25.
//
#ifndef MATRIXHOLDER_HPP
#define MATRIXHOLDER_HPP


#include "traits.hpp"

template <template <typename T, size_t... DimensionsA> class MatrixA,
          template <typename T, size_t... DimensionsB> class MatrixB,
          template <char... CsA> class LabelsA,
          template <char... CsB> class LabelsB
>
class MatrixHolder2 {

};




template <typename... Ts> class MatrixHolder {
  std::tuple<Ts...> matrices;
  template <typename... Us> friend class Einsum;

  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(MatrixHolder<Us...> &);
  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(const MatrixHolder<Us...> &);
  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(MatrixHolder<Us...> &&);

public:
  MatrixHolder(Ts... matrices) : matrices{matrices...} {}
  constexpr size_t num_matrices() { return sizeof...(Ts); }
};

namespace std {
template <typename... Ts>
struct tuple_size<MatrixHolder<Ts...>>
    : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <std::size_t N, typename... Ts>
struct tuple_element<N, MatrixHolder<Ts...>> {
  using type = std::tuple_element_t<N, std::tuple<Ts...>>;
};
}

template <std::size_t N, typename... Ts>
constexpr decltype(auto) get(MatrixHolder<Ts...> &w) {
  return std::get<N>(w.matrices);
}
template <std::size_t N, typename... Ts>
constexpr decltype(auto) get(const MatrixHolder<Ts...> &w) {
  return std::get<N>(w.matrices);
}

template <std::size_t N, typename... Ts>
constexpr decltype(auto) get(MatrixHolder<Ts...> &&w) {
  return std::get<N>(std::move(w.matrices));
}


#endif //MATRIXHOLDER_HPP
