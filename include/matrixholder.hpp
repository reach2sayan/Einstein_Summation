//
// Created by sayan on 4/26/25.
//
#ifndef MATRIXHOLDER_HPP
#define MATRIXHOLDER_HPP

#include "traits.hpp"

template <typename T, typename A, typename MatB, typename LabA, typename LabB>
class MatrixHolder;

template <typename T, size_t... DimsA, size_t... DimsB, char... CsA,
          char... CsB>
class MatrixHolder<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>, Labels<CsA...>,
                   Labels<CsB...>> {
private:
  using left_labels =
  matrix_with_labeled_dims_t<Matrix<T, DimsA...>, Labels<CsA...>>::dims;
  using right_labels =
      matrix_with_labeled_dims_t<Matrix<T, DimsB...>, Labels<CsB...>>::dims;
  static_assert(validity_checker<left_labels,right_labels>());

public:
  constexpr static auto left_label_dim_map = map_of<left_labels>::value;
  constexpr static auto right_label_dim_map = map_of<right_labels>::value;

private:
  std::pair<T *, T *> matrices = {nullptr, nullptr};

  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(MatrixHolder<Us...> &);
  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(const MatrixHolder<Us...> &);
  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(MatrixHolder<Us...> &&);
};

namespace std {
template <typename... Ts>
struct tuple_size<MatrixHolder<Ts...>>
    : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <std::size_t N, typename... Ts>
struct tuple_element<N, MatrixHolder<Ts...>> {
  using type = std::tuple_element_t<N, std::tuple<Ts...>>;
};
} // namespace std

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

using MatA = Matrix<int, 2, 2>;
using MatB = Matrix<int, 2, 2>;
using LabelsA = Labels<'i', 'j'>;
using LabelsB = Labels<'j', 'k'>;

using holder = MatrixHolder<int, MatA,MatB,LabelsA,LabelsB>;

constexpr auto m1 = holder::left_label_dim_map;

#endif // MATRIXHOLDER_HPP
