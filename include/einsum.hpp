//
// Created by sayan on 4/26/25.
//
#ifndef MATRIXHOLDER_HPP
#define MATRIXHOLDER_HPP

#ifndef TRAITS_HPP
#include "traits.hpp"
#endif

#include <experimental/mdspan>
template <typename T, typename MatrixA, typename MatrixB, typename LabelA, typename LabelB, typename LabelR>
class Einsum;

template <typename T, size_t... DimsA, size_t... DimsB, char... CsA,
          char... CsB, char... CsRes>
class Einsum<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>, Labels<CsA...>,
                   Labels<CsB...>, Labels<CsRes...>> {
private:
  using left_labels =
      matrix_with_labeled_dims_t<Matrix<T, DimsA...>, Labels<CsA...>>::dims;
  using right_labels =
      matrix_with_labeled_dims_t<Matrix<T, DimsB...>, Labels<CsB...>>::dims;
  static_assert(validity_checker<left_labels, right_labels>());

public:
  constexpr static auto left_label_dim_map = array_of<left_labels>::value;
  constexpr static auto right_label_dim_map = array_of<right_labels>::value;
  Einsum(std::mdspan<T, std::extents<size_t, DimsA...>> A,
               std::mdspan<T, std::extents<size_t, DimsB...>> B,
               fixed_string<sizeof...(CsA)> la, fixed_string<sizeof...(CsA)> lb,
               fixed_string<sizeof...(CsRes)> lres)
      : matrices{A, B}, lstr{la}, rstr{lb}, resstr{lres} {}

private:
  std::pair<std::mdspan<T, std::extents<size_t, DimsA...>>,
            std::mdspan<T, std::extents<size_t, DimsA...>>>
      matrices;
  fixed_string<sizeof...(CsA)> lstr;
  fixed_string<sizeof...(CsA)> rstr;
  fixed_string<sizeof...(CsA)> resstr;
  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(Einsum<Us...> &);
  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(const Einsum<Us...> &);
  template <std::size_t N, typename... Us>
  friend constexpr decltype(auto) get(Einsum<Us...> &&);
};

namespace std {
template <typename... Ts>
struct tuple_size<Einsum<Ts...>>
    : std::integral_constant<std::size_t, 2> {};

template <std::size_t N, typename... Ts>
struct tuple_element<N, Einsum<Ts...>> {
  using type = std::tuple_element_t<N, std::tuple<Ts...>>;
};
} // namespace std

template <std::size_t N, typename... Ts>
constexpr decltype(auto) get(Einsum<Ts...> &w) {
  return std::get<N>(w.matrices);
}
template <std::size_t N, typename... Ts>
constexpr decltype(auto) get(const Einsum<Ts...> &w) {
  return std::get<N>(w.matrices);
}

template <std::size_t N, typename... Ts>
constexpr decltype(auto) get(Einsum<Ts...> &&w) {
  return std::get<N>(std::move(w.matrices));
}

template <typename T, size_t... DimsA, size_t... DimsB, char... CsA, char... CsB, char... CsRes,
fixed_string LA, fixed_string LB, fixed_string LRES>
Einsum(std::mdspan<T, std::extents<size_t, DimsA...>> A,
             std::mdspan<T, std::extents<size_t, DimsB...>> B,
             fixed_string<sizeof...(CsA)> la, fixed_string<sizeof...(CsB)> lb,
             fixed_string<sizeof...(CsRes)> lres
             )
    -> Einsum<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>,
                    decltype(make_labels<LA>()), decltype(make_labels<LB>()), decltype(make_labels<LRES>())>;

#endif // MATRIXHOLDER_HPP
