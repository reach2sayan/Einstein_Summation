//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_MATRICES_HPP
#define EINSTEIN_SUMMATION2_MATRICES_HPP

#include <array>
#include <experimental/mdspan>
#include <type_traits>

template <typename T, typename LExt, typename RExt> struct Matrices;

template <typename T, std::size_t... Ls, std::size_t... Rs>
struct Matrices<T, std::index_sequence<Ls...>, std::index_sequence<Rs...>> {
  std::mdspan<T, std::extents<std::size_t, Ls...>> left;
  std::mdspan<T, std::extents<std::size_t, Rs...>> right;
  std::array<std::size_t, sizeof...(Ls)> lidx{Ls...};
  std::array<std::size_t, sizeof...(Rs)> ridx{Rs...};
  Matrices(auto &&L_, auto &&R_) : left{FWD(L_)}, right{FWD(R_)} {}
};

template <typename T, std::size_t... Ls, std::size_t... Rs>
Matrices(std::mdspan<T, std::extents<std::size_t, Ls...>>,
         std::mdspan<T, std::extents<std::size_t, Rs...>>)
    -> Matrices<T, std::index_sequence<Ls...>, std::index_sequence<Rs...>>;

// Primary template: false by default
template <typename> struct is_matrices : std::false_type {};
template <typename T, std::size_t... Ls, std::size_t... Rs>
struct is_matrices<
    Matrices<T, std::index_sequence<Ls...>, std::index_sequence<Rs...>>>
    : std::true_type {};

template <typename M>
concept CMatrices = is_matrices<std::remove_cvref_t<M>>::value;

#endif // EINSTEIN_SUMMATION2_MATRICES_HPP
