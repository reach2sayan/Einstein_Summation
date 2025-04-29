//
// Created by sayan on 4/29/25.
//

#ifndef TRAITS_HPP
#define TRAITS_HPP
#include <array>
#include <tuple>

template <typename T, size_t... Dimensions> struct Matrix {
  T *data = nullptr;
  constexpr static size_t rank = sizeof...(Dimensions);
  constexpr static std::array<size_t, sizeof...(Dimensions)> extents{
      Dimensions...};

  using value_type = T;
  using seq = std::index_sequence<Dimensions...>;
};

template <char... Cs> struct Labels {
  constexpr static std::array<char, sizeof...(Cs)> labels{Cs...};
};

template <size_t... Dims> struct Dimensions {
  constexpr static std::array<size_t, sizeof...(Dims)> dims{Dims...};
};

template <size_t Dim, char Label> struct LabeledDimension {
  static constexpr size_t dim = Dim;
  static constexpr char label = Label;
};

template <typename... LabeledDimensions> struct LabeledExtents {
  using dims = std::tuple<LabeledDimensions...>;
};

template <typename Dims, typename Labels> struct MatrixLabelCombinator;

template <size_t... Dims, char... Cs>
struct MatrixLabelCombinator<std::index_sequence<Dims...>, Labels<Cs...>> {
  static_assert(sizeof...(Dims) == sizeof...(Cs),
                "Mismatch in dimensions and labels");
  using type = LabeledExtents<LabeledDimension<Dims, Cs>...>;
  using dims = std::tuple<LabeledDimension<Dims, Cs>...>;
};

template <typename TMatrix, typename TLabel>
using matrix_with_labeled_dims_t =
    MatrixLabelCombinator<typename TMatrix::seq, TLabel>::type;

template <typename Tuple> struct map_of;
template <typename Head, typename... Tail>
struct map_of<std::tuple<Head, Tail...>> {
  constexpr static std::array<std::pair<char, size_t>, sizeof...(Tail) + 1>
      value = {std::make_pair(Head::label, Head::dim),
               std::make_pair(Tail::label, Tail::dim)...};
};

template <typename TupleA, typename TupleB> constexpr bool validity_checker() {
  for (auto &&lmap : map_of<TupleA>::value) {
    for (auto &&rmap : map_of<TupleB>::value) {
      if (lmap.first == rmap.first && lmap.second != rmap.second)
        return false;
    }
  }
  return true;
}

#endif // TRAITS_HPP

#ifdef HIDE
template <typename Tuple> struct labels_of;

template <typename Head, typename... Tail>
struct labels_of<std::tuple<Head, Tail...>> {
  constexpr static std::array value = {Head::label, Tail::label...};
};

template <typename Tuple> struct dims_of;

template <typename Head, typename... Tail>
struct dims_of<std::tuple<Head, Tail...>> {
  constexpr static std::array value = {Head::dim, Tail::dim...};
};

static_assert(get_dim_by_label<'i', lsA>::dim == 2);
static_assert(get_dim_by_label<'j', lsB>::dim == 2);
#endif