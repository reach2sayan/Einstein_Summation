//
// Created by sayan on 4/29/25.
//

#ifndef TRAITS_HPP
#define TRAITS_HPP

#ifndef FIXED_STRING_HPP
#include "fixed_string.hpp"
#endif

#include <array>
#include <tuple>

template <typename T, std::size_t... Dimensions> struct Matrix {
  T *data = nullptr;
  constexpr static std::size_t rank = sizeof...(Dimensions);
  constexpr static std::array<std::size_t, sizeof...(Dimensions)> extents{
      Dimensions...};

  using value_type = T;
  using seq = std::index_sequence<Dimensions...>;
};

template <char... Cs> struct Labels {
  constexpr static std::array<char, sizeof...(Cs)> labels{Cs...};
};

template <std::size_t... Dims> struct Dimensions {
  constexpr static std::array<std::size_t, sizeof...(Dims)> dims{Dims...};
};

template <std::size_t Dim, char Label> struct LabeledDimension {
  static constexpr std::size_t dim = Dim;
  static constexpr char label = Label;
};

template <typename... LabeledDimensions> struct LabeledExtents {
  using dims = std::tuple<LabeledDimensions...>;
};

template <typename Dims, typename Labels> struct MatrixLabelCombinator;

template <std::size_t... Dims, char... Cs>
struct MatrixLabelCombinator<std::index_sequence<Dims...>, Labels<Cs...>> {
  static_assert(sizeof...(Dims) == sizeof...(Cs),
                "Mismatch in dimensions and labels");
  using type = LabeledExtents<LabeledDimension<Dims, Cs>...>;
  using dims = std::tuple<LabeledDimension<Dims, Cs>...>;
};

template <typename TMatrix, typename TLabel>
using matrix_with_labeled_dims_t =
    MatrixLabelCombinator<typename TMatrix::seq, TLabel>::type;

template <typename Tuple> struct array_of;
template <typename Head, typename... Tail>
struct array_of<std::tuple<Head, Tail...>> {
  constexpr static std::array<std::pair<char, std::size_t>, sizeof...(Tail) + 1>
      value = {std::make_pair(Head::label, Head::dim),
               std::make_pair(Tail::label, Tail::dim)...};
};

template <char C, char... Cs>
struct contains : std::bool_constant<((C == Cs) || ...)> {};

template <typename A, typename B> struct concat;
template <char... As, char... Bs> struct concat<Labels<As...>, Labels<Bs...>> {
  using type = Labels<As..., Bs...>;
};

template <template <char> class Pred, typename Pack> struct filter;
template <template <char> class Pred, char... Cs>
struct filter<Pred, Labels<Cs...>> {
  template <char C>
  using maybe_label = std::conditional_t<Pred<C>::value, Labels<C>, Labels<>>;

  template <typename... Ls> struct concat_all;
  template <typename First, typename... Rest>
  struct concat_all<First, Rest...> {
    using type =
        typename concat<First, typename concat_all<Rest...>::type>::type;
  };

  template <typename Last> struct concat_all<Last> {
    using type = Last;
  };

  using type = typename concat_all<maybe_label<Cs>...>::type;
};

// Union of two label packs (allows duplicates)
template <typename A, typename B> struct union_of;
template <char... As, char... Bs>
struct union_of<Labels<As...>, Labels<Bs...>> {
  using type = Labels<As..., Bs...>;
};

// Difference: A - B (remove from A any element in B)
template <typename A, typename B> struct difference;
template <char... As, char... Bs>
struct difference<Labels<As...>, Labels<Bs...>> {
  template <char C>
  struct not_in_B : std::bool_constant<!contains<C, Bs...>::value> {};

  using type = typename filter<not_in_B, Labels<As...>>::type;
};

template <typename In, typename Seen = Labels<>, typename Out = Labels<>>
struct unique_impl;

template <char Head, char... Tail, char... SeenChars, char... OutChars>
struct unique_impl<Labels<Head, Tail...>, Labels<SeenChars...>,
                   Labels<OutChars...>> {
  static constexpr bool already_seen = contains<Head, SeenChars...>::value;

  using next_seen = std::conditional_t<already_seen, Labels<SeenChars...>,
                                       Labels<SeenChars..., Head>>;
  using next_out = std::conditional_t<already_seen, Labels<OutChars...>,
                                      Labels<OutChars..., Head>>;

  using type = typename unique_impl<Labels<Tail...>, next_seen, next_out>::type;
};

template <char... SeenChars, char... OutChars>
struct unique_impl<Labels<>, Labels<SeenChars...>, Labels<OutChars...>> {
  using type = Labels<OutChars...>;
};

template <typename L> struct unique {
  using type = typename unique_impl<L>::type;
};

template <typename A, typename B, typename Res> struct collapsed_dimensions {
  using unionAB = typename union_of<A, B>::type;
  using diff = typename difference<unionAB, Res>::type;
  using type = typename unique<diff>::type;
};

using A = Labels<'a', 'b'>;
using B = Labels<'b', 'c'>;
using Res = Labels<'c'>;

using Result = collapsed_dimensions<A, B, Res>::type;

template <typename Tuple>
struct make_iota_tuple;

// template <std::size_t Dim, char Label> struct LabeledDimension

template <typename... Ts>
struct make_iota_tuple<std::tuple<Ts...>> {
  using type = std::tuple<std::make_index_sequence<Ts::dim>...>;
};

template <typename... Ts>
using tuple_iota_t = make_iota_tuple<Ts...>::type;



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

template <typename K, typename V, std::size_t N1, std::size_t N2>
consteval auto merge_and_check_conflicts(const std::array<std::pair<K, V>, N1>& a1,
                                         const std::array<std::pair<K, V>, N2>& a2) {
  std::array<std::pair<K, V>, N1 + N2> result{};
  std::size_t result_size = 0;

  // First, copy all entries from a1
  for (std::size_t i = 0; i < N1; ++i) {
    result[result_size++] = a1[i];
  }

  // Then process entries from a2
  for (std::size_t i = 0; i < N2; ++i) {
    const auto& [key, val] = a2[i];
    bool found = false;

    // Check for conflicts
    for (std::size_t j = 0; j < result_size; ++j) {
      if (result[j].first == key) {
        found = true;
        if (result[j].second != val) {
          // Conflict detected - handle as appropriate for your needs
          // In a constexpr context, we could use static_assert or just return an error indicator
          return std::array<std::pair<K, V>, 0>{}; // Empty array signals error
        }
        break;
      }
    }

    // If not found and no conflict, add it
    if (!found) {
      result[result_size++] = a2[i];
    }
  }

  // Return only the filled part
  std::array<std::pair<K, V>, N1 + N2> final_result{};
  for (std::size_t i = 0; i < result_size; ++i) {
    final_result[i] = result[i];
  }

  return final_result;
}


static_assert(get_dim_by_label<'i', lsA>::dim == 2);
static_assert(get_dim_by_label<'j', lsB>::dim == 2);
#endif