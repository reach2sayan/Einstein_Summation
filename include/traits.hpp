//
// Created by sayan on 4/29/25.
//

#pragma once

#include "fixed_string.hpp"
#include <array>
#include <tuple>

namespace EinsumTraits {
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

template <std::size_t Dim, char Label> using LD = LabeledDimension<Dim, Label>;

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

template <typename A, typename B> struct union_of;
template <char... As, char... Bs>
struct union_of<Labels<As...>, Labels<Bs...>> {
  using type = Labels<As..., Bs...>;
};

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

template <typename Tuple> struct make_iota_tuple;
template <typename... Ts> struct make_iota_tuple<std::tuple<Ts...>> {
  using type = std::tuple<std::make_index_sequence<Ts::dim>...>;
};

template <typename... Ts> using tuple_iota_t = make_iota_tuple<Ts...>::type;

template <typename T, typename Tuple> struct prepend_all;

template <typename T, typename... Tuples>
struct prepend_all<T, std::tuple<Tuples...>> {
  using type = std::tuple<std::tuple<T, Tuples>...>;
};

template <typename... Seqs> struct cartesian_product;

template <std::size_t... Is>
struct cartesian_product<std::index_sequence<Is...>> {
  using type =
      std::tuple<std::tuple<std::integral_constant<std::size_t, Is>>...>;
};

template <std::size_t... Is, typename... Rest>
struct cartesian_product<std::index_sequence<Is...>, Rest...> {
  using rest_product = typename cartesian_product<Rest...>::type;

  template <std::size_t I>
  using prepend = typename prepend_all<std::integral_constant<std::size_t, I>,
                                       rest_product>::type;

  using type = decltype(std::tuple_cat(std::declval<prepend<Is>>()...));
};

template <typename TupleOfLabeledDims> struct cartesian_from_labeled_dims {
private:
  using iota_tuple = tuple_iota_t<TupleOfLabeledDims>;
  template <typename... Seqs> struct apply;
  template <typename... Seqs> struct apply<std::tuple<Seqs...>> {
    using type = typename cartesian_product<Seqs...>::type;
  };

public:
  using type = typename apply<iota_tuple>::type;
};

template <typename Tuple>
using cartesian_from_labeled_dims_t =
    typename cartesian_from_labeled_dims<Tuple>::type;

template <char Label, typename Tuple> struct find_by_label;

template <char Label> struct find_by_label<Label, std::tuple<>> {
  using type = void;
};

template <char Label, std::size_t Dim, char L, typename... Rest>
struct find_by_label<Label, std::tuple<LabeledDimension<Dim, L>, Rest...>> {
  using type = std::conditional_t<
      Label == L, LabeledDimension<Dim, L>,
      typename find_by_label<Label, std::tuple<Rest...>>::type>;
};

template <typename Labels, typename LabeledTuple>
struct extract_labeled_dimensions;

template <char... Cs, typename LabeledTuple>
struct extract_labeled_dimensions<Labels<Cs...>, LabeledTuple> {
  using type = std::tuple<
      LabeledDimension<find_by_label<Cs, LabeledTuple>::type::dim, Cs>...>;
};

template <typename Labels, typename LabeledTuple>
using extract_labeled_dimensions_t =
    typename extract_labeled_dimensions<Labels, LabeledTuple>::type;

template <typename T> struct flatten_tuple {
  using type = std::tuple<T>;
};

template <typename... Ts> struct flatten_tuple<std::tuple<Ts...>> {
  using type = decltype(std::tuple_cat(
      std::declval<typename flatten_tuple<Ts>::type>()...));
};

template <typename T> using flatten_tuple_t = typename flatten_tuple<T>::type;

template <typename Tuple> struct map_flatten_tuple;

template <typename... Ts> struct map_flatten_tuple<std::tuple<Ts...>> {
  using type = std::tuple<flatten_tuple_t<Ts>...>;
};

template <typename T>
using map_flatten_tuple_t = typename map_flatten_tuple<T>::type;

constexpr std::size_t NOT_FOUND = static_cast<std::size_t>(-1);

template <char Label, typename Tuple, std::size_t Index = 0>
struct find_index_by_label;

template <char Label, typename Head, typename... Tail, std::size_t Index>
struct find_index_by_label<Label, std::tuple<Head, Tail...>, Index> {
  static constexpr std::size_t value =
      (Head::label == Label)
          ? Index
          : find_index_by_label<Label, std::tuple<Tail...>, Index + 1>::value;
};

template <char Label, std::size_t Index>
struct find_index_by_label<Label, std::tuple<>, Index> {
  static constexpr std::size_t value = NOT_FOUND;
};

template <typename Out, typename Res, typename ResIdx, typename Col,
          typename ColIdx, std::size_t... Is>
constexpr auto build_result_tuple_impl(std::index_sequence<Is...>) {
  return std::tuple {
    []<std::size_t I>() {
      constexpr char label = std::tuple_element_t<I, Out>::label;

      constexpr std::size_t res_pos = find_index_by_label<label, Res>::value;
      if constexpr (res_pos != NOT_FOUND) {
        return std::tuple_element_t<res_pos, ResIdx>{};
      } else {
        constexpr std::size_t col_pos = find_index_by_label<label, Col>::value;
        static_assert(col_pos != NOT_FOUND, "Label not found in Res or Col");
        return std::tuple_element_t<col_pos, ColIdx>{};
      }
    }.template operator()<Is>()...
  };
}
template <typename Out, typename Res, typename ResIdx, typename Col,
          typename ColIdx>
constexpr auto build_result_tuple() {
  return build_result_tuple_impl<Out, Res, ResIdx, Col, ColIdx>(
      std::make_index_sequence<std::tuple_size_v<Out>>{});
}

template <typename Tuple, std::size_t... Is>
constexpr auto extract_indices(const Tuple &, std::index_sequence<Is...>) {
  return std::index_sequence<std::tuple_element_t<Is, Tuple>::value...>{};
}

// Helper to convert integral_constant tuple to index pack
template <typename Tuple, typename F, std::size_t... Is>
constexpr void apply_indices(const Tuple &, F &&f, std::index_sequence<Is...>) {
  f(std::tuple_element_t<Is, Tuple>::value...);
}

template <typename Tuple, typename F>
constexpr void for_each_index(const Tuple &t, F &&f) {
  constexpr std::size_t N = std::tuple_size_v<Tuple>;
  apply_indices(t, std::forward<F>(f), std::make_index_sequence<N>{});
}

template <typename Tuple> constexpr std::size_t tuple_dim_product() {
  auto tuple_dim_product_impl =
      []<std::size_t... Is>(std::index_sequence<Is...>) {
        return (1 * ... * (std::tuple_element_t<Is, Tuple>::dim));
      };
  return tuple_dim_product_impl(
      std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <typename Tuple> constexpr auto extract_dims() {
  auto extract_dims_impl = []<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::integer_sequence<std::size_t,
                                 std::tuple_element_t<Is, Tuple>::dim...>{};
  };
  return extract_dims_impl(
      std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <char Target, char... Cs>
constexpr std::size_t count = ((Target == Cs ? 1 : 0) + ...);

// Append a char to Labels
template <char C, typename L>
struct append;

template <char C, char... Cs>
struct append<C, Labels<Cs...>> {
  using type = Labels<Cs..., C>;
};

// Filter out characters that appear more than once
template <typename In>
struct filter_unique;

template <char... Cs>
struct filter_unique<Labels<Cs...>> {
private:
  template <typename Accum, char Current, char... Rest>
  struct helper {
    static constexpr std::size_t n = count<Current, Cs...>;
    using next = std::conditional_t<
      (n == 1),
      typename append<Current, Accum>::type,
      Accum>;
    using type = typename helper<next, Rest...>::type;
  };

  // Base case
  template <typename Accum, char Current>
  struct helper<Accum, Current> {
    static constexpr std::size_t n = count<Current, Cs...>;
    using type = std::conditional_t<
      (n == 1),
      typename append<Current, Accum>::type,
      Accum>;
  };

public:
  using type = typename helper<Labels<>, Cs...>::type;
};

// Alias for convenience
template <typename L>
using filter_unique_t = typename filter_unique<L>::type;
} // namespace EinsumTraits
