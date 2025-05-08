//
// Created by sayan on 4/29/25.
//

#pragma once

#include "fixed_string.hpp"

#include <algorithm>
#include <array>
#include <ranges>
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

template <char... Cs> struct Labels {};

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

template <> struct MatrixLabelCombinator<std::index_sequence<0>, Labels<>> {
  using dims = std::tuple<>;
  using type = LabeledExtents<>;
};

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

consteval auto array_of(std::tuple<>) {
  return std::array<std::pair<char, std::size_t>, 0>{};
}

template <typename Head, typename... Tail>
consteval auto array_of(std::tuple<Head, Tail...>) {
  return std::array{std::make_pair(Head::label, Head::dim),
                    std::make_pair(Tail::label, Tail::dim)...};
}

template <char C, char... Cs>
struct contains : std::bool_constant<((C == Cs) || ...)> {};

template <template <char> class Pred, typename Pack> struct filter;
template <template <char> class Pred, char... Cs>
struct filter<Pred, Labels<Cs...>> {
  template <char C>
  using maybe_label = std::conditional_t<Pred<C>::value, Labels<C>, Labels<>>;

  template <char... As, char... Bs>
  static auto concat(Labels<As...> &&, Labels<Bs...> &&) {
    return Labels<As..., Bs...>{};
  }

  template <typename Last>
  static consteval auto concat_all_impl(Last&&) {
    return Last{};
  }

  template <typename First, typename... Rest>
  static consteval auto concat_all_impl(First&&, Rest&&...) {
    using rest_concat = decltype(concat_all_impl(std::declval<Rest>()...));
    return decltype(concat(std::declval<First>(),
                        std::declval<rest_concat>())){};
  }

  using type = decltype(concat_all_impl(maybe_label<Cs>{}...));
};

template <char... As, char... Bs>
auto union_of(Labels<As...> &&, Labels<Bs...> &&) {
  return Labels<As..., Bs...>{};
}

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
  using unionAB = decltype(union_of(std::declval<A>(), std::declval<B>()));
  using diff = typename difference<unionAB, Res>::type;
  using type = typename unique<diff>::type;
};

template <typename... Ts>
consteval auto make_iota_tuple_impl(std::tuple<Ts...>)
    -> std::tuple<std::make_index_sequence<Ts::dim>...>;

template <typename Tuple>
using tuple_iota_t = decltype(make_iota_tuple_impl(std::declval<Tuple>()));

template <typename T, typename Tuple> struct prepend_all;

template <typename T, typename... Tuples>
struct prepend_all<T, std::tuple<Tuples...>> {
  using type = std::tuple<std::tuple<T, Tuples>...>;
};

template <typename... Seqs> struct cartesian_product;

template <> struct cartesian_product<> {
  using type = std::tuple<>;
};

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
  template <typename... Seqs> static consteval auto apply(std::tuple<Seqs...>) {
    return typename cartesian_product<Seqs...>::type{};
  }

public:
  using type =
      decltype(apply(std::declval<tuple_iota_t<TupleOfLabeledDims>>()));
};

template <typename Tuple>
using cartesian_from_labeled_dims_t =
    typename cartesian_from_labeled_dims<Tuple>::type;

template <char Label> void find_by_label(std::tuple<>);

template <char Label, std::size_t Dim, char L, typename... Rest>
consteval auto find_by_label(std::tuple<LabeledDimension<Dim, L>, Rest...>) {
  if constexpr (Label == L) {
    return LabeledDimension<Dim, L>{};
  } else {
    return find_by_label<Label>(std::tuple<Rest...>{});
  }
}

template <char... Cs, typename LabeledTuple>
auto extract_labeled_dimensions(Labels<Cs...>, LabeledTuple) {
  return std::make_tuple(
      LabeledDimension<decltype(find_by_label<Cs>(
                           std::declval<LabeledTuple>()))::dim,
                       Cs>{}...);
}

template <typename Labels, typename LabeledTuple>
using extract_labeled_dimensions_t = decltype(extract_labeled_dimensions(
    std::declval<Labels>(), std::declval<LabeledTuple>()));

template <typename T> auto flatten_tuple_impl(T &&) -> std::tuple<T>;

template <typename... Ts>
consteval auto flatten_tuple_impl(std::tuple<Ts...> &&) {
  if constexpr (sizeof...(Ts) == 1) {
    return std::tuple{Ts{}...};
  }
  return std::tuple_cat(flatten_tuple_impl(Ts{})...);
}

template <typename T>
using flatten_tuple_t = decltype(flatten_tuple_impl(std::declval<T>()));

template <typename... Ts>
auto consteval map_flatten_tuple(std::tuple<Ts...>)
    -> std::tuple<flatten_tuple_t<Ts>...>;

template <typename T>
using map_flatten_tuple_t = decltype(map_flatten_tuple(std::declval<T>()));

constexpr std::size_t NOT_FOUND = static_cast<std::size_t>(-1);

template <char Label, typename... Ts>
consteval std::size_t find_index_by_label(std::tuple<Ts...> &&) {
  std::array labels{Ts::label...};
  for (auto &&[index, ld] : std::ranges::enumerate_view(labels)) {
    if (ld == Label) {
      return index;
    }
  }
  return NOT_FOUND;
}

template <typename Out, typename Res, typename ResIdx, typename Col,
          typename ColIdx, std::size_t... Is>
consteval auto build_result_tuple_impl(std::index_sequence<Is...>) {
  return std::tuple {
    []<std::size_t I>() {
      constexpr char label = std::tuple_element_t<I, Out>::label;
      constexpr std::size_t res_pos = find_index_by_label<label>(Res{});
      if constexpr (res_pos != NOT_FOUND) {
        return std::tuple_element_t<res_pos, ResIdx>{};
      } else {
        constexpr std::size_t col_pos = find_index_by_label<label>(Col{});
        static_assert(col_pos != NOT_FOUND, "Label not found in Res or Col");
        return std::tuple_element_t<col_pos, ColIdx>{};
      }
    }.template operator()<Is>()...
  };
}
template <typename Out, typename Res, typename ResIdx, typename Col,
          typename ColIdx>
consteval auto build_result_tuple() {
  return build_result_tuple_impl<Out, Res, ResIdx, Col, ColIdx>(
      std::make_index_sequence<std::tuple_size_v<Out>>{});
}

template <typename Tuple, std::size_t... Is>
consteval auto extract_indices(const Tuple &, std::index_sequence<Is...>) {
  return std::index_sequence<std::tuple_element_t<Is, Tuple>::value...>{};
}

template <typename Tuple, typename F>
constexpr void for_each_index(const Tuple &t, F &&f) {
  auto apply_indices = []<std::size_t... Is>(const Tuple &, F &&f,
                                             std::index_sequence<Is...>) {
    f(std::tuple_element_t<Is, Tuple>::value...);
  };
  constexpr std::size_t N = std::tuple_size_v<Tuple>;
  apply_indices(t, std::forward<F>(f), std::make_index_sequence<N>{});
}

template <typename Tuple> consteval std::size_t tuple_dim_product() {
  auto tuple_dim_product_impl =
      []<std::size_t... Is>(std::index_sequence<Is...>) {
        return (1 * ... * (std::tuple_element_t<Is, Tuple>::dim));
      };
  return tuple_dim_product_impl(
      std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <typename Tuple> consteval auto extract_dims() {
  auto extract_dims_impl = []<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::integer_sequence<std::size_t,
                                 std::tuple_element_t<Is, Tuple>::dim...>{};
  };
  return extract_dims_impl(
      std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <char Target, char... Cs>
constexpr std::size_t count = ((Target == Cs ? 1 : 0) + ...);

template <char C, char... Cs> auto append(Labels<Cs...>) -> Labels<Cs..., C>;

template <typename In> struct filter_unique;
template <char... Cs> struct filter_unique<Labels<Cs...>> {
private:
  template <typename Accum, char Current, char... Rest> struct helper {
    static constexpr std::size_t n = count<Current, Cs...>;
    using next = std::conditional_t<
        (n == 1), decltype(append<Current>(std::declval<Accum>())), Accum>;
    using type = typename helper<next, Rest...>::type;
  };

  template <typename Accum, char Current> struct helper<Accum, Current> {
    static constexpr std::size_t n = count<Current, Cs...>;
    using type = std::conditional_t<
        (n == 1), decltype(append<Current>(std::declval<Accum>())), Accum>;
  };

public:
  using type = typename helper<Labels<>, Cs...>::type;
};

template <typename L> using filter_unique_t = typename filter_unique<L>::type;
} // namespace EinsumTraits
