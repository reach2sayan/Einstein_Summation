//
// Created by sayan on 4/26/25.
//
#pragma once

#include "traits.hpp"
#include <experimental/mdspan>
#include <memory>
#include <ostream>

#ifndef NDEBUG
#include <iostream>
#endif

template <typename TupleA, typename TupleB> constexpr bool validity_checker() {
  for (auto &&lmap : array_of<TupleA>::value) {
    for (auto &&rmap : array_of<TupleB>::value) {
      if (lmap.first == rmap.first && lmap.second != rmap.second)
        return false;
    }
  }
  return true;
}
template <typename T, typename MatrixA, typename MatrixB, typename LabelA,
          typename LabelB, typename LabelR>
class Einsum;

template <typename T, size_t... DimsA, size_t... DimsB, char... CsA,
          char... CsB, char... CsRes>
class Einsum<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>, Labels<CsA...>,
             Labels<CsB...>, Labels<CsRes...>> {
#ifdef NDEBUG
private:
#else
public:
#endif
  using left_labels =
      matrix_with_labeled_dims_t<Matrix<T, DimsA...>, Labels<CsA...>>::dims;
  using right_labels =
      matrix_with_labeled_dims_t<Matrix<T, DimsB...>, Labels<CsB...>>::dims;
  using merged_labels = decltype(std::tuple_cat(std::declval<left_labels>(),
                                                std::declval<right_labels>()));
  static_assert(validity_checker<left_labels, right_labels>());

  using collapsed_dims =
      typename collapsed_dimensions<Labels<CsA...>, Labels<CsB...>,
                                    Labels<CsRes...>>::type;
  using collapsed_labels =
      extract_labeled_dimensions_t<collapsed_dims, merged_labels>;
  using output_labels =
      extract_labeled_dimensions_t<Labels<CsRes...>, merged_labels>;

  using out_index =
      map_flatten_tuple_t<cartesian_from_labeled_dims_t<output_labels>>;
  using collapsed_index =
      map_flatten_tuple_t<cartesian_from_labeled_dims_t<collapsed_labels>>;

  friend std::ostream &operator<<(std::ostream &out, const Einsum &w) {

    auto printer = [&]<typename TupleLike>(TupleLike tpl) {
      std::apply(
          [&](auto &&...args) {
            ((out << std::decay_t<decltype(args)>::label << " "
                  << std::decay_t<decltype(args)>::dim << "\n"),
             ...);
          },
          tpl);
    };
    out << "Left Matrix Labels (with dimensions):\n";
    printer(left_labels{});
    out << "Right Matrix Labels (with dimensions):\n";
    printer(right_labels{});
    out << "Result Labels (with dimensions):\n";
    printer(output_labels{});
    out << "Collapsed Labels (with dimensions):\n";
    printer(collapsed_labels{});
    out << "\nLeft Matrix :\n";
    print_2dmd_span(out, w.matrices.first);
    out << "Right Matrix :\n";
    print_2dmd_span(out, w.matrices.second);

    return out;
  }
  constexpr static std::size_t result_size = tuple_dim_product<output_labels>();
  constexpr auto print_eval();

  template <typename MatRes, typename MatL, typename MatR, typename A,
            typename B1, typename C1>
  void assign(MatRes &matres, MatL &matl, MatR &matr, const A &a, const B1 &b1,
              const C1 &c1) {
    for_each_index(a, [&](auto... ai) {
      for_each_index(b1, [&](auto... bi) {
        for_each_index(c1, [&](auto... ci) {
          auto mat_l = matl[bi...];
          auto mat_r = matr[ci...];
          auto sum = mat_l * mat_r;
          matres[ai...] += sum;
        });
      });
    });
  }

  template <std::size_t... Dims>
  constexpr static auto
  make_mdspan(auto *data, std::integer_sequence<std::size_t, Dims...>) {
    return std::mdspan<T, std::extents<std::size_t, Dims...>>{data};
  }

public:
  Einsum(std::mdspan<T, std::extents<size_t, DimsA...>> A,
         std::mdspan<T, std::extents<size_t, DimsB...>> B,
         fixed_string<sizeof...(CsA)> la, fixed_string<sizeof...(CsB)> lb,
         fixed_string<sizeof...(CsRes)> lres)
      : matrices{A, B}, lstr{la}, rstr{lb}, resstr{lres},
        result_matrix{new T[std::decay_t<decltype(*this)>::result_size]},
        result_span{make_mdspan(result_matrix, extract_dims<output_labels>())} {
  }

  constexpr auto eval();
  auto get_result() const { return result_span; }

private:
  const std::pair<std::mdspan<T, std::extents<size_t, DimsA...>>,
                  std::mdspan<T, std::extents<size_t, DimsB...>>>
      matrices;

  const fixed_string<sizeof...(CsA)> lstr;
  const fixed_string<sizeof...(CsB)> rstr;
  const fixed_string<sizeof...(CsRes)> resstr;
  T *result_matrix;
  decltype(make_mdspan(result_matrix,
                       extract_dims<output_labels>())) result_span;
};

template <typename T, size_t... DimsA, size_t... DimsB, char... CsA,
          char... CsB, char... CsRes>
constexpr auto
Einsum<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>, Labels<CsA...>,
       Labels<CsB...>, Labels<CsRes...>>::print_eval() {

  using self = typename std::decay_t<decltype(*this)>;
  auto apply_single = [=]<typename CollapsedTupleIndex, typename OutTupleIndex>(
                          CollapsedTupleIndex, OutTupleIndex) {
    using ridx = flatten_tuple_t<
        decltype(build_result_tuple<typename self::right_labels,
                                    typename self::output_labels, OutTupleIndex,
                                    typename self::collapsed_labels,
                                    CollapsedTupleIndex>())>;
    using lidx = flatten_tuple_t<
        decltype(build_result_tuple<typename self::left_labels,
                                    typename self::output_labels, OutTupleIndex,
                                    typename self::collapsed_labels,
                                    CollapsedTupleIndex>())>;

    print_named_tuple(OutTupleIndex{}, "A");
    std::cout << " = ";
    print_named_tuple(ridx{}, "B1");
    std::cout << " * ";
    print_named_tuple(lidx{}, "C1");
    std::cout << std::endl;
    int _ = 42;
  };

  // inner loop
  auto collapsing_loop = [=]<typename TupleLike>(TupleLike) {
    std::apply(
        [=](auto &&...args_inner) {
          (apply_single(args_inner, TupleLike{}), ...);
        },
        typename self::collapsed_index{});
  };

  auto outer_loop = [=](auto &&...args) { (collapsing_loop(args), ...); };
  std::apply(outer_loop, typename self::out_index{});
}
template <typename T, size_t... DimsA, size_t... DimsB, char... CsA,
          char... CsB, char... CsRes>
constexpr auto
Einsum<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>, Labels<CsA...>,
       Labels<CsB...>, Labels<CsRes...>>::eval() {

  using self = typename std::decay_t<decltype(*this)>;
  auto apply_single = [this]<typename CollapsedTupleIndex,
                             typename OutTupleIndex>(CollapsedTupleIndex,
                                                     OutTupleIndex) {
    using ridx = flatten_tuple_t<
        decltype(build_result_tuple<typename self::right_labels,
                                    typename self::output_labels, OutTupleIndex,
                                    typename self::collapsed_labels,
                                    CollapsedTupleIndex>())>;
    using lidx = flatten_tuple_t<
        decltype(build_result_tuple<typename self::left_labels,
                                    typename self::output_labels, OutTupleIndex,
                                    typename self::collapsed_labels,
                                    CollapsedTupleIndex>())>;

    assign(result_span, matrices.first, matrices.second, OutTupleIndex{},
           lidx{}, ridx{});
  };

  // inner loop
  auto collapsing_loop = [=]<typename TupleLike>(TupleLike) {
    std::apply(
        [=](auto &&...args_inner) {
          (apply_single(args_inner, TupleLike{}), ...);
        },
        typename self::collapsed_index{});
  };

  auto outer_loop = [=](auto &&...args) { (collapsing_loop(args), ...); };
  std::apply(outer_loop, typename self::out_index{});
}

consteval std::pair<std::string_view, std::string_view>
split_arrow(std::string_view str) {
  auto dash_pos = str.find('-');
  auto arrow_pos = str.find('>');
  auto lview = str.substr(0, dash_pos);
  auto rview = str.substr(arrow_pos + 1);
  return {lview, rview};
}

consteval auto split_comma(std::string_view str) {
  const auto pos = str.find(',');
  return std::array<std::string_view, 2>{str.substr(0, pos),
                                         str.substr(pos + 1)};
}

template <typename MDSpan, std::size_t... Is>
constexpr auto make_matrix_from_mdspan(std::index_sequence<Is...>) {
  using T = typename MDSpan::element_type;
  using Extents = typename MDSpan::extents_type;
  return Matrix<T, Extents::static_extent(Is)...>{};
}

template <typename MDSpan> constexpr auto make_matrix_from_mdspan() {
  constexpr std::size_t rank = MDSpan::extents_type::rank();
  return make_matrix_from_mdspan<MDSpan>(std::make_index_sequence<rank>{});
}

template <fixed_string fsl, fixed_string fsr, fixed_string fsres,
          typename MDSpanA, typename MDSpanB>
constexpr auto make_einsum(MDSpanA mdA, MDSpanB mdB) {
  using T = typename MDSpanA::element_type;
  using MatA = decltype(make_matrix_from_mdspan<MDSpanA>());
  using MatB = decltype(make_matrix_from_mdspan<MDSpanB>());
  using LabelA = label_t<fsl>;
  using LabelB = label_t<fsr>;
  using LabelR = label_t<fsres>;
  return Einsum<T, MatA, MatB, LabelA, LabelB, LabelR>{mdA, mdB, fsl, fsr,
                                                       fsres};
}

#define einsum(left,right,result, A, B) make_einsum<left,right,result>(A,B)