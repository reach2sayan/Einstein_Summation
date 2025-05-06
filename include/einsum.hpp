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

namespace Einsum {

template <fixed_string fs> constexpr auto make_labels() {
  auto make_labels_impl = []<std::size_t... Is>(std::index_sequence<Is...>) {
    return EinsumTraits::Labels<fs[Is]...>{};
  };
  return make_labels_impl(std::make_index_sequence<fs.size()>{});
}

template <fixed_string fs> using label_t = decltype(make_labels<fs>());

template <typename TupleA, typename TupleB> constexpr bool validity_checker() {
  for (auto &&lmap : EinsumTraits::array_of<TupleA>::value) {
    for (auto &&rmap : EinsumTraits::array_of<TupleB>::value) {
      if (lmap.first == rmap.first && lmap.second != rmap.second)
        return false;
    }
  }
  return true;
}
using namespace EinsumTraits;
using EinsumTraits::Labels;
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

  template <typename MatRes, typename MatL, typename MatR, typename OutIndex,
            typename LeftIndex, typename RightIndex>
  constexpr void assign(MatRes &matres, MatL &matl, MatR &matr,
                        const OutIndex &outidx, const LeftIndex &leftidx,
                        const RightIndex &rightidx) {
    for_each_index(outidx, [&](auto... outi) {
      for_each_index(leftidx, [&](auto... li) {
        for_each_index(rightidx, [&](auto... ri) {
          auto mat_l = matl[li...];
          auto mat_r = matr[ri...];
          auto sum = mat_l * mat_r;
          matres[outi...] += sum;
        });
      });
    });
  }

  template <typename MatRes, typename Mat, typename OutIndex,
            typename RightIndex>
  constexpr void assign_noleft(MatRes &matres, Mat &matr, const OutIndex &out,
                               const RightIndex &rightidx) {
    for_each_index(out, [&](auto... outi) {
      for_each_index(rightidx, [&](auto... ri) {
        auto mat_r = matr[ri...];
        auto sum = mat_r;
        matres[outi...] += sum;
      });
    });
  }

  template <typename CollapsedTupleIndex, typename OutTupleIndex>
  auto apply_single_noleft(CollapsedTupleIndex, OutTupleIndex) {
    using ridx = flatten_tuple_t<
        decltype(build_result_tuple<right_labels, output_labels, OutTupleIndex,
                                    collapsed_labels, CollapsedTupleIndex>())>;
    assign_noleft(result_span, matrices.second, OutTupleIndex{}, ridx{});
  }

  template <typename CollapsedTupleIndex, typename OutTupleIndex>
  auto apply_single(CollapsedTupleIndex, OutTupleIndex) {
    using ridx = flatten_tuple_t<
        decltype(build_result_tuple<right_labels, output_labels, OutTupleIndex,
                                    collapsed_labels, CollapsedTupleIndex>())>;
    using lidx = flatten_tuple_t<
        decltype(build_result_tuple<left_labels, output_labels, OutTupleIndex,
                                    collapsed_labels, CollapsedTupleIndex>())>;
    assign(result_span, matrices.first, matrices.second, OutTupleIndex{},
           lidx{}, ridx{});
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
        result_matrix{new T[std::decay_t<decltype(*this)>::result_size]{}},
        result_span{make_mdspan(result_matrix, extract_dims<output_labels>())} {
  }

  ~Einsum() { delete[] result_matrix; }

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
#ifndef NDEBUG
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
#endif
}
template <typename T, size_t... DimsA, size_t... DimsB, char... CsA,
          char... CsB, char... CsRes>
constexpr auto
Einsum<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>, Labels<CsA...>,
       Labels<CsB...>, Labels<CsRes...>>::eval() {

  using self = typename std::decay_t<decltype(*this)>;
  // inner loop
  auto collapsing_loop = [&]<typename TupleLike>(TupleLike) {
    std::apply(
        [&](auto &&...args_inner) {
          if constexpr (sizeof...(args_inner) != 0 &&
                        std::tuple_size_v<left_labels> != 0) {
            (apply_single(args_inner, TupleLike{}), ...);
          } else if constexpr (sizeof...(args_inner) == 0 &&
                               std::tuple_size_v<left_labels> != 0) {
            using ridx = flatten_tuple_t<
                decltype(build_result_tuple<
                         typename self::right_labels,
                         typename self::output_labels, TupleLike,
                         typename self::collapsed_labels, std::tuple<>>())>;
            using lidx = flatten_tuple_t<
                decltype(build_result_tuple<
                         typename self::left_labels,
                         typename self::output_labels, TupleLike,
                         typename self::collapsed_labels, std::tuple<>>())>;
            assign(result_span, matrices.first, matrices.second, TupleLike{},
                   lidx{}, ridx{});
          } else if constexpr (sizeof...(args_inner) != 0 &&
                               std::tuple_size_v<left_labels> == 0) {
            (apply_single_noleft(args_inner, TupleLike{}), ...);
          } else {
            static_assert(sizeof...(args_inner) == 0 &&
                          std::tuple_size_v<left_labels> == 0);
            using ridx = flatten_tuple_t<
                decltype(build_result_tuple<
                         typename self::right_labels,
                         typename self::output_labels, TupleLike,
                         typename self::collapsed_labels, std::tuple<>>())>;
            assign_noleft(result_span, matrices.second, TupleLike{}, ridx{});
          }
        },
        typename self::collapsed_index{});
  };

  auto outer_loop = [&](auto &&...args) {
    if constexpr (sizeof...(args) != 0)
      (collapsing_loop(args), ...);
  };
  std::apply(outer_loop, typename self::out_index{});
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
template <typename Label> struct to_fixed_string;

template <char... Cs> struct to_fixed_string<Labels<Cs...>> {
  static constexpr fixed_string<sizeof...(Cs)> value =
      fixed_string<sizeof...(Cs)>{std::integer_sequence<char, Cs...>{}};
};

template <typename Label>
constexpr auto to_fixed_string_v = to_fixed_string<Label>::value;
template <fixed_string fsl, fixed_string fsr, fixed_string fsres,
          typename MDSpanA, typename MDSpanB>
constexpr auto make_einsum_impl(MDSpanA mdA, MDSpanB mdB) {
  using T = typename MDSpanA::element_type;
  using MatA = decltype(make_matrix_from_mdspan<MDSpanA>());
  using MatB = decltype(make_matrix_from_mdspan<MDSpanB>());
  using LabelA = label_t<fsl>;
  using LabelB = label_t<fsr>;
  if constexpr (fsres.size() == 0) {
    using LabelR = EinsumTraits::filter_unique_t<
        typename EinsumTraits::union_of<LabelA, LabelB>::type>;
    auto fre = to_fixed_string_v<LabelR>;
    return Einsum<T, MatA, MatB, LabelA, LabelB, LabelR>{mdA, mdB, fsl, fsr,
                                                         fre};
  } else {
    using LabelR = label_t<fsres>;
    return Einsum<T, MatA, MatB, LabelA, LabelB, LabelR>{mdA, mdB, fsl, fsr,
                                                         fsres};
  }
}

#define seinsum(right, result, B)                                              \
  [&]() {                                                                      \
    using TT = decltype(B)::element_type;                                      \
    std::vector<TT> empty{};                                                   \
    std::mdspan<TT, std::extents<size_t, 0>> mdempty{empty.data()};            \
    return Einsum::make_einsum_impl<"", right, result>(mdempty, B);            \
  }()

#define einsum(left, right, result, A, B)                                      \
  Einsum::make_einsum_impl<left, right, result>(A, B)

#define auto_einsum(left, right, A, B)                                         \
  Einsum::make_einsum_impl<left, right, "">(A, B)

} // namespace Einsum