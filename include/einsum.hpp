//
// Created by sayan on 4/26/25.
//
#pragma once

#include "traits.hpp"
#include <experimental/mdspan>
#include <ostream>

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
    return out;
  }

public:
  Einsum(std::mdspan<T, std::extents<size_t, DimsA...>> A,
         std::mdspan<T, std::extents<size_t, DimsB...>> B,
         fixed_string<sizeof...(CsA)> la, fixed_string<sizeof...(CsB)> lb,
         fixed_string<sizeof...(CsRes)> lres)
      : matrices{A, B}, lstr{la}, rstr{lb}, resstr{lres} {}

  constexpr auto eval();

private:
  const std::pair<std::mdspan<T, std::extents<size_t, DimsA...>>,
                  std::mdspan<T, std::extents<size_t, DimsB...>>>
      matrices;

  const fixed_string<sizeof...(CsA)> lstr;
  const fixed_string<sizeof...(CsB)> rstr;
  const fixed_string<sizeof...(CsRes)> resstr;
};

template <typename T, size_t... DimsA, size_t... DimsB, char... CsA,
          char... CsB, char... CsRes>
constexpr auto
Einsum<T, Matrix<T, DimsA...>, Matrix<T, DimsB...>, Labels<CsA...>,
       Labels<CsB...>, Labels<CsRes...>>::eval() {}

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
