//
// Created by sayan on 4/25/25.
//
#include "../include/einsum.hpp"
#include "helpers.hpp"
#include "traits.hpp"
#include <algorithm>
#include <iostream>
#include <ranges>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

using MatA = Matrix<int, 2, 2>;
using MatB = Matrix<int, 2, 2>;
using LabelsA = Labels<'i', 'j'>;
using LabelsB = Labels<'j', 'k'>;
using LabelsR = Labels<'i', 'k'>;
const char strp[4] = "ijk";
std::string_view str(strp);
constexpr fixed_string<2> fs("ij");
using lab = decltype(make_labels<fs>());
using holder = Einsum<int, MatA, MatB, LabelsA, LabelsB, LabelsR>;

int main() {

  std::vector A{1, 1, 1, 2};
  std::vector B{0, 1, 2, 3};

  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

  holder a{mdA, mdB, "ij", "jk", "ik"};
  constexpr auto lmap = holder::left_label_dim_map;
  constexpr auto rmap = holder::right_label_dim_map;
  //constexpr auto mm = merge_and_check_conflicts(lmap, rmap);
  holder::left_labels lla{};

  tuple_iota_t<holder::left_labels> tt{};
  cartesian_from_labeled_dims_t<holder::left_labels> cst{};
  int _ = 0;
}