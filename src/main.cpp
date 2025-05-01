//
// Created by sayan on 4/25/25.
//
#include "einsum.hpp"
#include <algorithm>
#include <iostream>
#include <ranges>
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

using A = Labels<'i', 'j'>;
using B = Labels<'j', 'k'>;
using Res = Labels<'i', 'k'>;

using collapsed = collapsed_dimensions<A, B, Res>::type;

int main() {

  std::vector A{1, 1, 1, 2};
  std::vector B{0, 1, 2, 3};

  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

  holder a{mdA, mdB, "ij", "jk", "ik"};
  constexpr auto lmap = holder::left_label_dim_map;
  constexpr auto rmap = holder::right_label_dim_map;
  holder::left_labels lla{};
  holder::right_labels rla{};
  holder::collapsed_labels lra{};
  holder::collapsed_dims cdims{};
  holder::output_labels lres{};
}