//
// Created by sayan on 4/25/25.
//
#include "../include/einsum.hpp"
#include "traits.hpp"
#include <algorithm>
#include <ranges>
#include <string_view>
#include <tuple>
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

constexpr auto m1 = holder::left_label_dim_map;
constexpr auto m2 = holder::right_label_dim_map;


int main() {

  std::vector A{1, 1, 1,2};
  std::vector B{0, 1, 2, 3};

  std::mdspan<int, std::extents<size_t,2,2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t,2,2>> mdB{B.data()};

  std::vector A2{1, 1, 1, 2};
  std::vector B2{0, 1, 2, 3};

  std::mdspan<int, std::extents<size_t,2,2>> mdA2{A2.data()};
  std::mdspan<int, std::extents<size_t,2,2>> mdB2{B2.data()};
  holder a{mdA, mdB,"ij","ji","ik"};
  int c = 0;
  //Einsum einsum("bhwi,bhwj->bij", mdA, mdB);
  //std::cout << einsum;

  //auto prod = einsum.make_result_indices();


}