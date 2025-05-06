//
// Created by sayan on 4/25/25.
//
#include "helper.hpp"
#include "einsum.hpp"
#include <iostream>
#include <vector>

using MatA = EinsumTraits::Matrix<int, 2, 2>;
using MatB = EinsumTraits::Matrix<int, 2, 2>;
using LabelsA = EinsumTraits::Labels<'i', 'j'>;
using LabelsB = EinsumTraits::Labels<'j', 'k'>;
using LabelsR = EinsumTraits::Labels<'i', 'k'>;
const char strp[4] = "ijk";
std::string_view str(strp);

template <typename...> struct TD;

using A = EinsumTraits::Labels<'i', 'j'>;
using B = EinsumTraits::Labels<'j', 'k'>;
using Res = EinsumTraits::Labels<'i', 'k'>;

using collapsed = EinsumTraits::collapsed_dimensions<A, B, Res>::type;

void second_test() {

  std::vector A2{1, 4, 1, 7, 8, 1, 2, 2, 7, 4, 3, 4, 2, 4, 7, 3};
  std::vector B2{2, 5, 0, 1, 5, 7, 9, 2, 2, 3, 5, 1, 7, 5, 6, 3};
  std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdA2{A2.data()};
  std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdB2{B2.data()};
  // bhwi,bhwj->bij
  auto a2 = einsum("bhwi","bhwj","bij",mdA2, mdB2);
  auto a3 = auto_einsum("bhwi","bhwj",mdA2, mdB2);
  a2.eval();
  a3.eval();
}

int main() {
  second_test();
}
