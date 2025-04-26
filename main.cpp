//
// Created by sayan on 4/25/25.
//
#include "einsum.hpp"
#include <algorithm>
#include <ranges>
#include <string_view>
#include <tuple>
#include <vector>

int main() {

  std::vector A{1, 1, 1,2, 2, 2,3, 3, 3};
  std::vector B{0, 1, 2, 3, 4, 5, 6, 7, 8};

  std::mdspan<int, std::extents<size_t,3,3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t,3,3>> mdB{B.data()};
  MatrixHolder holder{mdA, mdB};
  EinsumLabels labels("bhwi,bhwj->bij");

  Einsum einsum("bhwi,bhwj->bij", mdA, mdB);
  std::cout << einsum;

}