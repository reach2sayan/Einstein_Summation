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

int main() {

  std::vector A{1, 1, 1,2, 2, 2,3, 3, 3};
  std::vector B{0, 1, 2, 3, 4, 5, 6, 7, 8};

  std::mdspan<int, std::extents<size_t,3,3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t,3,3>> mdB{B.data()};

  std::vector A2{1, 1, 1, 2};
  std::vector B2{0, 1, 2, 3};

  std::mdspan<int, std::extents<size_t,2,2>> mdA2{A2.data()};
  std::mdspan<int, std::extents<size_t,2,2>> mdB2{B2.data()};


  //MatrixHolder holder2{mdA2, mdB2};
  //Einsum einsum2("ij,jk->ki", mdA2, mdB2);
  //EinsumLabels labels2("ij,jk->ki");

  MatrixHolder holder{mdA, mdB};
  auto [mdAA, mdBB] = holder;
  EinsumLabels labels("bhwi,bhwj->bij");
  int a = 0;

  //Einsum einsum("bhwi,bhwj->bij", mdA, mdB);
  //std::cout << einsum;

  //auto prod = einsum.make_result_indices();


}