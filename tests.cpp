//
// Created by sayan on 5/4/25.
//

#include "einsum.hpp"
#include "helper.hpp"
#include <list>
#include <algorithm>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

void iota(auto &r, int init) {
  std::ranges::generate(r, [init] mutable { return init++; });
}
TEST(EinsumTest, 2DMatrix1) {
  std::vector A{0,1,2,3,4,5};
  std::vector B{1,2,3,4,5,6,7,8,9,10,11,12};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};
  auto ein = einsum("ij","jk","ik",mdA, mdB);
  ein.eval();
  auto res_einsum = ein.get_result();

  std::vector res{0,0,0,0,0,0,0,0};
  std::mdspan<int, std::extents<size_t, 2, 4>> mdres{res.data()};
  for (auto i = 0; i < 2; ++i) {
    for (auto k = 0; k < 4; ++k) {
      mdres[i,k] = 0;
      for (auto j = 0; j < 3; ++j) {
        mdres[i,k] += mdA[i,j] * mdB[j,k];
      }
    }
  }
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 4; ++j) {
      auto v1 = mdres[i,j];
      auto v2 = res_einsum[i,j];
      ASSERT_EQ(v1,v2);
    }
  }
}

TEST(EinsumTest, 2DMatrix2) {
  std::vector A{1, 1, 1, 2};
  std::vector B{0, 1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

  constexpr fixed_string ls("ij");
  constexpr fixed_string rs("jk");
  constexpr fixed_string ress("ik");
  auto a = einsum(ls,rs,ress,mdA, mdB);
  a.eval();
  auto res_einsum = a.get_result();

  std::vector res{0,0,0,0,0,0,0,0};
  std::mdspan<int, std::extents<size_t, 2, 4>> mdres{res.data()};
  for (auto i = 0; i < 2; ++i) {
    for (auto k = 0; k < 2; ++k) {
      mdres[i,k] = 0;
      for (auto j = 0; j < 2; ++j) {
        mdres[i,k] += mdA[i,j] * mdB[j,k];
      }
    }
  }
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 2; ++j) {
      auto v1 = mdres[i,j];
      auto v2 = res_einsum[i,j];
      ASSERT_EQ(v1,v2);
    }
  }
}

TEST(EinsumTest, MatrixResultTypeCheck) {

  std::vector<int> L(6*6);
  iota(L, 0);

  std::vector<int> R(6*12);
  iota(R, 0);

  std::mdspan<int, std::extents<size_t, 6, 2, 3>> mdA{L.data()};
  std::mdspan<int, std::extents<size_t, 6, 3, 4>> mdB{R.data()};

  auto a = einsum("bmd","bdn","bmn",mdA, mdB);
  a.eval();
  static_assert(std::is_same_v<decltype(a.get_result()), std::mdspan<int, std::extents<size_t, 6, 2, 4>>>);
}