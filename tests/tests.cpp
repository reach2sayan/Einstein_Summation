//
// Created by sayan on 9/20/25.
//
#include "einsum.hpp"
#include "labels.hpp"
#include "matrices.hpp"
#include <algorithm>
#include <list>
#include <ranges>
#include <vector>

#define BOOST_TEST_MODULE EinsumTestSuite
#include <boost/test/included/unit_test.hpp>

std::vector vec{0,1,2,3};
std::vector mat1{11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44};
std::vector mat2{1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4};
std::mdspan<int, std::extents<size_t, 4, 4>> mdmat1{mat1.data()};
std::mdspan<int, std::extents<size_t, 4, 4>> mdmat2{mat2.data()};

BOOST_AUTO_TEST_CASE(EinsumTest_2DMatrix1) {
  std::vector A{0, 1, 2, 3, 4, 5};
  std::vector B{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::mdspan<int, std::extents<size_t, 2, 3>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdB{B.data()};
  make_einsum(ein, "ij,jk->ik", mdA, mdB);
  ein.eval();
  auto res_einsum = ein.get_result();

  std::vector res{0, 0, 0, 0, 0, 0, 0, 0};
  std::mdspan<int, std::extents<size_t, 2, 4>> mdres{res.data()};
  for (auto i = 0; i < 2; ++i) {
    for (auto k = 0; k < 4; ++k) {
      mdres[i, k] = 0;
      for (auto j = 0; j < 3; ++j) {
        mdres[i, k] += mdA[i, j] * mdB[j, k];
      }
    }
  }
  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 4; ++j) {
      auto v1 = mdres[i, j];
      auto v2 = res_einsum[i, j];
      BOOST_CHECK_EQUAL(v1, v2);
    }
  }
}

