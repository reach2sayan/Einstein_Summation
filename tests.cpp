//
// Created by sayan on 4/25/25.
//

#include <gtest/gtest.h>
#include "einsum.hpp"

TEST(StringTest, SplitArrow) {
  auto bind = split_arrow("ij,jk->ik");
  auto f = bind.first;
  auto labels = split_comma(f);
  ASSERT_EQ(labels.size(), 2);
}

TEST(StringTest, SplitArrow2) {
  auto bind = split_arrow("bhwi,bhwj->bij");
  auto f = bind.first;
  auto labels = split_comma(f);
  ASSERT_EQ(labels.size(), 2);
}

TEST(ArrayTest, Entry) {
  std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  ArrayT<int,3,3> arr{a};
  for (auto i : arr.get_shape()) {
    ASSERT_EQ(i, 3);
  }
}

TEST(MapTest, A) {
  auto bind = split_arrow("ij,jk->ik");
  auto f = bind.first;
  auto labels = split_comma(f);

  std::vector<int> a{1,4,1,7, 8,1,2,2, 7,4,3,4};
  ArrayT<int,3,4> arr{a};

  std::vector<int> b{2,5, 0,1, 5,7, 9,2};
  ArrayT<int,4,2> arr2{b};

  auto ret = make_label_and_extents_map(labels, arr, arr2);
  ASSERT_EQ(ret.at('k'), 2);
  ASSERT_EQ(ret.at('j'), 4);
  ASSERT_EQ(ret.at('i'), 3);
}

