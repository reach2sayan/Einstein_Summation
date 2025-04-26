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

  auto labelmap = make_label_and_extents_map(labels, arr, arr2);
  ASSERT_EQ(labelmap.at('k'), 2);
  ASSERT_EQ(labelmap.at('j'), 4);
  ASSERT_EQ(labelmap.at('i'), 3);

  auto iotas = make_iotas(labelmap);
  std::vector<std::vector<size_t>> prod;
  cartesian_product(iotas,prod);
  auto keys = get_map_keys(labelmap);
  for (auto&& indices : prod) {
    std::unordered_map<char,size_t> tmp;
    assert(indices.size() == keys.size());
    for (auto [keyc, index] : std::ranges::zip_view(keys, indices)) {
      tmp[keyc] = index;
    }
    std::vector<std::vector<std::vector<size_t>>> res_indices;
    for (auto c : bind.second) {
      std::vector<std::vector<size_t>> tmpp{std::vector<size_t>{tmp.at(c)}};
      res_indices.push_back(tmpp);
    }

    std::vector<std::vector<std::vector<std::vector<size_t>>>> out;
    for (auto&& inp : labels) {
      std::vector<std::vector<std::vector<size_t>>> res_indices;
      for (auto c : inp) {
        std::vector<std::vector<size_t>> tmpp{std::vector<size_t>{tmp.at(c)}};
        res_indices.push_back(tmpp);
      }
      out.push_back(res_indices);
    }

  }

}

