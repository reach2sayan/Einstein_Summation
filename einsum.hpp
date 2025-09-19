//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_HPP
#define EINSTEIN_SUMMATION2_EINSUM_HPP

#include "input_handler.hpp"
#include "labels.hpp"
#include "matrices.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

template <CLabels Labels, CMatrices Matrices> class Einsum {
private:
  constexpr static auto make_label2dim_map(std::ranges::input_range auto &&a,
                                           std::ranges::input_range auto &&b) {
    std::map<char, std::size_t> retmap;
    for (auto [label, dim] : std::views::zip(a, b)) {
      if (retmap.contains(label)) {
        assert(retmap.at(label) == dim);
      } else {
        retmap[label] = dim;
      }
    }
    return retmap;
  }

  using LM = decltype(std::declval<Matrices>().left);
  using RM = decltype(std::declval<Matrices>().right);
  using LL = decltype(std::declval<Labels>().left);
  using RL = decltype(std::declval<Labels>().right);
  using OUT = decltype(std::declval<Labels>().out);

  using LZ = decltype(make_label2dim_map(std::declval<LL>(),
                                         std::declval<Matrices>().lidx));
  using RZ = decltype(make_label2dim_map(std::declval<RL>(),
                                         std::declval<Matrices>().ridx));
  LZ left_map;
  RZ right_map;
  OUT out_labels;

public:
  constexpr Einsum(std::same_as<Labels> auto &&labels,
                   std::same_as<Matrices> auto &&matrices)
      : left_map{make_label2dim_map(labels.left, matrices.lidx)},
        right_map{make_label2dim_map(labels.right, matrices.ridx)},
        out_labels{labels.out} {
    std::array<std::size_t, std::remove_cvref_t<Labels>::out_size> dims{};
    for (auto [idx, outlabel] : std::views::enumerate(out_labels)) {

      assert(left_map.contains(outlabel) || right_map.contains(outlabel));
      if (left_map.contains(outlabel) && right_map.contains(outlabel)) {
        assert(left_map[outlabel] == right_map[outlabel]);
      }
      if (left_map.contains(outlabel)) {
        dims[idx] = left_map[outlabel];
      } else if (right_map.contains(outlabel)) {
        dims[idx] = right_map[outlabel];
      }
    }
  }
};

template <CMatrices Matrices, CLabels Labels>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

#endif // EINSTEIN_SUMMATION2_EINSUM_HPP
