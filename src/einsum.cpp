//
// Created by sayan on 4/25/25.
//

#include "einsum.hpp"
#include "traits.hpp"
#include <algorithm>
#include <numeric>

template<typename... Ts>
bool validate(EinsumLabels labels, MatrixHolder<Ts...> matrices) {
  auto output_label = labels.out_str;
  for (auto label : output_label) {
    auto [beg, end] = labels.missing_from_out.equal_range(label);
    auto [matrix_index, pos_index] = beg->second;
    auto extent_first = matrices[matrix_index].extent(pos_index);
    for (auto it = beg; it != end; ++it) {
      auto [matrix_index, pos_index] = it->second;
      if (extent_first != matrices[matrix_index].extents(pos_index))
        return false;
    }
  }
  return true;
}
