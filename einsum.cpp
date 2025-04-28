//
// Created by sayan on 4/25/25.
//

#include "einsum.hpp"
#include <algorithm>
#include <numeric>

constexpr size_t EinsumLabels::find_axis(char c) {
  auto iter = std::ranges::find_if(output_axis_map,
                                   [c](auto &p) { return p.first == c; });
  if (iter != output_axis_map.end())
    return iter->second;
}
