//
// Created by sayan on 4/25/25.
//

#ifndef EINSUM_HPP
#define EINSUM_HPP

#ifndef HELPER_HPP
#include "helper.hpp"
#endif

#ifndef MATRIXHOLDER_HPP
#include "matrixholder.hpp"
#endif

#ifndef LABELS_HPP
#include "labels.hpp"
#endif

#include <array>
#include <experimental/mdspan>
#include <iostream>
#include <tuple>
#include <ostream>

template <typename... Ts> class Einsum {
  MatrixHolder<Ts...> matrices;
  EinsumLabels labels;

public:
  Einsum(std::string_view str, Ts... mats) : matrices{mats...}, labels{str} {}

  friend std::ostream &operator<<(std::ostream &out, const Einsum &einsum) {
    out << "\n";
    out << einsum.labels;
  }
  auto make_result_indices();
  bool validate() { return true; }
};

#endif // EINSUM_HPP
