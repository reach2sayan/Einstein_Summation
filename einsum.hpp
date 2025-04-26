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
    return out;
  }
  constexpr std::vector<std::vector<size_t>> make_result_indices() {

    auto helper = []<size_t... Is>(auto &&inputs, auto &&mat,
                                   std::index_sequence<Is...>) {
      return make_label_and_extents_map(inputs, std::get<Is>(mat.matrices)...);
    };
    auto label_map = helper(labels.inputs, matrices,
                            std::make_index_sequence<sizeof...(Ts)>{});
    auto iotas = make_iotas(label_map);
    std::vector<std::vector<size_t>> product;
    cartesian_product(iotas, product);
    return product;
  }
  bool validate() { return true; }
};

#endif // EINSUM_HPP
