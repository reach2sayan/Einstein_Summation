//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_HELPERS_HPP
#define EINSTEIN_SUMMATION2_HELPERS_HPP
#pragma once
#include "fmt/printf.h"
#include "labels.hpp"
#include "matrices.hpp"
#include <array>
#include <ranges>
#include <string_view>

consteval auto parse_input(std::string_view input) {
  auto in_iter = input.find("->");
  auto in = input.substr(0, in_iter);
  auto mid_iter = in.find(',');
  auto left = in.substr(0, mid_iter);
  auto right = in.substr(mid_iter + 1);

  auto out = input.substr(in_iter + 2);
  return std::make_tuple(left, right, out);
}

template <typename T, typename U, std::size_t N>
constexpr auto zip_arrays(const std::array<T, N> &a,
                          const std::array<U, N> &b) {

  auto _zip_arrays_impl =
      []<typename TT, typename UU, std::size_t NN, std::size_t... Is>(
          const std::array<TT, NN> &a, const std::array<UU, NN> &b,
          std::index_sequence<Is...>) {
        return std::array<std::pair<TT, UU>, NN>{{{a[Is], b[Is]}...}};
      };
  return _zip_arrays_impl(a, b, std::make_index_sequence<N>{});
}

constexpr bool input_verifier(CLabels auto &&labels,
                              CMatrices auto &&matrices) {
  auto &&lzipped = zip_arrays(FWD(labels).left, FWD(matrices).lidx);
  auto &&rzipped = zip_arrays(FWD(labels).right, FWD(matrices).ridx);

  return std::ranges::none_of(std::views::zip(FWD(lzipped), FWD(rzipped)),
                              [](auto &&pair) {
                                auto &&[lzip, rzip] = pair;
                                auto &&[ll, ldim] = lzip;
                                auto &&[rl, rdim] = rzip;
                                return ll == rl && ldim != rdim;
                              });
}

#endif // EINSTEIN_SUMMATION2_HELPERS_HPP
