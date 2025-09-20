//
// Created by sayan on 9/19/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_2_HPP
#define EINSTEIN_SUMMATION2_EINSUM_2_HPP
#pragma once
#include "labels.hpp"
#include "matrices.hpp"
#include <unordered_map>
#include <utility>
namespace {
template <typename Keys, typename Values>
consteval auto make_map_from_sequences(Keys keys, Values values) {
  return boost::hana::unpack(
      boost::hana::zip(keys, values), [](auto... tuples) {
        return boost::hana::make_map(boost::hana::make_pair(
            boost::hana::at_c<0>(tuples), boost::hana::at_c<1>(tuples))...);
      });
}
} // namespace

template <CLabels Labels, CMatrices Matrices> struct Einsum {
  constexpr static auto lmap =
      make_map_from_sequences(std::remove_cvref_t<Labels>::left_labels,
                              std::remove_cvref_t<Matrices>::left_extents);
  constexpr static auto rmap =
      make_map_from_sequences(std::remove_cvref_t<Labels>::right_labels,
                              std::remove_cvref_t<Matrices>::right_extents);
  consteval Einsum(std::same_as<Labels> auto &&, std::same_as<Matrices> auto &&) {}
};

template <CLabels Labels, CMatrices Matrices>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

#endif // EINSTEIN_SUMMATION2_EINSUM_2_HPP
