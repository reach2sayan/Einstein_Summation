//
// Created by sayan on 9/19/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_2_HPP
#define EINSTEIN_SUMMATION2_EINSUM_2_HPP
#pragma once
#include "labels.hpp"
#include "matrices.hpp"
#include <boost/hana.hpp>

namespace {
template <typename Keys, typename Values>
consteval auto make_map_from_sequences(Keys keys, Values values) {
  return boost::hana::unpack(
      boost::hana::zip(keys, values), [](auto... tuples) {
        return boost::hana::make_map(boost::hana::make_pair(
            boost::hana::at_c<0>(tuples), boost::hana::at_c<1>(tuples))...);
      });
}

template <typename LMap, typename RMap>
consteval bool perform_input_check(LMap lmap, RMap rmap) {
  auto common_keys = boost::hana::intersection(lmap, rmap);
  auto ok = boost::hana::all_of(
      common_keys, [&](auto key) { return lmap[key] == rmap[key]; });
  return ok;
}

template <typename Extents> consteval auto make_iota(Extents extents) {
  auto iotas = boost::hana::transform(extents, [](auto v) {
    return boost::hana::unpack(
        boost::hana::make_range(boost::hana::size_c<0>, v),
        [](auto... xs) { return boost::hana::tuple_c<int, xs...>; });
  });
  return iotas;
}

#define DECAY(x) std::remove_cvref_t<x>
} // namespace

template <CLabels Labels, CMatrices Matrices> struct Einsum {
private:
  constexpr static auto lmap = make_map_from_sequences(
      DECAY(Labels)::left_labels, DECAY(Matrices)::left_extents);
  constexpr static auto rmap = make_map_from_sequences(
      DECAY(Labels)::right_labels, DECAY(Matrices)::right_extents);
  static_assert(perform_input_check(lmap, rmap), "Input check failed");

public:
  constexpr static auto out_dims =
      boost::hana::transform(DECAY(Labels)::out_labels, [](auto k) {
        return boost::hana::at_key(boost::hana::union_(lmap, rmap), k);
      });
  constexpr static auto out_index_list =
      boost::hana::cartesian_product(make_iota(out_dims));

  consteval Einsum(std::same_as<Labels> auto &&,
                   std::same_as<Matrices> auto &&) {}
};

template <CLabels Labels, CMatrices Matrices>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

#endif // EINSTEIN_SUMMATION2_EINSUM_2_HPP
