//
// Created by sayan on 9/19/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_2_HPP
#define EINSTEIN_SUMMATION2_EINSUM_2_HPP
#pragma once
#include "labels.hpp"
#include "matrices.hpp"
#include "printers.hpp"
#include <boost/hana.hpp>
namespace {
template <typename Keys, typename Values>
consteval auto make_map_from_sequences(Keys keys, Values values) {
  auto retmap =
      boost::hana::unpack(boost::hana::zip(keys, values), [](auto... tuples) {
        return boost::hana::make_map(boost::hana::make_pair(
            boost::hana::at_c<0>(tuples), boost::hana::at_c<1>(tuples))...);
      });
  return retmap;
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

template <typename ValueList, typename Key>
consteval auto make_output_iterator_label_map(ValueList iterator_indices,
                                              Key label) {
  auto maps = boost::hana::transform(iterator_indices, [&](auto tup) {
    auto pairs = boost::hana::zip_with(
        [](auto k, auto v) { return boost::hana::make_pair(k, v); }, label,
        tup);
    return boost::hana::unpack(pairs, boost::hana::make_map);
  });
  return maps;
}

template <typename Labels, typename LMap, typename RMap>
consteval auto make_label_and_dim_map(Labels labels, LMap lmap, RMap rmap) {
  auto retmap = boost::hana::transform(labels, [&](auto key) {
    auto combined_map = boost::hana::union_(lmap, rmap);
    return boost::hana::at_key(combined_map, key);
  });
  return retmap;
}

#define DECAY(x) std::remove_cvref_t<x>
} // namespace
#ifndef NDEBUG
#define private public
#endif

template <CLabels Labels, CMatrices Matrices> struct Einsum {
private:
  using value_type = DECAY(Matrices)::value_type;
  constexpr static auto lmap = make_map_from_sequences(
      DECAY(Labels)::left_labels, DECAY(Matrices)::left_extents);
  constexpr static auto rmap = make_map_from_sequences(
      DECAY(Labels)::right_labels, DECAY(Matrices)::right_extents);
  static_assert(perform_input_check(lmap, rmap), "Input check failed");

public:
  constexpr static auto out_dims =
      make_label_and_dim_map(DECAY(Labels)::out_labels, lmap, rmap);

  constexpr static auto out_index_list =
      boost::hana::cartesian_product(make_iota(out_dims));

  constexpr static auto output_iterator_label_map =
      make_output_iterator_label_map(out_index_list, DECAY(Labels)::out_labels);

  constexpr static auto collapsed_dims =
      make_label_and_dim_map(DECAY(Labels)::collapsed_labels, lmap, rmap);

  constexpr static auto collapsed_index_list =
      boost::hana::cartesian_product(make_iota(collapsed_dims));

  constexpr static auto collapsed_iterator_label_map =
      make_output_iterator_label_map(collapsed_index_list,
                                     DECAY(Labels)::collapsed_labels);

  constexpr Einsum(std::same_as<Labels> auto &&,
                   std::same_as<Matrices> auto &&) noexcept {}

  constexpr void eval() const;

private:
  constexpr static auto extents =
      boost::hana::unpack(out_dims, [](auto... dims) {
        return std::extents<std::size_t, dims...>();
      });
  constexpr static auto output_size = boost::hana::fold_left(
      out_dims, 1, [](auto x, auto y) { return x * y.value; });
  std::array<value_type, output_size> result{value_type{}};
  std::mdspan<value_type, DECAY(decltype(extents))> output_span{result.data(),
                                                                extents};
};

template <CLabels Labels, CMatrices Matrices>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

template <CLabels Labels, CMatrices Matrices>
constexpr void Einsum<Labels, Matrices>::eval() const {
  boost::hana::for_each(out_index_list, [&](auto out_indices) {
    boost::hana::for_each(collapsed_index_list, [&](auto collapsed_indices) {
      boost::hana::unpack(out_indices, [&](auto... out_index) {
        output_span[out_index...] = 5;
      });
    });
  });
  for (auto i = 0; i < output_span.extent(0); ++i) {
    for (auto j = 0; j < output_span.extent(1); ++j) {
      std::cout << output_span[i, j] << " ";
    }
    std::cout << std::endl;
  }
}

#endif // EINSTEIN_SUMMATION2_EINSUM_2_HPP
