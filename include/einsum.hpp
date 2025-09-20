//
// Created by sayan on 9/19/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_2_HPP
#define EINSTEIN_SUMMATION2_EINSUM_2_HPP
#pragma once
#include <boost/hana.hpp>
#include "labels.hpp"
#include "matrices.hpp"
#include "printers.hpp"
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

template<typename Dims>
consteval auto get_extents(Dims dims) {
  auto extent = boost::hana::unpack(dims, [](auto... dims) {
    return std::extents<std::size_t, dims...>();
  });
  return extent;
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
      boost::hana::transform(DECAY(Labels)::out_labels, [](auto key) {
        return boost::hana::at_key(boost::hana::union_(lmap, rmap), key);
      });
  constexpr static auto out_index_list =
      boost::hana::cartesian_product(make_iota(out_dims));

  constexpr static auto collapsed_dims =
      boost::hana::transform(DECAY(Labels)::collapsed_labels, [](auto key) {
        return boost::hana::at_key(boost::hana::union_(lmap, rmap), key);
      });
  constexpr static auto collapsed_index_list =
      boost::hana::cartesian_product(make_iota(collapsed_dims));

  constexpr static auto output_iterator_label_map =
      make_output_iterator_label_map(out_index_list, DECAY(Labels)::out_labels);

  constexpr static auto collapsed_iterator_label_map =
      make_output_iterator_label_map(collapsed_index_list, DECAY(Labels)::collapsed_labels);

  constexpr Einsum(std::same_as<Labels> auto &&,
                   std::same_as<Matrices> auto &&) noexcept {}
  constexpr static auto extents = get_extents(out_dims);
  constexpr void eval() const;
private:
  constexpr static auto output_size = boost::hana::fold_left(
      out_dims, 1, [](auto x, auto y) { return x * y.value; });
  std::array<value_type, output_size> result{};
  std::mdspan<value_type, DECAY(decltype(extents))> output{result.data(), extents};
};

template <CLabels Labels, CMatrices Matrices>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

template <CLabels Labels, CMatrices Matrices>
constexpr void Einsum<Labels, Matrices>::eval() const {

  boost::hana::for_each(out_index_list, [](auto out_index) {
    boost::hana::for_each(collapsed_index_list, [&](auto collapsed_index) {
      std::cout << "(" << boost::hana::at_c<0>(out_index) << ","
                << boost::hana::at_c<1>(out_index) << ","
                << boost::hana::at_c<2>(out_index) << ") - ("
                << boost::hana::at_c<0>(collapsed_index) << ","
                << boost::hana::at_c<1>(collapsed_index) << ")\n";
    });
  });
}

#endif // EINSTEIN_SUMMATION2_EINSUM_2_HPP
