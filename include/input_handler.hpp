//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_HELPERS_HPP
#define EINSTEIN_SUMMATION2_HELPERS_HPP
#pragma once
#include <string_view>
#include <tuple>

consteval auto parse_input(auto input_string) {
  auto [l,r,out] =
      boost::hana::unpack(input_string, [](auto... chars) {
        auto input = boost::hana::make_tuple(chars...);

        // get left input
        constexpr auto pos_comma = boost::hana::index_if(
            input, [](auto c) { return c == boost::hana::char_c<','>; });
        auto ls = boost::hana::take_front(input, *pos_comma);

        // remove left input
        auto rest =
            boost::hana::drop_front(input, *pos_comma + boost::hana::size_c<1>);

        // trim right input
        constexpr auto pos_arrow = boost::hana::index_if(
            rest, [](auto c) { return c == boost::hana::char_c<'-'>; });
        auto rs = boost::hana::take_front(rest, *pos_arrow);

        // trim right input
        auto final =
            boost::hana::drop_front(rest, *pos_comma + boost::hana::size_c<2>);

        auto tuple_to_string = [](auto tup) {
          return boost::hana::unpack(tup, [](auto... cs) {
            return boost::hana::string<decltype(cs)::value...>{};
          });
        };

        return std::make_tuple(tuple_to_string(ls), tuple_to_string(rs),
                               tuple_to_string(final));
      });

  return boost::hana::make_tuple(l,r,out);
}

consteval auto make_label_from_inputs(auto input_string) {
  auto lrout = parse_input(input_string);
  auto l = boost::hana::at_c<0>(lrout);
  auto r = boost::hana::at_c<1>(lrout);
  auto out = boost::hana::at_c<2>(lrout);
  return make_labels(l,r,out);
}

#endif // EINSTEIN_SUMMATION2_HELPERS_HPP
