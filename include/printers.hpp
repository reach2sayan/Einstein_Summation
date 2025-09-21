//
// Created by sayan on 9/19/25.
//

#ifndef EINSTEIN_SUMMATION2_PRINTERS_HPP
#define EINSTEIN_SUMMATION2_PRINTERS_HPP
#pragma once
#include <iostream>

template <typename Tuple> void print_tuple(const Tuple &tup) {
  std::cout << "(";
  bool first = true;
  boost::hana::for_each(tup, [&](auto const &x) {
    if (!first)
      std::cout << ", ";
    if constexpr (boost::hana::IntegralConstant<decltype(x)>::value) {
      std::cout << x.value;
    } else {
      std::cout << x;
    }
    first = false;
  });
  std::cout << ")";
}

template <typename Sequence> void print_sequence(const Sequence &seq) {
  std::cout << "[";
  bool first = true;
  boost::hana::for_each(seq, [&](auto const &x) {
    if (!first)
      std::cout << ", ";
    // if element is itself a tuple, delegate to tuple printer
    if constexpr (boost::hana::Sequence<decltype(x)>::value) {
      print_tuple(x);
    } else if constexpr (boost::hana::IntegralConstant<decltype(x)>::value) {
      std::cout << x.value;
    } else {
      std::cout << x;
    }
    first = false;
  });
  std::cout << "]";
}

template <typename Map> void print_map(const Map &map) {
  std::cout << "{";
  bool first = true;
  boost::hana::for_each(map, [&](auto const &pair) {
    if (!first)
      std::cout << ", ";
    first = false;

    auto key = boost::hana::first(pair);
    auto value = boost::hana::second(pair);
    std::cout << key << ": " << value;
  });
  std::cout << "}\n";
}

template <typename Map, typename Label>
void print_labeled_iterators(const Map &map, Label labels) {
  boost::hana::for_each(map, [&](auto const &m) {
    std::cout << "{ ";
    boost::hana::for_each(labels, [&](auto k) {
      std::cout << k.value << ": " << boost::hana::at_key(m, k) << " ";
    });
    std::cout << "}\n";
  });
}

#endif // EINSTEIN_SUMMATION2_PRINTERS_HPP
