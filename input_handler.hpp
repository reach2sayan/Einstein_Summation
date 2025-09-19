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
#include <map>
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

#endif // EINSTEIN_SUMMATION2_HELPERS_HPP
