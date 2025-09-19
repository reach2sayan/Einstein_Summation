//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_EINSUM_HPP
#define EINSTEIN_SUMMATION2_EINSUM_HPP

#include "input_handler.hpp"
#include "labels.hpp"
#include "matrices.hpp"

template <CLabels Labels, CMatrices Matrices> struct Einsum {
  using LM = decltype(std::declval<Matrices>().left);
  using RM = decltype(std::declval<Matrices>().right);
  using LL = decltype(std::declval<Labels>().left);
  using RL = decltype(std::declval<Labels>().right);
  using OUT = decltype(std::declval<Labels>().out);
  LM lm;
  RM rm;
  LL ll;
  RL rl;
  OUT out;
  constexpr Einsum(std::same_as<Labels> auto &&labels,
         std::same_as<Matrices> auto &&matrices)
      : lm{FWD(matrices).left}, rm{FWD(matrices).right}, ll{FWD(labels).left},
        rl{FWD(labels).right}, out{FWD(labels).out} {
    if (!input_verifier(labels, matrices)) {
      throw std::runtime_error("Invalid input");
    }
  }
};

template <CMatrices Matrices, CLabels Labels>
Einsum(Labels &&, Matrices &&) -> Einsum<Labels, Matrices>;

#endif // EINSTEIN_SUMMATION2_EINSUM_HPP
