//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_MATRIX_HPP
#define EINSTEIN_SUMMATION2_MATRIX_HPP

#include "traits.hpp"
#include <algorithm>
#include <array>
#include <ranges>

template <typename LSeq, typename RSeq, typename OutSeq> struct Labels;

template <char... Ls, char... Rs, char... Out>
struct Labels<cseq<Ls...>, cseq<Rs...>, cseq<Out...>> {
  std::array<char, sizeof...(Ls)> left{};
  std::array<char, sizeof...(Rs)> right{};
  std::array<char, sizeof...(Out)> out{Out...};
  constexpr static std::size_t out_size = sizeof...(Out);
  Labels(std::ranges::input_range auto &&left_,
         std::ranges::input_range auto &&right_,
         std::ranges::input_range auto &&out_) {
    std::ranges::move(left_.begin(), left_.end(), left.begin());
    std::ranges::move(right_.begin(), right_.end(), right.begin());
    std::ranges::move(out_.begin(), out_.end(), out.begin());
  }
};

template <typename> inline constexpr bool is_labels_v = false;
template <char... Ls, char... Rs, char... Out>
inline constexpr bool
    is_labels_v<Labels<cseq<Ls...>, cseq<Rs...>, cseq<Out...>>> = true;

template <typename T>
concept CLabels = is_labels_v<std::remove_cvref_t<T>>;

#endif // EINSTEIN_SUMMATION2_MATRIX_HPP
