//
// Created by sayan on 4/30/25.
//

#pragma once
#include <utility>

template <std::size_t N> struct fixed_string {
  char data[N];
  constexpr fixed_string(const char (&str)[N + 1]) {
    for (std::size_t i = 0; i < N; ++i)
      data[i] = str[i];
  }
  constexpr char operator[](std::size_t i) const { return data[i]; }
  constexpr std::size_t size() const { return N; }
};

template<std::size_t N>
fixed_string(const char (&str)[N]) -> fixed_string<N-1>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
template <typename CharT, CharT... Cs>
constexpr auto operator""_fs() {
  constexpr char str[] = {Cs...};
  return fixed_string<sizeof...(Cs)>{str};
}
#pragma GCC diagnostic pop
