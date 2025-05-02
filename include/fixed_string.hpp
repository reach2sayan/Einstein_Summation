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

template <char... Cs> struct Labels;
template <typename Tuple> struct array_of;

template <fixed_string fs> constexpr auto make_labels() {
  auto helper = []<std::size_t... Is>(std::index_sequence<Is...>) {
    return Labels<fs[Is]...>{};
  };
  return helper(std::make_index_sequence<fs.size()>{});
}

template <typename TupleA, typename TupleB> constexpr bool validity_checker() {
  for (auto &&lmap : array_of<TupleA>::value) {
    for (auto &&rmap : array_of<TupleB>::value) {
      if (lmap.first == rmap.first && lmap.second != rmap.second)
        return false;
    }
  }
  return true;
}

template<std::size_t N>
fixed_string(const char (&str)[N + 1]) -> fixed_string<N>;

template <fixed_string fs> using label_t = decltype(make_labels<fs>());
