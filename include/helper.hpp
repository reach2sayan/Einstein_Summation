//
// Created by sayan on 5/4/25.
//

#include <utility>
#include <iostream>

template <typename T> void print_integral_constant(T) { std::cout << T::value; }

// Print inner tuple like (0, 1)
template <typename Tuple, std::size_t... Is>
void print_inner(const Tuple &t, std::index_sequence<Is...>) {
  std::cout << "(";
  ((std::cout << std::tuple_element_t<Is, Tuple>::value
              << (Is + 1 < sizeof...(Is) ? ", " : "")),
   ...);
  std::cout << ")";
}

// Print outer tuple of tuples
template <typename... Tuples> void print_outer(const std::tuple<Tuples...> &) {

  ((print_inner<Tuples>(Tuples{},
                        std::make_index_sequence<std::tuple_size_v<Tuples>>{}),
    std::cout << "\n"),
   ...);
}

constexpr auto print_2dmd_span(std::ostream &out, auto &&mdspan) {
  for (std::size_t i = 0; i < mdspan.extent(0); ++i) {
    for (std::size_t j = 0; j < mdspan.extent(1); ++j) {
      out << mdspan[i, j] << " ";
    }
    out << "\n";
  }
  out << "\n";
}

template <typename Tuple, std::size_t... Is>
constexpr void print_tuple(const Tuple & /*tuple*/, const char *name,
                           std::index_sequence<Is...>) {
  ((std::cout << (Is == 0 ? "" : ", ") << name << "("
              << std::tuple_element_t<Is, Tuple>::value << ")"),
   ...);
}

template <typename Tuple>
constexpr void print_named_tuple(const Tuple &tuple, const char *name) {
  constexpr std::size_t N = std::tuple_size_v<Tuple>;
  std::cout << "(";
  print_tuple(tuple, name, std::make_index_sequence<N>{});
  std::cout << ")";
}