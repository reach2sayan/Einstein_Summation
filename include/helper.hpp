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