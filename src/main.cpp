//
// Created by sayan on 4/25/25.
//
#include "einsum.hpp"
#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>

using MatA = Matrix<int, 2, 2>;
using MatB = Matrix<int, 2, 2>;
using LabelsA = Labels<'i', 'j'>;
using LabelsB = Labels<'j', 'k'>;
using LabelsR = Labels<'i', 'k'>;
const char strp[4] = "ijk";
std::string_view str(strp);

template<typename...>
struct TD;

using A = Labels<'i', 'j'>;
using B = Labels<'j', 'k'>;
using Res = Labels<'i', 'k'>;

using collapsed = collapsed_dimensions<A, B, Res>::type;

template <typename T>
void print_integral_constant(T) {
  std::cout << T::value;
}

// Print inner tuple like (0, 1)
template <typename Tuple, std::size_t... Is>
void print_inner(const Tuple& t, std::index_sequence<Is...>) {
  std::cout << "(";
  ((std::cout << std::tuple_element_t<Is, Tuple>::value << (Is + 1 < sizeof...(Is) ? ", " : "")), ...);
  std::cout << ")";
}

// Print outer tuple of tuples
template <typename... Tuples>
void print_outer(const std::tuple<Tuples...>&) {
  ((print_inner<Tuples>(Tuples{}, std::make_index_sequence<std::tuple_size_v<Tuples>>{}), std::cout << "\n"), ...);
}

void first_test() {
  std::vector A{1, 1, 1, 2};
  std::vector B{0, 1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

  constexpr fixed_string<2> ls("ij");
  constexpr fixed_string<2> rs("jk");
  constexpr fixed_string<2> ress("ik");

  using holder = Einsum<int, MatA, MatB, label_t<ls>, label_t<rs>, label_t<ress>>;
  holder a{mdA, mdB, ls, rs, ress};

  std::cout << a << std::endl;

  using outindex = flatten_tuple_t<cartesian_from_labeled_dims_t<holder::output_labels>>;
  using collapsed_index = cartesian_from_labeled_dims_t<holder::collapsed_labels>;

  print_outer(outindex{});
  print_outer(collapsed_index{});
}

int main() {


  std::vector A2{1,4,1,7, 8,1,2,2, 7,4,3,4};
  std::vector B2{2,5, 0,1, 5,7, 9,2};
  std::mdspan<int, std::extents<size_t, 3, 4>> mdA2{A2.data()};
  std::mdspan<int, std::extents<size_t, 4, 2>> mdB2{B2.data()};

  constexpr fixed_string<2> ls2("ij");
  constexpr fixed_string<2> rs2("jk");
  constexpr fixed_string<2> ress2("ki");

  using MatA2 = Matrix<int, 3, 4>;
  using MatB2 = Matrix<int, 4, 2>;
  using holder2 = Einsum<int, MatA2, MatB2, label_t<ls2>, label_t<rs2>, label_t<ress2>>;
  holder2 a2{mdA2, mdB2, ls2, rs2, ress2};

  using outindex = flatten_tuple_t<cartesian_from_labeled_dims_t<holder2::output_labels>>;
  using collapsed_index = cartesian_from_labeled_dims_t<holder2::collapsed_labels>;

  std::cout << a2 << std::endl;
  print_outer(outindex{});
  print_outer(collapsed_index{});
  //int _ = 42;
}