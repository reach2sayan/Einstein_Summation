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

template <typename...> struct TD;

using A = Labels<'i', 'j'>;
using B = Labels<'j', 'k'>;
using Res = Labels<'i', 'k'>;

using collapsed = collapsed_dimensions<A, B, Res>::type;

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

void first_test() {
  std::vector A{1, 1, 1, 2};
  std::vector B{0, 1, 2, 3};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdA{A.data()};
  std::mdspan<int, std::extents<size_t, 2, 2>> mdB{B.data()};

  auto ks = fixed_string("ij");
  constexpr fixed_string ls("ij");
  constexpr fixed_string rs("jk");
  constexpr fixed_string ress("ik");

  using holder =
      Einsum<int, MatA, MatB, label_t<ls>, label_t<rs>, label_t<ress>>;
  holder a{mdA, mdB, ls, rs, ress};

  std::cout << a << std::endl;

  using outindex =
      map_flatten_tuple_t<cartesian_from_labeled_dims_t<holder::output_labels>>;
  using collapsed_index =
      map_flatten_tuple_t<cartesian_from_labeled_dims_t<holder::collapsed_labels>>;

  print_outer(outindex{});
  print_outer(collapsed_index{});
  using right_labels = holder::right_labels;
  using left_labels = holder::left_labels;
  using output_labels = holder::output_labels;
  using collapsed_labels = holder::collapsed_labels;

  using outindex_f = std::tuple_element_t<0, outindex>;
  using colindex_f = std::tuple_element_t<0, collapsed_index>;
  using ridx =
        flatten_tuple_t<decltype(build_result_tuple<right_labels, output_labels, outindex,
                                    collapsed_labels, collapsed_index>())>;
  using lidx =
        flatten_tuple_t<decltype(build_result_tuple<left_labels, output_labels, outindex,
                                    collapsed_labels, collapsed_index>())>;
  outindex _o{};

  outindex_f _of{};
  ridx _ridx{};
  lidx _lidx{};

  a.print_eval();
  a.eval();
  print_2dmd_span(std::cout, a.get_result());
  auto _ = 42;
}

void second_test() {

  std::vector A2{1, 4, 1, 7, 8, 1, 2, 2, 7, 4, 3, 4, 2, 4, 7, 3};
  std::vector B2{2, 5, 0, 1, 5, 7, 9, 2, 2, 3, 5, 1, 7, 5, 6, 3};
  std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdA2{A2.data()};
  std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdB2{B2.data()};
  // bhwi,bhwj->bij
  constexpr fixed_string<4> ls2("bhwi");
  constexpr fixed_string<4> rs2("bhwj");
  constexpr fixed_string<3> ress2("bij");

  using MatA2 = Matrix<int, 2, 2, 2, 2>;
  using MatB2 = Matrix<int, 2, 2, 2, 2>;
  using holder =
      Einsum<int, MatA2, MatB2, label_t<ls2>, label_t<rs2>, label_t<ress2>>;
  holder a2{mdA2, mdB2, ls2, rs2, ress2};

  using outindex =
       map_flatten_tuple_t<cartesian_from_labeled_dims_t<holder::output_labels>>;
  using collapsed_index =
      map_flatten_tuple_t<cartesian_from_labeled_dims_t<holder::collapsed_labels>>;

  print_outer(outindex{});
  print_outer(collapsed_index{});
  using right_labels = holder::right_labels;
  using left_labels = holder::left_labels;
  using output_labels = holder::output_labels;
  using collapsed_labels = holder::collapsed_labels;

  using outindex_f = std::tuple_element_t<0, outindex>;
  using colindex_f = std::tuple_element_t<0, collapsed_index>;
  using ridx =
        flatten_tuple_t<decltype(build_result_tuple<right_labels, output_labels, outindex,
                                    collapsed_labels, collapsed_index>())>;
  using lidx =
        flatten_tuple_t<decltype(build_result_tuple<left_labels, output_labels, outindex,
                                    collapsed_labels, collapsed_index>())>;
  outindex _o{};

  outindex_f _of{};
  ridx _ridx{};
  lidx _lidx{};
  constexpr auto dims = extract_dims<output_labels>();
  int* res = new int[8];
  auto md = holder::make_mdspan(res, dims);
  int _ = 42;
  a2.print_eval();
  constexpr auto rs = holder::result_size;

  //auto md = make_md
}


void third_test() {
  using MatA = Matrix<int, 6,2,3>;
  using MatB = Matrix<int, 6,3,4>;

  constexpr fixed_string ls2("bmd");
  constexpr fixed_string rs2("bdn");
  constexpr fixed_string ress2("bmn");
  using holder2 =
      Einsum<int, MatA, MatB, label_t<ls2>, label_t<rs2>, label_t<ress2>>;

  holder2::right_labels rl{};
  //TD<cartesian_from_labeled_dims_t<holder2::output_labels>>{};
  using out_index = map_flatten_tuple_t<
      cartesian_from_labeled_dims_t<holder2::output_labels>>;
  using collapsed_index = map_flatten_tuple_t<
      cartesian_from_labeled_dims_t<holder2::collapsed_labels>>;

  using out1 = std::tuple_element_t<3, out_index>;
  using out2 = std::tuple_element_t<1, collapsed_index>;

  using result = decltype(build_result_tuple<holder2::right_labels, holder2::output_labels, out1, holder2::collapsed_labels, out2>());
  print_outer(out_index{});
  print_outer(collapsed_index{});
}

int main() {
  first_test();
  //second_test();
  //third_test();
}
