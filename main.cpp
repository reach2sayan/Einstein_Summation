#include "einsum.hpp"
#include "input_handler.hpp"

#include <vector>

template <typename... T> struct TD;
int main() {
  //std::vector A2{1, 4, 1, 7, 8, 1, 2, 2, 7, 4, 3, 4, 2, 4, 7, 3};
  //std::vector B2{2, 5, 0, 1, 5, 7, 9, 2, 2, 3, 5, 1, 7, 5, 6, 3};
  std::vector A2{11, 12, 13, 14, 21, 22, 23, 24,31, 32, 33, 34,41, 42, 43, 44};
  std::vector B2{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

  //std::mdspan<int, std::extents<std::size_t, 2, 2, 2, 2>> mdA2{A2.data()};
  //std::mdspan<int, std::extents<std::size_t, 2, 2, 2, 2>> mdB2{B2.data()};
  std::mdspan<int, std::extents<std::size_t, 4,4>> mdA2{A2.data()};
  std::mdspan<int, std::extents<std::size_t, 4,4>> mdB2{B2.data()};
  Matrices m{mdA2, mdB2};
  //auto [l,r,out] = parse_input("ij,jk->ik");
  make_einsum(einsum, "ij,jk->ik", mdA2, mdB2);
  einsum.eval();
  auto res = einsum.get_result();

  for (auto i = 0; i < 4; i++) {
    for (auto j = 0; j < 4; j++) {
      std::cout << res[i,j] << " ";
    }
    std::cout << "\n";
  }
}

/*

*/
