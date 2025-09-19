#include "einsum.hpp"
#include "fmt/chrono.h"
#include <fmt/core.h>
#include <iostream>
#include <vector>

// clang-format off
/*
C = np.einsum('bhwi,bhwj->bij', A, B)
assert A.shape[0] == B.shape[0]
assert A.shape[1] == B.shape[1]
assert A.shape[2] == B.shape[2]
C = np.zeros((A.shape[0], A.shape[3], B.shape[3]))
for b in range(A.shape[0]): # b indexes both A and B, or B.shape[0], which must be the same
  for i in range(A.shape[3]):
    for j in range(B.shape[3]):
      # h and w can come from either A or B
      for h in range(A.shape[1]):
        for w in range(A.shape[2]):
          C[b, i, j] += A[b, h, w, i] * B[b, h, w, j]
 */
// clang-format on

auto print_all(auto... args) {
  ((fmt::print("{}", fmt::join(args, "")),std::cout << "\n"), ...);
  std::cout << "\n";
}

int main() {
  std::vector A2{1, 4, 1, 7, 8, 1, 2, 2, 7, 4, 3, 4, 2, 4, 7, 3};
  std::vector B2{2, 5, 0, 1, 5, 7, 9, 2, 2, 3, 5, 1, 7, 5, 6, 3};
  std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdA2{A2.data()};
  std::mdspan<int, std::extents<size_t, 2, 2, 2, 2>> mdB2{B2.data()};
  Matrices m{mdA2, mdB2};
  constexpr std::string_view input = "bhwi,bhwj->bij";
  auto [l,r, out] = parse_input(input);
  Labels<cseq<'b','h','w','i'>,cseq<'b','h','w','j'>, cseq<'b','i','j'>>
  labels(l, r, out);
  Einsum ein(labels, m);

  print_all(labels.left,labels.right, m.lidx, m.ridx);
  return 0;
}
