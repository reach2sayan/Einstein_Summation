#include "fmt/chrono.h"
#include <boost/hana.hpp>
#include "einsum.hpp"
#include "input_handler.hpp"
#include "matrices.hpp"
#include "printers.hpp"
#include "labels.hpp"
#include <fmt/core.h>
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

template <typename... T>
struct TD;

int main() {
  std::vector A2{1, 4, 1, 7, 8, 1, 2, 2, 7, 4, 3, 4, 2, 4, 7, 3};
  std::vector B2{2, 5, 0, 1, 5, 7, 9, 2, 2, 3, 5, 1, 7, 5, 6, 3};
  std::mdspan<int, std::extents<std::size_t, 2, 2, 2, 2>> mdA2{A2.data()};
  std::mdspan<int, std::extents<std::size_t, 2, 2, 2, 2>> mdB2{B2.data()};
  Matrices m{mdA2, mdB2};
  auto lstr = BOOST_HANA_STRING("bhwi");
  auto rstr = BOOST_HANA_STRING("bhwj");
  auto outstr = BOOST_HANA_STRING("bij");
  Labels labels = make_labels(lstr,rstr,outstr);
  using labels_t = decltype(labels);
  auto ll = labels_t::left_labels;
  auto rr = labels_t::right_labels;
  auto ol = labels_t::out_labels;

  auto alll = boost::hana::sort(boost::hana::concat(ll, rr));
  auto diff = boost::hana::filter(alll, [&](auto l) {
        return boost::hana::not_(boost::hana::contains(ol, l));
    });
  print_sequence(boost::hana::unique(diff));
  Einsum einsum(labels, m);
  using ein_t = decltype(einsum);
  //print_sequence(ein_t::out_index_list);
  //TD<decltype(rr)> _;
  std::cout << "\n";
}


