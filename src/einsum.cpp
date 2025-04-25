// Einstein_Summation.cpp : Defines the entry point for the application.
//

#include "einsum.hpp"
#include <iostream>

int main() {
  constexpr auto bind = split_arrow("ij,jk->ik");
  constexpr auto f = bind.first;
  auto labels = split_comma(f);
  for (auto word : labels) {
    std::cout << word << "\n";
  }

  int *a = new int[9]{};
  int *b = new int[9]{};
}

