// Einstein_Summation.cpp : Defines the entry point for the application.
//

#include "helpers.hpp"
#include <iostream>

int main() {
    constexpr auto bind = split_arrow("ij,jk->ik");
    constexpr auto f = bind.first;
    auto vec = split_comma(f);
    for (auto word : vec) {
        std::cout << word << "\n";
    }
}
