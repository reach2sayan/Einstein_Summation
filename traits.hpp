//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_TRAITS_HPP
#define EINSTEIN_SUMMATION2_TRAITS_HPP
#pragma once
#include <utility>
#include <type_traits>

template <char... Cs>
struct cseq : std::integer_sequence<char, Cs...> {};

#define FWD(x) std::forward<decltype(x)>(x)

#endif // EINSTEIN_SUMMATION2_TRAITS_HPP
