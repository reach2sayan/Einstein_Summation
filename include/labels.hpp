//
// Created by sayan on 9/18/25.
//

#ifndef EINSTEIN_SUMMATION2_MATRIX_HPP
#define EINSTEIN_SUMMATION2_MATRIX_HPP

#include <algorithm>
#include <array>
#include <ranges>

#include <boost/hana.hpp>
#include <cstddef>
#include <utility>

template <typename LSeq, typename RSeq, typename OutSeq> struct Labels {
  static_assert(false);
};

template <char... Ls, char... Rs, char... Out>
struct Labels<boost::hana::string<Ls...>, boost::hana::string<Rs...>,
              boost::hana::string<Out...>> {
  constexpr static auto left_labels =
      boost::hana::make_tuple(boost::hana::char_c<Ls>...);
  constexpr static auto right_labels =
      boost::hana::make_tuple(boost::hana::char_c<Rs>...);
  constexpr static auto out_labels =
      boost::hana::make_tuple(boost::hana::char_c<Out>...);
  constexpr static std::size_t out_size = sizeof...(Out);
};

template <char... Ls, char... Rs, char... Out>
consteval auto make_labels(boost::hana::string<Ls...>, boost::hana::string<Rs...>,
                 boost::hana::string<Out...>) {
  return Labels<boost::hana::string<Ls...>, boost::hana::string<Rs...>,
                boost::hana::string<Out...>>{};
}

template <typename> inline constexpr bool is_labels_v = false;
template <char... Ls, char... Rs, char... Out>
inline constexpr bool
    is_labels_v<Labels<boost::hana::string<Ls...>, boost::hana::string<Rs...>,
                       boost::hana::string<Out...>>> = true;

template <typename T>
concept CLabels = is_labels_v<std::remove_cvref_t<T>>;

#endif // EINSTEIN_SUMMATION2_MATRIX_HPP
