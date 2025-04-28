//
// Created by sayan on 4/28/25.
//

#ifndef TYPELIST_HPP
#define TYPELIST_HPP


#include <tuple>

template<typename...Ts>
struct typelist {
  using type = std::tuple<Ts...>;
  constexpr static type item{};
  constexpr auto operator[](size_t i) const { return std::get<i>(item); }
};

template<typename...Ts>
using typelist_t = typename typelist<Ts...>::type;

template<size_t I, char C>
struct indexed_char {
  using type = std::pair<std::integral_constant<size_t, I>, std::integral_constant<char, C>>;
};

template<char... Cs>
auto make_indexed_list() {
  auto helper =
      []<size_t... IIs, char... CCs>(std::index_sequence<IIs...>,
                                     std::integer_sequence<char, CCs...>) {
        return std::tuple<typename indexed_char<IIs, CCs>::type...>{};
      };
  return helper(
      std::make_index_sequence<sizeof...(Cs)>{},
      std::integer_sequence<char, Cs...>{}
  );
}

template<char... Cs>
struct axis_list {
  using type = decltype(make_indexed_list<Cs...>());
};

using axis_list_t = axis_list<'a','b'>::type;

template<std::size_t... Is>
constexpr auto string_to_chars(const char* str, std::index_sequence<Is...>) {
  return std::integer_sequence<char, str[Is]...>{};
}
constexpr char* str = "ijk";
constexpr auto chars = string_to_chars(str, std::make_index_sequence<4>{});

#endif //TYPELIST_HPP
