//
// Created by sayan on 4/26/25.
//

#ifndef LABELS_HPP
#define LABELS_HPP

#include <string_view>
#include <vector>
#include <ostream>

#ifndef HELPER_HPP
#include "helper.hpp"
#endif

class EinsumLabels {
  std::string_view out_str;
  std::vector<std::string_view> inputs;
  std::map<char, size_t> output_axis_map;
  std::multimap<char, size_t> missing_from_out;
  template <typename... Ts> friend class Einsum;

public:
  constexpr EinsumLabels(std::string_view str)
      : out_str{split_arrow(str).second},
        inputs{split_comma(split_arrow(str).first)},
        output_axis_map{make_label_axis_map(out_str)},
        missing_from_out{make_missing_axis(inputs, out_str)}
  {}

  constexpr size_t find_axis(char c);

  constexpr friend std::ostream &operator<<(std::ostream &out,
                                            const EinsumLabels &labels) {
    out << "Output: " << labels.out_str << "\n";
    for (auto input : labels.inputs) {
      out << "Input: " << input << "\n";
    }
    out << "Axis Map:\n";
    for (auto [c, i] : labels.output_axis_map) {
      out << c << " -> " << i << "\n";
    }
    out << std::endl;
    return out;
  }

  auto num_inputs() const { return inputs.size(); }
  std::string_view common_axis() const;
  auto get_map() const { return output_axis_map; }

  auto make_result_indices();
};

#endif // LABELS_HPP
