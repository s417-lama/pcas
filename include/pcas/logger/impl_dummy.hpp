#pragma once

#include <cstdint>

namespace pcas {
namespace logger {

template <typename P>
class impl_dummy {
  using kind = typename P::logger_kind_t;

public:
  using begin_data_t = void*;

  static void init(int, int) {}
  static void flush(uint64_t, uint64_t) {}
  static void flush_and_print_stat(uint64_t, uint64_t) {}
  static void warmup() {}
  static void clear() {}
  template <typename kind::value K>
  static begin_data_t begin_event() { return nullptr; }
  template <typename kind::value K>
  static void end_event(begin_data_t) {}
  template <typename kind::value K, typename Misc>
  static void end_event(begin_data_t, Misc) {}
};

}
}
