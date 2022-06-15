#pragma once

#include <cstdint>

#include "pcas/logger/kind.hpp"

namespace pcas {
namespace logger_impl {

class policy_dummy {
public:
  using begin_data_t = void*;

  static void init(int rank, int n_ranks) {}
  static void flush(uint64_t t_begin, uint64_t t_end) {}
  static void flush_and_print_stat(uint64_t t_begin, uint64_t t_end) {}
  static void warmup() {}
  static void clear() {}
  template <kind::value K>
  static begin_data_t begin_event() { return nullptr; }
  template <kind::value K>
  static void end_event(begin_data_t bd) {}
  template <kind::value K, typename MISC>
  static void end_event(begin_data_t bd, MISC m) {}
};

}
}
