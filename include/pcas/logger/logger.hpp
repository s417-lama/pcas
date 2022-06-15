#pragma once

#include <cstdio>
#include <cstdint>

#include "pcas/logger/kind.hpp"
#include "pcas/logger/policy_dummy.hpp"
#include "pcas/logger/policy_trace.hpp"

namespace pcas {
namespace logger_impl {

template <typename P>
class logger_if {
public:
  using begin_data_t = typename P::begin_data_t;

  static void init(int rank, int n_ranks) {
    P::init(rank, n_ranks);
  }

  static void flush(uint64_t t_begin, uint64_t t_end) {
    P::flush(t_begin, t_end);
  }

  static void flush_and_print_stat(uint64_t t_begin, uint64_t t_end) {
    P::flush_and_print_stat(t_begin, t_end);
  }

  static void warmup() {
    P::warmup();
  }

  static void clear() {
    P::clear();
  }

  template <kind::value K>
  static begin_data_t begin_event() {
    return P::template begin_event<K>();
  }

  template <kind::value K>
  static void end_event(begin_data_t bd) {
    P::template end_event<K>(bd);
  }

  template <kind::value K, typename MISC>
  static void end_event(begin_data_t bd, MISC m) {
    P::template end_event<K, MISC>(bd, m);
  }

  template <kind::value K>
  class scope_event {
    begin_data_t bd_;
  public:
    scope_event() {
      bd_ = begin_event<K>();
    };
    ~scope_event() {
      end_event<K>(bd_);
    }
  };

  template <kind::value K, typename MISC>
  class scope_event_m {
    begin_data_t bd_;
    MISC m_;
  public:
    scope_event_m(MISC m) {
      bd_ = begin_event<K>();
      m_ = m;
    };
    ~scope_event_m() {
      end_event<K>(bd_, m_);
    }
  };

  template <kind::value K>
  static scope_event<K> record() {
    return scope_event<K>();
  }

  template <kind::value K, typename Misc>
  static scope_event_m<K, Misc> record(Misc m) {
    return scope_event_m<K, Misc>(m);
  }

};

}

#ifndef PCAS_LOGGER_POLICY
#define PCAS_LOGGER_POLICY policy_dummy
#endif

using logger = logger_impl::logger_if<logger_impl::PCAS_LOGGER_POLICY>;

#undef PCAS_LOGGER_POLICY

using logger_kind = logger_impl::kind::value;

}
