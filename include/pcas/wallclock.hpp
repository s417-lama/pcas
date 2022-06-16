#pragma once

#include <cstdint>
#include <ctime>

namespace pcas {

class wallclock_native {
public:
  static void init() {}
  static void sync() {}
  static uint64_t get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + (uint64_t)ts.tv_nsec;
  }
};

}
