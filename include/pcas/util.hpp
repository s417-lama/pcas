#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>

namespace pcas {

__attribute__((noinline))
inline void die(const char* fmt, ...) {
  constexpr int slen = 128;
  char msg[slen];

  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, slen, fmt, args);
  va_end(args);

  fprintf(stderr, "\x1b[31m%s\x1b[39m", msg);

  exit(1);
}

}
