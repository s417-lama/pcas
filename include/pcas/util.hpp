#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <algorithm>
#include <tuple>

#include "doctest/doctest.h"

namespace pcas {

__attribute__((noinline))
inline void die(const char* fmt, ...) {
  constexpr int slen = 128;
  char msg[slen];

  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, slen, fmt, args);
  va_end(args);

  fprintf(stderr, "\x1b[31m%s\x1b[39m\n", msg);

  exit(1);
}

constexpr uint64_t min_block_size = 4096;

inline uint64_t local_block_size(uint64_t size, int nproc) {
  int nblock_g = (size + min_block_size - 1) / min_block_size;
  int nblock_l = (nblock_g + nproc - 1) / nproc;
  return nblock_l * min_block_size;
}

TEST_CASE("calculate local block size") {
  CHECK(local_block_size(min_block_size * 4     , 4) == min_block_size    );
  CHECK(local_block_size(min_block_size * 12    , 4) == min_block_size * 3);
  CHECK(local_block_size(min_block_size * 13    , 4) == min_block_size * 4);
  CHECK(local_block_size(min_block_size * 12 + 1, 4) == min_block_size * 4);
  CHECK(local_block_size(min_block_size * 12 - 1, 4) == min_block_size * 3);
  CHECK(local_block_size(1                      , 4) == min_block_size    );
  CHECK(local_block_size(1                      , 1) == min_block_size    );
  CHECK(local_block_size(min_block_size * 3     , 1) == min_block_size * 3);
}

inline auto block_index_info(uint64_t index,
                             uint64_t size,
                             int      nproc) {
  CHECK(index < size);
  uint64_t size_l = local_block_size(size, nproc);
  int      owner  = index / size_l;
  uint64_t idx_b  = owner * size_l;
  uint64_t idx_e  = std::min((owner + 1) * size_l, size);
  return std::make_tuple(owner, idx_b, idx_e);
}

TEST_CASE("get block information at specified index") {
  int mb = min_block_size;
  CHECK(block_index_info(0         , mb * 4     , 4) == std::make_tuple(0, 0     , mb         ));
  CHECK(block_index_info(mb        , mb * 4     , 4) == std::make_tuple(1, mb    , mb * 2     ));
  CHECK(block_index_info(mb * 2    , mb * 4     , 4) == std::make_tuple(2, mb * 2, mb * 3     ));
  CHECK(block_index_info(mb * 3    , mb * 4     , 4) == std::make_tuple(3, mb * 3, mb * 4     ));
  CHECK(block_index_info(mb * 4 - 1, mb * 4     , 4) == std::make_tuple(3, mb * 3, mb * 4     ));
  CHECK(block_index_info(0         , mb * 12    , 4) == std::make_tuple(0, 0     , mb * 3     ));
  CHECK(block_index_info(mb        , mb * 12    , 4) == std::make_tuple(0, 0     , mb * 3     ));
  CHECK(block_index_info(mb * 3    , mb * 12    , 4) == std::make_tuple(1, mb * 3, mb * 6     ));
  CHECK(block_index_info(mb * 11   , mb * 12 - 1, 4) == std::make_tuple(3, mb * 9, mb * 12 - 1));
}

}
