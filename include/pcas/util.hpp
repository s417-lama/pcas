#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <algorithm>
#include <tuple>

#ifdef DOCTEST_LIBRARY_INCLUDED

#define PCAS_TEST_CASE(name)            DOCTEST_TEST_CASE(name)
#define PCAS_SUBCASE(name)              DOCTEST_SUBCASE(name)
#define PCAS_CHECK(cond)                DOCTEST_CHECK(cond)
#define PCAS_CHECK_MESSAGE(cond, ...)   DOCTEST_CHECK_MESSAGE(cond, __VA_ARGS__)
#define PCAS_REQUIRE(cond)              DOCTEST_REQUIRE(cond)
#define PCAS_REQUIRE_MESSAGE(cond, ...) DOCTEST_REQUIRE_MESSAGE(cond, __VA_ARGS__)

#else

#include <cassert>

#define PCAS_CONCAT_(x, y) x##y
#define PCAS_CONCAT(x, y) PCAS_CONCAT_(x, y)
#ifdef __COUNTER__
#define PCAS_ANON_NAME(x) PCAS_CONCAT(x, __COUNTER__)
#else
#define PCAS_ANON_NAME(x) PCAS_CONCAT(x, __LINE__)
#endif

#define PCAS_TEST_CASE(name)            [[maybe_unused]] static inline void PCAS_ANON_NAME(__pcas_test_anon_fn)()
#define PCAS_SUBCASE(name)
#define PCAS_CHECK(cond)                PCAS_ASSERT(cond)
#define PCAS_CHECK_MESSAGE(cond, ...)   PCAS_ASSERT(cond)
#define PCAS_REQUIRE(cond)              PCAS_ASSERT(cond)
#define PCAS_REQUIRE_MESSAGE(cond, ...) PCAS_ASSERT(cond)

#endif

#ifdef NDEBUG
#define PCAS_ASSERT(cond) do { (void)sizeof(cond); } while (0)
#else
#define PCAS_ASSERT(cond) assert(cond)
#endif

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

PCAS_TEST_CASE("[pcas::util] calculate local block size") {
  PCAS_CHECK(local_block_size(min_block_size * 4     , 4) == min_block_size    );
  PCAS_CHECK(local_block_size(min_block_size * 12    , 4) == min_block_size * 3);
  PCAS_CHECK(local_block_size(min_block_size * 13    , 4) == min_block_size * 4);
  PCAS_CHECK(local_block_size(min_block_size * 12 + 1, 4) == min_block_size * 4);
  PCAS_CHECK(local_block_size(min_block_size * 12 - 1, 4) == min_block_size * 3);
  PCAS_CHECK(local_block_size(1                      , 4) == min_block_size    );
  PCAS_CHECK(local_block_size(1                      , 1) == min_block_size    );
  PCAS_CHECK(local_block_size(min_block_size * 3     , 1) == min_block_size * 3);
}

inline auto block_index_info(uint64_t index,
                             uint64_t size,
                             int      nproc) {
  PCAS_CHECK(index < size);
  uint64_t size_l = local_block_size(size, nproc);
  int      owner  = index / size_l;
  uint64_t idx_b  = owner * size_l;
  uint64_t idx_e  = std::min((owner + 1) * size_l, size);
  return std::make_tuple(owner, idx_b, idx_e);
}

PCAS_TEST_CASE("[pcas::util] get block information at specified index") {
  int mb = min_block_size;
  PCAS_CHECK(block_index_info(0         , mb * 4     , 4) == std::make_tuple(0, 0     , mb         ));
  PCAS_CHECK(block_index_info(mb        , mb * 4     , 4) == std::make_tuple(1, mb    , mb * 2     ));
  PCAS_CHECK(block_index_info(mb * 2    , mb * 4     , 4) == std::make_tuple(2, mb * 2, mb * 3     ));
  PCAS_CHECK(block_index_info(mb * 3    , mb * 4     , 4) == std::make_tuple(3, mb * 3, mb * 4     ));
  PCAS_CHECK(block_index_info(mb * 4 - 1, mb * 4     , 4) == std::make_tuple(3, mb * 3, mb * 4     ));
  PCAS_CHECK(block_index_info(0         , mb * 12    , 4) == std::make_tuple(0, 0     , mb * 3     ));
  PCAS_CHECK(block_index_info(mb        , mb * 12    , 4) == std::make_tuple(0, 0     , mb * 3     ));
  PCAS_CHECK(block_index_info(mb * 3    , mb * 12    , 4) == std::make_tuple(1, mb * 3, mb * 6     ));
  PCAS_CHECK(block_index_info(mb * 11   , mb * 12 - 1, 4) == std::make_tuple(3, mb * 9, mb * 12 - 1));
}

}
