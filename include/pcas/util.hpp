#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <algorithm>
#include <tuple>
#include <sstream>
#include <forward_list>
#include <optional>
#include <random>

#ifdef DOCTEST_LIBRARY_INCLUDED

#define PCAS_TEST_CASE(name)                 DOCTEST_TEST_CASE(name)
#define PCAS_SUBCASE(name)                   DOCTEST_SUBCASE(name)
#define PCAS_CHECK(cond)                     DOCTEST_CHECK(cond)
#define PCAS_CHECK_MESSAGE(cond, ...)        DOCTEST_CHECK_MESSAGE(cond, __VA_ARGS__)
#define PCAS_REQUIRE(cond)                   DOCTEST_REQUIRE(cond)
#define PCAS_REQUIRE_MESSAGE(cond, ...)      DOCTEST_REQUIRE_MESSAGE(cond, __VA_ARGS__)
#define PCAS_CHECK_THROWS_AS(exp, exception) DOCTEST_CHECK_THROWS_AS(exp, exception)

#else

#define PCAS_CONCAT_(x, y) x##y
#define PCAS_CONCAT(x, y) PCAS_CONCAT_(x, y)
#ifdef __COUNTER__
#define PCAS_ANON_NAME(x) PCAS_CONCAT(x, __COUNTER__)
#else
#define PCAS_ANON_NAME(x) PCAS_CONCAT(x, __LINE__)
#endif

#define PCAS_TEST_CASE(name)                 [[maybe_unused]] static inline void PCAS_ANON_NAME(__pcas_test_anon_fn)()
#define PCAS_SUBCASE(name)
#define PCAS_CHECK(cond)                     PCAS_ASSERT(cond)
#define PCAS_CHECK_MESSAGE(cond, ...)        PCAS_ASSERT(cond)
#define PCAS_REQUIRE(cond)                   PCAS_ASSERT(cond)
#define PCAS_REQUIRE_MESSAGE(cond, ...)      PCAS_ASSERT(cond)
#define PCAS_CHECK_THROWS_AS(exp, exception) exp

#endif

#ifdef NDEBUG
#define PCAS_ASSERT(cond) do { (void)sizeof(cond); } while (0)
#else
#include <cassert>
#define PCAS_ASSERT(cond) assert(cond)
#endif

namespace pcas {

using obj_id_t = uint64_t;

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

template <typename T>
inline T get_env_(const char* env_var, T default_val) {
  if (const char* val_str = std::getenv(env_var)) {
    T val;
    std::stringstream ss(val_str);
    ss >> val;
    if (ss.fail()) {
      fprintf(stderr, "Environment variable '%s' is invalid.\n", env_var);
      exit(1);
    }
    return val;
  } else {
    return default_val;
  }
}

template <typename T>
inline T get_env(const char* env_var, T default_val, int rank) {
  static bool print_env = get_env_("PCAS_PRINT_ENV", false);

  T val = get_env_(env_var, default_val);
  if (print_env && rank == 0) {
    std::cout << env_var << " = " << val << std::endl;
  }
  return val;
}

inline size_t sys_mmap_entry_limit() {
  std::ifstream ifs("/proc/sys/vm/max_map_count");
  if (!ifs) {
    die("Cannot open /proc/sys/vm/max_map_count");
  }
  size_t sys_limit;
  ifs >> sys_limit;
  return sys_limit;
}

template <typename T>
inline bool empty(const std::vector<std::optional<T>>& v) {
  return std::none_of(v.begin(), v.end(),
                      [](const std::optional<T>& x) { return x.has_value(); });
}

constexpr inline uint64_t next_pow2(uint64_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

PCAS_TEST_CASE("[pcas::util] next_pow2") {
  PCAS_CHECK(next_pow2(0) == 0);
  PCAS_CHECK(next_pow2(1) == 1);
  PCAS_CHECK(next_pow2(2) == 2);
  PCAS_CHECK(next_pow2(3) == 4);
  PCAS_CHECK(next_pow2(4) == 4);
  PCAS_CHECK(next_pow2(5) == 8);
  PCAS_CHECK(next_pow2(15) == 16);
  PCAS_CHECK(next_pow2(((uint64_t)1 << 38) - 100) == (uint64_t)1 << 38);
}

// Section
// -----------------------------------------------------------------------------

using section = std::pair<uint32_t, uint32_t>;
using sections = std::forward_list<section>;

inline section section_merge(section s1, section s2) {
  return section{std::min(s1.first, s2.first), std::max(s1.second, s2.second)};
}

inline void sections_insert(sections& ss, section s) {
  auto it = ss.before_begin();

  // skip until it overlaps s (or s < it)
  while (std::next(it) != ss.end() && std::next(it)->second < s.first) it++;

  if (std::next(it) == ss.end() || s.second < std::next(it)->first) {
    // no overlap
    ss.insert_after(it, s);
  } else {
    // at least two sections are overlapping -> merge
    it++;
    *it = section_merge(*it, s);

    while (std::next(it) != ss.end() && it->second >= std::next(it)->first) {
      *it = section_merge(*it, *std::next(it));
      ss.erase_after(it);
    }
  }
}

PCAS_TEST_CASE("[pcas::util] sections insert") {
  sections ss;
  sections_insert(ss, {2, 5});
  PCAS_CHECK(ss == (sections{{2, 5}}));
  sections_insert(ss, {11, 20});
  PCAS_CHECK(ss == (sections{{2, 5}, {11, 20}}));
  sections_insert(ss, {20, 21});
  PCAS_CHECK(ss == (sections{{2, 5}, {11, 21}}));
  sections_insert(ss, {15, 23});
  PCAS_CHECK(ss == (sections{{2, 5}, {11, 23}}));
  sections_insert(ss, {8, 23});
  PCAS_CHECK(ss == (sections{{2, 5}, {8, 23}}));
  sections_insert(ss, {7, 25});
  PCAS_CHECK(ss == (sections{{2, 5}, {7, 25}}));
  sections_insert(ss, {0, 7});
  PCAS_CHECK(ss == (sections{{0, 25}}));
  sections_insert(ss, {30, 50});
  PCAS_CHECK(ss == (sections{{0, 25}, {30, 50}}));
  sections_insert(ss, {30, 50});
  PCAS_CHECK(ss == (sections{{0, 25}, {30, 50}}));
  sections_insert(ss, {35, 45});
  PCAS_CHECK(ss == (sections{{0, 25}, {30, 50}}));
  sections_insert(ss, {60, 100});
  PCAS_CHECK(ss == (sections{{0, 25}, {30, 50}, {60, 100}}));
  sections_insert(ss, {0, 120});
  PCAS_CHECK(ss == (sections{{0, 120}}));
  sections_insert(ss, {200, 300});
  PCAS_CHECK(ss == (sections{{0, 120}, {200, 300}}));
  sections_insert(ss, {600, 700});
  PCAS_CHECK(ss == (sections{{0, 120}, {200, 300}, {600, 700}}));
  sections_insert(ss, {400, 500});
  PCAS_CHECK(ss == (sections{{0, 120}, {200, 300}, {400, 500}, {600, 700}}));
  sections_insert(ss, {300, 600});
  PCAS_CHECK(ss == (sections{{0, 120}, {200, 700}}));
  sections_insert(ss, {50, 600});
  PCAS_CHECK(ss == (sections{{0, 700}}));
}

inline sections sections_inverse(const sections& ss, section s_range) {
  sections ret;
  auto it = ret.before_begin();
  for (auto [b, e] : ss) {
    if (s_range.first < b) {
      it = ret.insert_after(it, {s_range.first, std::min(b, s_range.second)});
    }
    if (s_range.first < e) {
      s_range.first = e;
      if (s_range.first >= s_range.second) break;
    }
  }
  if (s_range.first < s_range.second) {
    ret.insert_after(it, s_range);
  }
  return ret;
}

PCAS_TEST_CASE("[pcas::util] sections inverse") {
  sections ss{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  PCAS_CHECK(sections_inverse(ss, {0, 120}) == (sections{{0, 2}, {5, 6}, {9, 11}, {20, 50}, {100, 120}}));
  PCAS_CHECK(sections_inverse(ss, {0, 100}) == (sections{{0, 2}, {5, 6}, {9, 11}, {20, 50}}));
  PCAS_CHECK(sections_inverse(ss, {0, 25}) == (sections{{0, 2}, {5, 6}, {9, 11}, {20, 25}}));
  PCAS_CHECK(sections_inverse(ss, {8, 15}) == (sections{{9, 11}}));
  PCAS_CHECK(sections_inverse(ss, {30, 40}) == (sections{{30, 40}}));
  PCAS_CHECK(sections_inverse(ss, {50, 100}) == (sections{}));
  PCAS_CHECK(sections_inverse(ss, {60, 90}) == (sections{}));
  PCAS_CHECK(sections_inverse(ss, {2, 5}) == (sections{}));
  PCAS_CHECK(sections_inverse(ss, {2, 6}) == (sections{{5, 6}}));
  sections ss_empty{};
  PCAS_CHECK(sections_inverse(ss_empty, {0, 100}) == (sections{{0, 100}}));
}

// Freelist
// -----------------------------------------------------------------------------

struct span {
  uint64_t addr;
  uint64_t size;
};

inline std::optional<span> freelist_get(std::list<span>& fl, uint64_t size) {
  auto it = fl.begin();
  while (it != fl.end()) {
    if (it->size == size) {
      span ret = *it;
      fl.erase(it);
      return ret;
    } else if (it->size > size) {
      span ret {it->addr, size};
      it->addr += size;
      it->size -= size;
      return ret;
    }
    it = std::next(it);
  }
  return std::nullopt;
}

inline void freelist_add(std::list<span>& fl, span s) {
  auto it = fl.begin();
  while (it != fl.end()) {
    if (s.addr + s.size == it->addr) {
      it->addr = s.addr;
      it->size += s.size;
      return;
    } else if (s.addr + s.size < it->addr) {
      fl.insert(it, s);
      return;
    } else if (s.addr == it->addr + it->size) {
      it->size += s.size;
      auto next_it = std::next(it);
      if (next_it != fl.end() &&
          next_it->addr == it->addr + it->size) {
        it->size += next_it->size;
        fl.erase(next_it);
      }
      return;
    }
    it = std::next(it);
  }
  fl.insert(it, s);
}

PCAS_TEST_CASE("[pcas::util] freelist for span") {
  span s0 {100, 920};
  std::list<span> fl(1, s0);
  std::vector<span> got;

  uint64_t n = 100;
  for (uint64_t i = 0; i < s0.size / n; i++) {
    auto s = freelist_get(fl, n);
    PCAS_CHECK(s.has_value());
    got.push_back(*s);
  }
  PCAS_CHECK(!freelist_get(fl, n).has_value());

  // check for no overlap
  for (uint64_t i = 0; i < got.size(); i++) {
    for (uint64_t j = 0; j < got.size(); j++) {
      if (i != j) {
        PCAS_CHECK((got[i].addr + got[i].size <= got[j].addr ||
                    got[j].addr + got[j].size <= got[i].addr));
      }
    }
  }

  // random shuffle
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::shuffle(got.begin(), got.end(), engine);

  for (auto& s : got) {
    freelist_add(fl, s);
  }

  for (auto s : fl) {
    printf("%ld %ld\n", s.addr, s.size);
  }

  PCAS_CHECK(fl.size() == 1);
  PCAS_CHECK(fl.begin()->addr == s0.addr);
  PCAS_CHECK(fl.begin()->size == s0.size);
}

}
