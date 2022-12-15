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
#include <mpi.h>

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

__attribute__((noinline))
inline void die(const char* fmt, ...) {
  constexpr int slen = 128;
  char msg[slen];

  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, slen, fmt, args);
  va_end(args);

  fprintf(stderr, "\x1b[31m%s\x1b[39m\n", msg);
  fflush(stderr);

  std::abort();
}

template <typename T>
inline T get_env_(const char* env_var, T default_val) {
  if (const char* val_str = std::getenv(env_var)) {
    T val;
    std::stringstream ss(val_str);
    ss >> val;
    if (ss.fail()) {
      die("Environment variable '%s' is invalid.\n", env_var);
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

inline std::size_t sys_mmap_entry_limit() {
  std::ifstream ifs("/proc/sys/vm/max_map_count");
  if (!ifs) {
    die("Cannot open /proc/sys/vm/max_map_count");
  }
  std::size_t sys_limit;
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
  PCAS_CHECK(next_pow2((uint64_t(1) << 38) - 100) == uint64_t(1) << 38);
}

template <typename T>
constexpr inline bool is_pow2(T x) {
  return !(x & (x - 1));
}

template <typename T>
constexpr inline T round_down_pow2(T x, T alignment) {
  PCAS_CHECK(is_pow2(alignment));
  return x & ~(alignment - 1);
}

template <typename T>
constexpr inline T round_up_pow2(T x, T alignment) {
  PCAS_CHECK(is_pow2(alignment));
  return (x + alignment - 1) & ~(alignment - 1);
}

PCAS_TEST_CASE("[pcas::util] round up/down for integers") {
  PCAS_CHECK(is_pow2(128));
  PCAS_CHECK(!is_pow2(129));
  PCAS_CHECK(round_down_pow2(1100, 128) == 1024);
  PCAS_CHECK(round_down_pow2(128, 128) == 128);
  PCAS_CHECK(round_down_pow2(129, 128) == 128);
  PCAS_CHECK(round_down_pow2(255, 128) == 128);
  PCAS_CHECK(round_down_pow2(73, 128) == 0);
  PCAS_CHECK(round_down_pow2(0, 128) == 0);
  PCAS_CHECK(round_up_pow2(1100, 128) == 1152);
  PCAS_CHECK(round_up_pow2(128, 128) == 128);
  PCAS_CHECK(round_up_pow2(129, 128) == 256);
  PCAS_CHECK(round_up_pow2(255, 128) == 256);
  PCAS_CHECK(round_up_pow2(73, 128) == 128);
  PCAS_CHECK(round_up_pow2(0, 128) == 0);
}

class win_manager {
  MPI_Win win_ = MPI_WIN_NULL;
public:
  win_manager(MPI_Comm comm) {
    MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &win_);
    MPI_Win_lock_all(0, win_);
  }
  win_manager(MPI_Comm comm, std::size_t size, void** vm_addr) {
    MPI_Win_allocate(size, 1, MPI_INFO_NULL, comm, vm_addr, &win_);
    MPI_Win_lock_all(0, win_);
  }
  win_manager(MPI_Comm comm, void* vm_addr, std::size_t size) {
    MPI_Win_create(vm_addr,
                   size,
                   1,
                   MPI_INFO_NULL,
                   comm,
                   &win_);
    MPI_Win_lock_all(0, win_);
  }
  ~win_manager() {
    if (win_ != MPI_WIN_NULL) {
      MPI_Win_unlock_all(win_);
      MPI_Win_free(&win_);
    }
  }
  win_manager(const win_manager&) = delete;
  win_manager& operator=(const win_manager&) = delete;
  win_manager(win_manager&& wm) noexcept : win_(wm.win_) { wm.win_ = MPI_WIN_NULL; }
  win_manager& operator=(win_manager&& wm) noexcept {
    this->~win_manager();
    this->win_ = wm.win_;
    wm.win_ = MPI_WIN_NULL;
    return *this;
  }
  MPI_Win win() const { return win_; }
};

// Section
// -----------------------------------------------------------------------------

template <typename T>
using section = std::pair<T, T>;

template <typename T>
using sections = std::forward_list<section<T>>;

template <typename T>
inline section<T> section_merge(section<T> s1, section<T> s2) {
  return section<T>{std::min(s1.first, s2.first), std::max(s1.second, s2.second)};
}

template <typename T>
inline bool section_overlap(section<T> s1, section<T> s2) {
  return s1.first < s2.second && s1.second < s2.first;
}

template <typename T>
inline bool section_contain(section<T> s1, section<T> s2) {
  // true if s1 contains s2
  return s1.first <= s2.first && s2.second <= s1.second;
}

template <typename T>
inline bool sections_overlap(const sections<T>& ss, section<T> s) {
  for (const auto& s_ : ss) {
    if (s.second <= s_.first) break;
    if (s.first < s_.second) return true;
  }
  return false;
}

PCAS_TEST_CASE("[pcas::util] sections overlap") {
  sections<int> ss{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  PCAS_CHECK(sections_overlap(ss, {2, 5}));
  PCAS_CHECK(sections_overlap(ss, {11, 20}));
  PCAS_CHECK(sections_overlap(ss, {50, 100}));
  PCAS_CHECK(sections_overlap(ss, {0, 110}));
  PCAS_CHECK(sections_overlap(ss, {4, 7}));
  PCAS_CHECK(sections_overlap(ss, {10, 12}));
  PCAS_CHECK(sections_overlap(ss, {9, 21}));
  PCAS_CHECK(!sections_overlap(ss, {10, 11}));
  PCAS_CHECK(!sections_overlap(ss, {0, 2}));
  PCAS_CHECK(!sections_overlap(ss, {100, 1000}));
  PCAS_CHECK(!sections_overlap(ss, {5, 5}));
}

template <typename T>
inline bool sections_contain(const sections<T>& ss, section<T> s) {
  for (const auto& s_ : ss) {
    if (s.first < s_.first) break;
    if (s.second <= s_.second) return true;
  }
  return false;
}

PCAS_TEST_CASE("[pcas::util] sections contain") {
  sections<int> ss{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  PCAS_CHECK(sections_contain(ss, {2, 5}));
  PCAS_CHECK(sections_contain(ss, {3, 5}));
  PCAS_CHECK(sections_contain(ss, {2, 4}));
  PCAS_CHECK(sections_contain(ss, {3, 4}));
  PCAS_CHECK(sections_contain(ss, {7, 9}));
  PCAS_CHECK(sections_contain(ss, {50, 100}));
  PCAS_CHECK(!sections_contain(ss, {9, 11}));
  PCAS_CHECK(!sections_contain(ss, {3, 6}));
  PCAS_CHECK(!sections_contain(ss, {7, 10}));
  PCAS_CHECK(!sections_contain(ss, {2, 100}));
  PCAS_CHECK(!sections_contain(ss, {0, 3}));
}

template <typename T>
inline void sections_insert(sections<T>& ss, section<T> s) {
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
  sections<int> ss;
  sections_insert(ss, {2, 5});
  PCAS_CHECK(ss == (sections<int>{{2, 5}}));
  sections_insert(ss, {11, 20});
  PCAS_CHECK(ss == (sections<int>{{2, 5}, {11, 20}}));
  sections_insert(ss, {20, 21});
  PCAS_CHECK(ss == (sections<int>{{2, 5}, {11, 21}}));
  sections_insert(ss, {15, 23});
  PCAS_CHECK(ss == (sections<int>{{2, 5}, {11, 23}}));
  sections_insert(ss, {8, 23});
  PCAS_CHECK(ss == (sections<int>{{2, 5}, {8, 23}}));
  sections_insert(ss, {7, 25});
  PCAS_CHECK(ss == (sections<int>{{2, 5}, {7, 25}}));
  sections_insert(ss, {0, 7});
  PCAS_CHECK(ss == (sections<int>{{0, 25}}));
  sections_insert(ss, {30, 50});
  PCAS_CHECK(ss == (sections<int>{{0, 25}, {30, 50}}));
  sections_insert(ss, {30, 50});
  PCAS_CHECK(ss == (sections<int>{{0, 25}, {30, 50}}));
  sections_insert(ss, {35, 45});
  PCAS_CHECK(ss == (sections<int>{{0, 25}, {30, 50}}));
  sections_insert(ss, {60, 100});
  PCAS_CHECK(ss == (sections<int>{{0, 25}, {30, 50}, {60, 100}}));
  sections_insert(ss, {0, 120});
  PCAS_CHECK(ss == (sections<int>{{0, 120}}));
  sections_insert(ss, {200, 300});
  PCAS_CHECK(ss == (sections<int>{{0, 120}, {200, 300}}));
  sections_insert(ss, {600, 700});
  PCAS_CHECK(ss == (sections<int>{{0, 120}, {200, 300}, {600, 700}}));
  sections_insert(ss, {400, 500});
  PCAS_CHECK(ss == (sections<int>{{0, 120}, {200, 300}, {400, 500}, {600, 700}}));
  sections_insert(ss, {300, 600});
  PCAS_CHECK(ss == (sections<int>{{0, 120}, {200, 700}}));
  sections_insert(ss, {50, 600});
  PCAS_CHECK(ss == (sections<int>{{0, 700}}));
}

template <typename T>
inline void sections_remove(sections<T>& ss, section<T> s) {
  auto it = ss.before_begin();

  while (std::next(it) != ss.end()) {
    if (s.second <= std::next(it)->first) break;

    if (std::next(it)->second <= s.first) {
      // no overlap
      it++;
    } else if (s.first <= std::next(it)->first && std::next(it)->second <= s.second) {
      // s contains std::next(it)
      ss.erase_after(it);
      // do not increment it
    } else if (s.first <= std::next(it)->first && s.second <= std::next(it)->second) {
      // the left end of std::next(it) is overlaped
      std::next(it)->first = s.second;
      it++;
    } else if (std::next(it)->first <= s.first && std::next(it)->second <= s.second) {
      // the right end of std::next(it) is overlaped
      std::next(it)->second = s.first;
      it++;
    } else if (std::next(it)->first <= s.first && s.second <= std::next(it)->second) {
      // std::next(it) contains s
      section<T> new_s = {std::next(it)->first, s.first};
      std::next(it)->first = s.second;
      ss.insert_after(it, new_s);
      it++;
    } else {
      die("Something is wrong in sections_remove()\n");
    }
  }
}

PCAS_TEST_CASE("[pcas::util] sections remove") {
  sections<int> ss{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  sections_remove(ss, {6, 9});
  PCAS_CHECK(ss == (sections<int>{{2, 5}, {11, 20}, {50, 100}}));
  sections_remove(ss, {4, 10});
  PCAS_CHECK(ss == (sections<int>{{2, 4}, {11, 20}, {50, 100}}));
  sections_remove(ss, {70, 80});
  PCAS_CHECK(ss == (sections<int>{{2, 4}, {11, 20}, {50, 70}, {80, 100}}));
  sections_remove(ss, {18, 55});
  PCAS_CHECK(ss == (sections<int>{{2, 4}, {11, 18}, {55, 70}, {80, 100}}));
  sections_remove(ss, {10, 110});
  PCAS_CHECK(ss == (sections<int>{{2, 4}}));
  sections_remove(ss, {2, 4});
  PCAS_CHECK(ss == (sections<int>{}));
  sections_remove(ss, {2, 4});
  PCAS_CHECK(ss == (sections<int>{}));
}

template <typename T>
inline sections<T> sections_inverse(const sections<T>& ss, section<T> s) {
  sections<T> ret;
  auto it = ret.before_begin();
  for (auto [b, e] : ss) {
    if (s.first < b) {
      it = ret.insert_after(it, {s.first, std::min(b, s.second)});
    }
    if (s.first < e) {
      s.first = e;
      if (s.first >= s.second) break;
    }
  }
  if (s.first < s.second) {
    ret.insert_after(it, s);
  }
  return ret;
}

PCAS_TEST_CASE("[pcas::util] sections inverse") {
  sections<int> ss{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  PCAS_CHECK(sections_inverse(ss, {0, 120}) == (sections<int>{{0, 2}, {5, 6}, {9, 11}, {20, 50}, {100, 120}}));
  PCAS_CHECK(sections_inverse(ss, {0, 100}) == (sections<int>{{0, 2}, {5, 6}, {9, 11}, {20, 50}}));
  PCAS_CHECK(sections_inverse(ss, {0, 25}) == (sections<int>{{0, 2}, {5, 6}, {9, 11}, {20, 25}}));
  PCAS_CHECK(sections_inverse(ss, {8, 15}) == (sections<int>{{9, 11}}));
  PCAS_CHECK(sections_inverse(ss, {30, 40}) == (sections<int>{{30, 40}}));
  PCAS_CHECK(sections_inverse(ss, {50, 100}) == (sections<int>{}));
  PCAS_CHECK(sections_inverse(ss, {60, 90}) == (sections<int>{}));
  PCAS_CHECK(sections_inverse(ss, {2, 5}) == (sections<int>{}));
  PCAS_CHECK(sections_inverse(ss, {2, 6}) == (sections<int>{{5, 6}}));
  sections<int> ss_empty{};
  PCAS_CHECK(sections_inverse(ss_empty, {0, 100}) == (sections<int>{{0, 100}}));
}

// Freelist
// -----------------------------------------------------------------------------

struct span {
  std::size_t addr;
  std::size_t size;
};

inline std::optional<span> freelist_get(std::list<span>& fl, std::size_t size) {
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
  if (s.size == 0) return;

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

  std::size_t n = 100;
  for (std::size_t i = 0; i < s0.size / n; i++) {
    auto s = freelist_get(fl, n);
    PCAS_CHECK(s.has_value());
    got.push_back(*s);
  }
  PCAS_CHECK(!freelist_get(fl, n).has_value());

  // check for no overlap
  for (std::size_t i = 0; i < got.size(); i++) {
    for (std::size_t j = 0; j < got.size(); j++) {
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

  for (auto&& s : got) {
    freelist_add(fl, s);
  }

  PCAS_CHECK(fl.size() == 1);
  PCAS_CHECK(fl.begin()->addr == s0.addr);
  PCAS_CHECK(fl.begin()->size == s0.size);
}

}
