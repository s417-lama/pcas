#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <vector>
#include <list>
#include <random>

#include "pcas/util.hpp"
#include "pcas/physical_mem.hpp"

namespace pcas {

template <uint64_t BlockSize>
class cache_system {
  static_assert(BlockSize % min_block_size == 0);

  struct entry {
    bool     cached         = false;
    bool     fetched        = false;
    int      checkout_count = 0;
    uint64_t pm_offset;
  };

public:
  using entry_t = entry*;

  static constexpr uint64_t block_size = BlockSize;

private:
  physical_mem        pm_;
  uint64_t            size_;
  uint64_t            nblocks_;
  std::list<entry*>   entries_; // TODO: needed?
  std::vector<entry*> cache_map_;

  void evict(uint64_t block_num) {
    PCAS_CHECK(block_num < nblocks_);
    entry* e = cache_map_[block_num];
    PCAS_CHECK(e);
    PCAS_CHECK(e->checkout_count == 0);
    e->cached = false;
    e->fetched = false;
    cache_map_[block_num] = nullptr;
  }

  uint64_t evict_one() {
    // TODO: implement more smart eviction (e.g., LRU)
    // randomly select a victim first
    uint64_t max_trial = nblocks_ * 10;
    static std::mt19937 engine(0);
    std::uniform_int_distribution<> dist(0, nblocks_ - 1);
    for (uint64_t i = 0; i < max_trial; i++) {
      uint64_t b = dist(engine);
      entry* e = cache_map_[b];
      if (e && e->checkout_count == 0) {
        evict(b);
        return b;
      }
    }
    // check sequentially
    for (uint64_t b = 0; b < nblocks_; b++) {
      entry* e = cache_map_[b];
      if (e && e->checkout_count == 0) {
        evict(b);
        return b;
      }
    }
    die("cache is exhausted (too many objects are being checked out)");
    return 0;
  }

  uint64_t get_empty_block() {
    for (uint64_t b = 0; b < nblocks_; b++) {
      if (cache_map_[b] == nullptr) {
        return b;
      }
    }
    return evict_one();
  }

public:
  cache_system(uint64_t size) : size_(size) {
    PCAS_CHECK(size % BlockSize == 0);
    nblocks_ = size / BlockSize;
    pm_ = physical_mem(size);
    cache_map_ = std::vector<entry*>(nblocks_, nullptr);
  }

  ~cache_system() {
    PCAS_CHECK(entries_.size() == 0);
    for (auto e : cache_map_) {
      PCAS_CHECK(e == nullptr);
    }
  }

  physical_mem& pm() { return pm_; }

  entry_t alloc_entry() {
    entry* e = new entry();
    entries_.push_back(e);
    return e;
  }

  void free_entry(entry_t e) {
    PCAS_CHECK(e);
    if (e->cached) {
      PCAS_CHECK(e->checkout_count == 0);
      uint64_t b = e->pm_offset / BlockSize;
      PCAS_CHECK(b < nblocks_);
      cache_map_[b] = nullptr;
    }
    entries_.remove(e); // TODO: heavy?
    delete e;
  }

  bool checkout(entry_t e) {
    PCAS_CHECK(e);
    e->checkout_count++;
    if (e->cached) {
      return true;
    } else {
      int block_num = get_empty_block();
      e->pm_offset = block_num * BlockSize;
      e->cached = true;
      cache_map_[block_num] = e;
      return false;
    }
  }

  void checkin(entry_t e) {
    PCAS_CHECK(e);
    PCAS_CHECK(e->checkout_count > 0);
    PCAS_CHECK(e->cached);
    e->checkout_count--;
  }

  bool needs_fetch(entry_t e, bool wants_fetch) {
    PCAS_CHECK(e);
    PCAS_CHECK(e->cached);
    if (wants_fetch && !e->fetched) {
      e->fetched = true;
      return true;
    } else {
      return false;
    }
  }

  void already_fetched(entry_t e) {
    PCAS_CHECK(e);
    PCAS_CHECK(e->cached);
    e->fetched = true;
  }

  void evict_all() {
    for (uint64_t b = 0; b < nblocks_; b++) {
      entry* e = cache_map_[b];
      if (e) {
        evict(b);
      }
    }
  }

};

PCAS_TEST_CASE("[pcas::cache] testing cache system") {
  int nblk = 100;
  using cache_t = cache_system<min_block_size>;
  cache_t cs(nblk * min_block_size);

  std::vector<cache_t::entry_t> cache_entries;
  int nent = 1000;
  for (int i = 0; i < nent; i++) {
    cache_entries.push_back(cs.alloc_entry());
  }

  PCAS_SUBCASE("basic test") {
    for (int i = 0; i < nent; i++) {
      bool hit = cs.checkout(cache_entries[i]);
      PCAS_CHECK_MESSAGE(!hit, "should not be cached at the beginning");
      cs.checkin(cache_entries[i]);
    }

    cs.evict_all();

    for (int i = 0; i < nblk; i++) {
      bool hit = cs.checkout(cache_entries[i]);
      PCAS_CHECK_MESSAGE(!hit, "should not be cached after evicting all cache");
      cs.checkin(cache_entries[i]);
    }

    for (int it = 0; it < 3; it++) {
      for (int i = 0; i < nblk; i++) {
        bool hit = cs.checkout(cache_entries[i]);
        PCAS_CHECK_MESSAGE(hit, "should be cached when the working set fits into the cache");
        cs.checkin(cache_entries[i]);
      }
    }
  }

  PCAS_SUBCASE("cache entry being checking out should not be evicted") {
    cs.checkout(cache_entries[0]);

    for (int it = 0; it < 3; it++) {
      for (int i = 1; i < nent; i++) {
        cs.checkout(cache_entries[i]);
        PCAS_CHECK(cache_entries[i]->pm_offset != cache_entries[0]->pm_offset);
        cs.checkin(cache_entries[i]);
      }
    }

    cs.checkin(cache_entries[0]);
  }

  PCAS_SUBCASE("a block can be checkout out for many times") {
    cs.checkout(cache_entries[0]);
    PCAS_CHECK(cache_entries[0]->checkout_count == 1);
    cs.checkout(cache_entries[0]);
    PCAS_CHECK(cache_entries[0]->checkout_count == 2);

    cs.checkin(cache_entries[0]);
    PCAS_CHECK(cache_entries[0]->checkout_count == 1);
    cs.checkin(cache_entries[0]);
    PCAS_CHECK(cache_entries[0]->checkout_count == 0);
  }

  cs.evict_all();

  for (auto e : cache_entries) {
    cs.free_entry(e);
  }
}

}
