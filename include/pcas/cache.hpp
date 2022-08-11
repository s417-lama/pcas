#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <vector>
#include <list>
#include <random>
#include <limits>

#include "pcas/util.hpp"
#include "pcas/physical_mem.hpp"
#include "pcas/virtual_mem.hpp"

namespace pcas {

template <uint64_t BlockSize>
class cache_system {
  static_assert(BlockSize % min_block_size == 0);

public:
  using block_num_t = uint64_t;

  struct entry {
    bool        cached         = false;
    bool        fetched        = false;
    bool        flushing       = false;
    int         checkout_count = 0;
    block_num_t block_num      = std::numeric_limits<block_num_t>::max();
    uint8_t*    vm_addr        = nullptr;
    obj_id_t    obj_id;
    sections    dirty_sections;
    typename std::list<entry*>::iterator lru_it;
  };

  using entry_t = entry*;

  // FIXME: not used inside this class
  static constexpr uint64_t block_size = BlockSize;

private:
  physical_mem         pm_;
  uint64_t             size_;
  block_num_t          nblocks_;
  std::vector<entry_t> cache_map_;
  std::list<entry_t>   lru_;

  bool is_evictable(entry_t e) {
    return e && e->cached && e->checkout_count == 0 && !e->flushing && e->dirty_sections.empty();
  }

  void invalidate(block_num_t b) {
    PCAS_CHECK(b < nblocks_);
    entry* e = cache_map_[b];
    PCAS_CHECK(is_evictable(e));
    e->cached = false;
    e->fetched = false;
  }

  block_num_t evict_one() {
    if (lru_.empty()) {
      die("cache is exhausted (too many objects are being checked out)");
    }
    entry_t e = lru_.back();
    lru_.pop_back();
    e->lru_it = lru_.end();
    invalidate(e->block_num);
    return e->block_num;
  }

  block_num_t get_empty_block() {
    for (block_num_t b = 0; b < nblocks_; b++) {
      entry_t e = cache_map_[b];
      if (!e || !e->cached) {
        return b;
      }
    }
    return evict_one();
  }

public:
  cache_system(uint64_t size, int intra_rank) : size_(size) {
    PCAS_CHECK(size % BlockSize == 0);
    nblocks_ = size / BlockSize;
    pm_ = physical_mem(size, 0, intra_rank, true, true);
    cache_map_ = std::vector<entry*>(nblocks_, nullptr);
  }

  ~cache_system() {
    for (auto e : cache_map_) {
      PCAS_CHECK(e == nullptr);
    }
  }

  physical_mem& pm() { return pm_; }

  entry_t alloc_entry(obj_id_t obj_id) {
    entry* e = new entry();
    e->obj_id = obj_id;
    e->lru_it = lru_.end();
    return e;
  }

  void free_entry(entry_t e) {
    PCAS_CHECK(e);
    PCAS_CHECK(e->dirty_sections.empty());
    block_num_t b = e->block_num;
    if (b < nblocks_ && cache_map_[b] == e) {
      PCAS_CHECK(e->checkout_count == 0);
      virtual_mem::unmap(e->vm_addr, BlockSize);
      cache_map_[b] = nullptr;
    }
    if (e->lru_it != lru_.end()) {
      lru_.erase(e->lru_it);
    }
    delete e;
  }

  // return (hit, prev_entry)
  std::tuple<bool, entry_t> checkout(entry_t e) {
    PCAS_CHECK(e);
    e->checkout_count++;
    if (e->cached) {
      // cache hit
      return std::make_tuple(true, nullptr);
    } else if (e->block_num < nblocks_ && cache_map_[e->block_num] == e) {
      // the entry has been invalidated but remains in the cache
      PCAS_CHECK(e->dirty_sections.empty());
      e->cached = true;
      if (e->lru_it != lru_.end()) {
        lru_.erase(e->lru_it);
        e->lru_it = lru_.end();
      }
      return std::make_tuple(false, nullptr);
    } else {
      // the entry needs a new cache block
      block_num_t b = get_empty_block();
      e->block_num = b;
      e->cached = true;
      entry_t prev_e = cache_map_[b];
      cache_map_[b] = e;
      return std::make_tuple(false, prev_e);
    }
  }

  void checkin(entry_t e) {
    PCAS_CHECK(e);
    PCAS_CHECK(e->checkout_count > 0);
    PCAS_CHECK(e->cached);
    e->checkout_count--;
    if (is_evictable(e)) {
      if (e->lru_it != lru_.end()) {
        lru_.erase(e->lru_it);
        e->lru_it = lru_.end();
      }
      lru_.push_front(e);
      e->lru_it = lru_.begin();
    }
  }

  void invalidate_all() {
    for (block_num_t b = 0; b < nblocks_; b++) {
      entry* e = cache_map_[b];
      if (e && e->cached) {
        invalidate(b);
      }
    }
  }

  template <typename Func>
  void for_each_block(Func f) {
    for (block_num_t b = 0; b < nblocks_; b++) {
      entry* e = cache_map_[b];
      f(e);
    }
  }

};

PCAS_TEST_CASE("[pcas::cache] testing cache system") {
  int nblk = 100;
  using cache_t = cache_system<min_block_size>;
  cache_t cs(nblk * min_block_size, -1);

  std::vector<cache_t::entry_t> cache_entries;
  int nent = 1000;
  for (int i = 0; i < nent; i++) {
    cache_entries.push_back(cs.alloc_entry(0));
  }

  PCAS_SUBCASE("basic test") {
    for (int i = 0; i < nent; i++) {
      auto [hit, _prev_e] = cs.checkout(cache_entries[i]);
      PCAS_CHECK_MESSAGE(!hit, "should not be cached at the beginning");
      cs.checkin(cache_entries[i]);
    }

    cs.invalidate_all();

    for (int i = 0; i < nblk; i++) {
      auto [hit, _prev_e] = cs.checkout(cache_entries[i]);
      PCAS_CHECK_MESSAGE(!hit, "should not be cached after evicting all cache");
      cs.checkin(cache_entries[i]);
    }

    for (int it = 0; it < 3; it++) {
      for (int i = 0; i < nblk; i++) {
        auto [hit, _prev_e] = cs.checkout(cache_entries[i]);
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
        PCAS_CHECK(cache_entries[i]->block_num != cache_entries[0]->block_num);
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

  cs.invalidate_all();

  for (auto e : cache_entries) {
    cs.free_entry(e);
  }
}

}
