#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <vector>
#include <list>
#include <random>
#include <limits>

#include <mpi.h>

#include "pcas/util.hpp"
#include "pcas/physical_mem.hpp"
#include "pcas/virtual_mem.hpp"

namespace pcas {

class cache_full_exception : public std::exception {};

template <uint64_t BlockSize>
class cache_system {
  static_assert(BlockSize % min_block_size == 0);

public:
  using block_num_t = uint64_t;

  enum class state_t {
    evicted,  // this entry is not in the cache
    invalid,  // this entry is in the cache, but the data is not up-to-date
    fetching, // communication (read) is in-progress
    valid,    // the data is up-to-date
  };

  struct entry {
    state_t     state          = state_t::evicted;
    bool        flushing       = false;
    int         checkout_count = 0;
    block_num_t block_num      = std::numeric_limits<block_num_t>::max();
    uint8_t*    vm_addr        = nullptr;
    entry*      prev_entry     = nullptr;
    MPI_Request req            = MPI_REQUEST_NULL;
    obj_id_t    obj_id;
    sections    dirty_sections;
    sections    partial_sections; // for write-only update
    typename std::list<entry*>::iterator lru_it;
  };

  using entry_t = entry*;

  // FIXME: not used inside this class
  static constexpr uint64_t block_size = BlockSize;

  bool is_evictable(entry_t e) {
    return e && e->checkout_count == 0 && !e->flushing && e->dirty_sections.empty();
  }

  void set_evictable(entry_t e) {
    PCAS_CHECK(is_evictable(e));
    PCAS_CHECK(e->lru_it == lru_.end());
    lru_.push_front(e);
    e->lru_it = lru_.begin();
  }

  void unset_evictable(entry_t e) {
    if (e->lru_it != lru_.end()) {
      lru_.erase(e->lru_it);
      e->lru_it = lru_.end();
    }
  }

private:
  physical_mem         pm_;
  uint64_t             size_;
  block_num_t          nblocks_;
  std::vector<entry_t> cache_map_;
  std::list<entry_t>   lru_; // contains only evictable entries

  void invalidate(block_num_t b) {
    PCAS_CHECK(b < nblocks_);
    entry* e = cache_map_[b];
    PCAS_CHECK(is_evictable(e));
    e->state = state_t::invalid;
    e->partial_sections.clear();
  }

  block_num_t evict_one() {
    if (lru_.empty()) {
      throw cache_full_exception{};
    }
    entry_t e = lru_.back();
    lru_.pop_back();
    e->lru_it = lru_.end();
    if (e->state == state_t::fetching) {
      PCAS_CHECK(e->req != MPI_REQUEST_NULL);
      // FIXME: MPI_Cancel causes segfault
      /* MPI_Cancel(&e->req); */
      /* MPI_Request_free(&e->req); */
      MPI_Wait(&e->req, MPI_STATUS_IGNORE);
      PCAS_CHECK(e->req == MPI_REQUEST_NULL);
    }
    invalidate(e->block_num);
    e->state = state_t::evicted;
    return e->block_num;
  }

  block_num_t get_empty_block() {
    // FIXME: inefficient; merge with LRU data structure
    for (block_num_t b = 0; b < nblocks_; b++) {
      entry_t e = cache_map_[b];
      if (!e) {
        return b;
      }
      if (e->state == state_t::invalid) {
        e->state = state_t::evicted;
        unset_evictable(e);
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
    PCAS_CHECK(e->checkout_count == 0);
    PCAS_CHECK(e->dirty_sections.empty());
    block_num_t b = e->block_num;
    if (b < nblocks_ && cache_map_[b] == e) {
      virtual_mem::unmap(e->vm_addr, BlockSize);
      cache_map_[b] = nullptr;
    }
    unset_evictable(e);
    delete e;
  }

  bool checkout(entry_t e, bool prefetch = false) {
    PCAS_CHECK(e);
    bool hit;
    switch (e->state) {
      case state_t::evicted: {
        // the entry needs a new cache block
        block_num_t b = get_empty_block();
        e->block_num = b;
        e->prev_entry = cache_map_[b];
        cache_map_[b] = e;
        PCAS_CHECK(e->lru_it == lru_.end());
        hit = false;
        break;
      }
      case state_t::invalid: {
        // the entry has been invalidated but remains in the cache
        PCAS_CHECK(e->block_num < nblocks_);
        PCAS_CHECK(cache_map_[e->block_num] == e);
        PCAS_CHECK(e->dirty_sections.empty());
        unset_evictable(e);
        hit = false;
        break;
      }
      default: {
        // cache hit
        unset_evictable(e);
        hit = true;
        break;
      }
    }
    if (!prefetch) {
      e->checkout_count++;
    }
    if (prefetch && is_evictable(e)) {
      set_evictable(e);
    }
    return hit;
  }

  void checkin(entry_t e) {
    PCAS_CHECK(e);
    PCAS_CHECK(e->checkout_count > 0);
    PCAS_CHECK(e->state == state_t::valid);
    e->checkout_count--;
    if (is_evictable(e)) {
      set_evictable(e);
    }
  }

  void invalidate_all() {
    for (block_num_t b = 0; b < nblocks_; b++) {
      entry* e = cache_map_[b];
      if (e) {
        if (e->state == state_t::valid) {
          invalidate(b);
        } else if (e->state == state_t::fetching) {
          PCAS_CHECK(e->req != MPI_REQUEST_NULL);
          // FIXME: MPI_Cancel causes segfault
          /* MPI_Cancel(&e->req); */
          /* MPI_Request_free(&e->req); */
          MPI_Wait(&e->req, MPI_STATUS_IGNORE);
          PCAS_CHECK(e->req == MPI_REQUEST_NULL);
          invalidate(b);
        }
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
      auto hit = cs.checkout(cache_entries[i]);
      cache_entries[i]->state = cache_t::state_t::valid;
      PCAS_CHECK_MESSAGE(!hit, "should not be cached at the beginning");
      cs.checkin(cache_entries[i]);
    }

    cs.invalidate_all();

    for (int i = 0; i < nblk; i++) {
      auto hit = cs.checkout(cache_entries[i]);
      cache_entries[i]->state = cache_t::state_t::valid;
      PCAS_CHECK_MESSAGE(!hit, "should not be cached after evicting all cache");
      cs.checkin(cache_entries[i]);
    }

    for (int it = 0; it < 3; it++) {
      for (int i = 0; i < nblk; i++) {
        auto hit = cs.checkout(cache_entries[i]);
        cache_entries[i]->state = cache_t::state_t::valid;
        PCAS_CHECK_MESSAGE(hit, "should be cached when the working set fits into the cache");
        cs.checkin(cache_entries[i]);
      }
    }
  }

  PCAS_SUBCASE("cache entry being checking out should not be evicted") {
    cs.checkout(cache_entries[0]);
    cache_entries[0]->state = cache_t::state_t::valid;

    for (int it = 0; it < 3; it++) {
      for (int i = 1; i < nent; i++) {
        cs.checkout(cache_entries[i]);
        cache_entries[i]->state = cache_t::state_t::valid;
        PCAS_CHECK(cache_entries[i]->block_num != cache_entries[0]->block_num);
        cs.checkin(cache_entries[i]);
      }
    }

    cs.checkin(cache_entries[0]);
  }

  PCAS_SUBCASE("a block can be checkout out for many times") {
    cs.checkout(cache_entries[0]);
    cache_entries[0]->state = cache_t::state_t::valid;
    PCAS_CHECK(cache_entries[0]->checkout_count == 1);
    cs.checkout(cache_entries[0]);
    cache_entries[0]->state = cache_t::state_t::valid;
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
