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

namespace pcas {

using cache_block_num_t = uint64_t;

class cache_full_exception : public std::exception {};

template <typename Entry>
class cache_system {
public:
  struct cache_block {
    Entry                                           entry     = nullptr;
    cache_block_num_t                               block_num = std::numeric_limits<cache_block_num_t>::max();
    typename std::list<cache_block_num_t>::iterator lru_it;
  };

private:
  cache_block_num_t            nblocks_;
  std::vector<cache_block>     blocks_; // cache_block_num_t -> cache_block
  std::list<cache_block_num_t> lru_; // front (oldest) <----> back (newest)

  cache_block& entry2block(Entry e) {
    PCAS_CHECK(e->is_cached());
    cache_block_num_t b = e->get_cache_block_num();
    PCAS_CHECK(b < nblocks_);
    PCAS_CHECK(blocks_[b].entry == e);
    return blocks_[b];
  }

  void move_to_back_lru(cache_block& cb) {
    lru_.erase(cb.lru_it);
    lru_.push_back(cb.block_num);
    cb.lru_it = --lru_.end();
    PCAS_CHECK(*cb.lru_it == cb.block_num);
  }

  cache_block& get_empty_block() {
    // FIXME: Performance issue?
    for (auto it = lru_.begin(); it != lru_.end(); it++) {
      cache_block_num_t b = *it;
      cache_block& cb = blocks_[b];
      if (!cb.entry) {
        return cb;
      } else if (cb.entry->is_evictable()) {
        cb.entry->on_evict();
        return cb;
      }
    }
    throw cache_full_exception{};
  }

public:
  cache_system(cache_block_num_t nblocks) : nblocks_(nblocks) {
    for (cache_block_num_t b = 0; b < nblocks; b++) {
      cache_block cb;
      cb.block_num = b;
      lru_.push_front(b);
      cb.lru_it = lru_.begin();
      PCAS_CHECK(*cb.lru_it == b);
      blocks_.push_back(std::move(cb));
    }
  }

  ~cache_system() {
    for (auto& cb : blocks_) {
      PCAS_CHECK(cb.entry == nullptr);
    }
  }

  cache_block_num_t num_blocks() { return nblocks_; }

  void ensure_cached(Entry e) {
    if (!e->is_cached()) {
      cache_block& cb = get_empty_block();
      move_to_back_lru(cb);
      cache_block_num_t b = cb.block_num;
      Entry prev_e = cb.entry;
      e->on_cache_remap(b, prev_e);
      cb.entry = e;
    }
    PCAS_CHECK(e->is_cached());
    use(e);
  }

  void ensure_evicted(Entry e) {
    PCAS_CHECK(e);
    PCAS_CHECK(e->is_evictable());
    if (e->is_cached()) {
      cache_block& cb = entry2block(e);
      cb.entry = nullptr;
    }
  }

  void use(Entry e) {
    PCAS_CHECK(e->is_cached());
    cache_block& cb = entry2block(e);
    move_to_back_lru(cb);
  }

  template <typename Func>
  void for_each_block(Func&& f) {
    for (auto& cb : blocks_) {
      if (cb.entry) {
        f(*cb.entry);
      }
    }
  }

};

PCAS_TEST_CASE("[pcas::cache] testing cache system") {
  struct test_entry {
    bool              cached    = false;
    bool              evictable = true;
    cache_block_num_t block_num = std::numeric_limits<cache_block_num_t>::max();

    bool is_evictable() const { return evictable; }
    bool is_cached() const { return cached; }
    cache_block_num_t get_cache_block_num() const { return block_num; }
    void on_cache_remap(cache_block_num_t b, test_entry* prev_block [[maybe_unused]]) {
      block_num = b;
      cached = true;
    }
    void on_evict() { cached = false; }
  };

  int nblk = 100;
  cache_system<test_entry*> cs(nblk);

  int nent = 1000;
  std::vector<test_entry> entries(nent);

  PCAS_SUBCASE("basic test") {
    for (int i = 0; i < nent; i++) {
      cs.ensure_cached(&entries[i]);
      PCAS_CHECK(entries[i].cached);
      auto b = entries[i].block_num;
      for (int j = 0; j < 10; j++) {
        cs.ensure_cached(&entries[i]);
        PCAS_CHECK(entries[i].block_num == b);
      }
    }
  }

  PCAS_SUBCASE("all entries should be cached when the number of entries is small enough") {
    for (int i = 0; i < nblk; i++) {
      cs.ensure_cached(&entries[i]);
      PCAS_CHECK(entries[i].cached);
    }
    for (int i = 0; i < nblk; i++) {
      cs.ensure_cached(&entries[i]);
      for (int j = 0; j < nblk; j++) {
        PCAS_CHECK(entries[j].cached);
      }
    }
  }

  PCAS_SUBCASE("nonevictable entries should not be evicted") {
    int nrem = 50;
    for (int i = 0; i < nrem; i++) {
      entries[i].evictable = false;
      cs.ensure_cached(&entries[i]);
      PCAS_CHECK(entries[i].cached);
    }
    for (int i = 0; i < nent; i++) {
      cs.ensure_cached(&entries[i]);
      PCAS_CHECK(entries[i].cached);
      for (int j = 0; j < nrem; j++) {
        PCAS_CHECK(entries[j].cached);
      }
    }
    for (int i = 0; i < nrem; i++) {
      entries[i].evictable = true;
    }
  }

  PCAS_SUBCASE("should throw exception if cache is full") {
    for (int i = 0; i < nblk; i++) {
      entries[i].evictable = false;
      cs.ensure_cached(&entries[i]);
      PCAS_CHECK(entries[i].cached);
    }
    PCAS_CHECK_THROWS_AS(cs.ensure_cached(&entries[nblk]), cache_full_exception);
    for (int i = 0; i < nblk; i++) {
      entries[i].evictable = true;
    }
    cs.ensure_cached(&entries[nblk]);
    PCAS_CHECK(entries[nblk].cached);
  }

  PCAS_SUBCASE("LRU eviction") {
    for (int i = 0; i < nent; i++) {
      cs.ensure_cached(&entries[i]);
      PCAS_CHECK(entries[i].cached);
      for (int j = 0; j <= i - nblk; j++) {
        PCAS_CHECK(!entries[j].cached);
      }
      for (int j = std::max(0, i - nblk + 1); j < i; j++) {
        PCAS_CHECK(entries[j].cached);
      }
    }
  }

  for (auto& e : entries) {
    cs.ensure_evicted(&e);
  }
}

}
