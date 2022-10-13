#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <vector>
#include <list>
#include <limits>
#include <unordered_map>
#include <iterator>

#include "pcas/util.hpp"

namespace pcas {

using cache_entry_num_t = uint64_t;

class cache_full_exception : public std::exception {};

template <typename Key, typename Entry>
class cache_system {
  struct cache_entry {
    bool                                            allocated;
    Key                                             key;
    Entry                                           entry;
    cache_entry_num_t                               entry_num = std::numeric_limits<cache_entry_num_t>::max();
    typename std::list<cache_entry_num_t>::iterator lru_it;

    template <typename... Args>
    cache_entry(Args&&... args) : entry(std::forward<Args>(args)...) {}
  };

  cache_entry_num_t                          nentries_;
  std::vector<cache_entry>                   entries_; // cache_entry_num_t -> cache_entry
  std::unordered_map<Key, cache_entry_num_t> table_; // hash table (Key -> cache_entry_num_t)
  std::list<cache_entry_num_t>               lru_; // front (oldest) <----> back (newest)

  void move_to_back_lru(cache_entry& cb) {
    lru_.splice(lru_.end(), lru_, cb.lru_it);
    PCAS_CHECK(std::prev(lru_.end()) == cb.lru_it);
    PCAS_CHECK(*cb.lru_it == cb.entry_num);
  }

  cache_entry_num_t get_empty_slot() {
    // FIXME: Performance issue?
    for (auto it = lru_.begin(); it != lru_.end(); it++) {
      cache_entry_num_t b = *it;
      cache_entry& cb = entries_[b];
      if (!cb.allocated) {
        return cb.entry_num;
      }
      if (cb.entry.is_evictable()) {
        Key prev_key = cb.key;
        table_.erase(prev_key);
        cb.entry.on_evict();
        cb.allocated = false;
        return cb.entry_num;
      }
    }
    throw cache_full_exception{};
  }

public:
  template <typename... Args>
  cache_system(cache_entry_num_t nentries, Args&&... args)
    : nentries_(nentries) {
    table_.reserve(nentries_);
    for (cache_entry_num_t b = 0; b < nentries_; b++) {
      cache_entry& cb = entries_.emplace_back(std::forward<Args>(args)...);
      cb.allocated = false;
      cb.entry_num = b;
      lru_.push_front(b);
      cb.lru_it = lru_.begin();
      PCAS_CHECK(*cb.lru_it == b);
    }
  }

  ~cache_system() {
    PCAS_CHECK(table_.empty());
    for (auto& cb : entries_) {
      PCAS_CHECK(!cb.allocated);
    }
  }

  cache_entry_num_t num_entries() { return nentries_; }

  bool is_cached(Key key) {
    return table_.find(key) != table_.end();
  }

  Entry& ensure_cached(Key key) {
    auto it = table_.find(key);
    if (it == table_.end()) {
      cache_entry_num_t b = get_empty_slot();
      cache_entry& cb = entries_[b];

      cb.entry.on_cache_map(b);

      cb.allocated = true;
      cb.key = key;
      table_[key] = b;
      move_to_back_lru(cb);
      return cb.entry;
    } else {
      cache_entry_num_t b = it->second;
      cache_entry& cb = entries_[b];
      move_to_back_lru(cb);
      return cb.entry;
    }
  }

  void ensure_evicted(Key key) {
    auto it = table_.find(key);
    if (it != table_.end()) {
      cache_entry_num_t b = it->second;
      cache_entry& cb = entries_[b];
      PCAS_CHECK(cb.entry.is_evictable());
      cb.entry.on_evict();
      table_.erase(key);
      cb.allocated = false;
    }
  }

  template <typename Func>
  void for_each_entry(Func&& f) {
    for (auto& cb : entries_) {
      if (cb.allocated) {
        f(cb.entry);
      }
    }
  }

};

PCAS_TEST_CASE("[pcas::cache] testing cache system") {
  using key_t = int;
  struct test_entry {
    bool              evictable = true;
    cache_entry_num_t entry_num = std::numeric_limits<cache_entry_num_t>::max();

    bool is_evictable() const { return evictable; }
    void on_cache_map(cache_entry_num_t b) { entry_num = b; }
    void on_evict() {}
  };

  int nblk = 100;
  cache_system<key_t, test_entry> cs(nblk);

  int nkey = 1000;
  std::vector<key_t> keys;
  for (int i = 0; i < nkey; i++) {
    keys.push_back(i);
  }

  PCAS_SUBCASE("basic test") {
    for (key_t k : keys) {
      test_entry& e = cs.ensure_cached(k);
      PCAS_CHECK(cs.is_cached(k));
      for (int i = 0; i < 10; i++) {
        test_entry& e2 = cs.ensure_cached(k);
        PCAS_CHECK(e.entry_num == e2.entry_num);
      }
    }
  }

  PCAS_SUBCASE("all entries should be cached when the number of entries is small enough") {
    for (int i = 0; i < nblk; i++) {
      cs.ensure_cached(keys[i]);
      PCAS_CHECK(cs.is_cached(keys[i]));
    }
    for (int i = 0; i < nblk; i++) {
      cs.ensure_cached(keys[i]);
      PCAS_CHECK(cs.is_cached(keys[i]));
      for (int j = 0; j < nblk; j++) {
        PCAS_CHECK(cs.is_cached(keys[j]));
      }
    }
  }

  PCAS_SUBCASE("nonevictable entries should not be evicted") {
    int nrem = 50;
    for (int i = 0; i < nrem; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      PCAS_CHECK(cs.is_cached(keys[i]));
      e.evictable = false;
    }
    for (key_t k : keys) {
      cs.ensure_cached(k);
      PCAS_CHECK(cs.is_cached(k));
      for (int j = 0; j < nrem; j++) {
        PCAS_CHECK(cs.is_cached(keys[j]));
      }
    }
    for (int i = 0; i < nrem; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      PCAS_CHECK(cs.is_cached(keys[i]));
      e.evictable = true;
    }
  }

  PCAS_SUBCASE("should throw exception if cache is full") {
    for (int i = 0; i < nblk; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      PCAS_CHECK(cs.is_cached(keys[i]));
      e.evictable = false;
    }
    PCAS_CHECK_THROWS_AS(cs.ensure_cached(keys[nblk]), cache_full_exception);
    for (int i = 0; i < nblk; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      PCAS_CHECK(cs.is_cached(keys[i]));
      e.evictable = true;
    }
    cs.ensure_cached(keys[nblk]);
    PCAS_CHECK(cs.is_cached(keys[nblk]));
  }

  PCAS_SUBCASE("LRU eviction") {
    for (int i = 0; i < nkey; i++) {
      cs.ensure_cached(keys[i]);
      PCAS_CHECK(cs.is_cached(keys[i]));
      for (int j = 0; j <= i - nblk; j++) {
        PCAS_CHECK(!cs.is_cached(keys[j]));
      }
      for (int j = std::max(0, i - nblk + 1); j < i; j++) {
        PCAS_CHECK(cs.is_cached(keys[j]));
      }
    }
  }

  for (key_t k : keys) {
    cs.ensure_evicted(k);
  }
}

}
