#pragma once

#include <cstdint>
#include <unistd.h>

#include <pcas/util.hpp>

namespace pcas {

using checkout_id_t = uint64_t;

// sort of unique_ptr
template <typename T>
class local_ptr {
  using this_t = local_ptr<T>;

  checkout_id_t cid_;
  T*            raw_ptr_;

public:
  local_ptr() : cid_(0), raw_ptr_(nullptr) {}
  local_ptr(checkout_id_t cid, T* raw_ptr) : cid_(cid), raw_ptr_(raw_ptr) {}

  local_ptr(const this_t&) = delete;
  this_t& operator=(const this_t&) = delete;

  local_ptr(this_t&& p) : cid_(p.cid_), raw_ptr_(p.raw_ptr_) {
    p.cid_ = 0;
    p.raw_ptr_ = nullptr;
  }
  this_t& operator=(this_t&& p) {
    cid_ = p.cid_;
    raw_ptr_ = p.raw_ptr_;
    p.cid_ = 0;
    p.raw_ptr_ = nullptr;
    return *this;
  }

  T* get() const noexcept { return raw_ptr_; }
  checkout_id_t cid() const noexcept { return cid_; }

  operator bool() const noexcept { return raw_ptr_; }
  bool operator!() const noexcept { return !raw_ptr_; }

  T& operator*() const { return *raw_ptr_; }
  T& operator[](std::size_t i) const { return raw_ptr_[i]; }
};

template <typename T, typename U>
inline bool operator==(const local_ptr<T>& p1, const local_ptr<U>& p2) noexcept {
  return p1.cid() == p2.cid();
}

template <typename T, typename U>
inline bool operator!=(const local_ptr<T>& p1, const local_ptr<U>& p2) noexcept {
  return p1.cid() != p2.cid();
}

PCAS_TEST_CASE("[pcas::local_ptr] basic test") {
  local_ptr<int> p1;
  PCAS_CHECK(!p1);

  int a1 = 32;
  local_ptr<int> p2(1, &a1);
  PCAS_CHECK(p2);
  PCAS_CHECK(*p2 == a1);
  p1 = std::move(p2);
  PCAS_CHECK(!p2);
  PCAS_CHECK(*p1 == a1);
  PCAS_CHECK(p1.cid() == 1);

  int a2[3] = {1, 2, 3};
  local_ptr<int> p3(2, a2);
  PCAS_CHECK(p3);
  PCAS_CHECK(p2 != p3);
  PCAS_CHECK(p3 == p3);
  PCAS_CHECK(p3[2] == 3);
  PCAS_CHECK(p3.get() == a2);
  PCAS_CHECK(p3.get()[2] == 3);
}

}
