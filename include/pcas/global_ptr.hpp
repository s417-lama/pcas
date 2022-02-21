#pragma once

#include <cstdint>
#include <unistd.h>

#include "doctest/doctest.h"

namespace pcas {

using obj_id_t = uint64_t;

template <typename T>
class global_ptr {
  using this_t = global_ptr<T>;

  int      owner_;
  obj_id_t id_;
  uint64_t offset_;

  class global_ptr_deref {
    global_ptr ptr_;
  public:
    global_ptr_deref(const global_ptr& p) : ptr_(p) {}
    this_t operator&() const noexcept { return this_t(ptr_); }
  };

public:
  global_ptr() : owner_(-2), id_(0), offset_(0) {}
  global_ptr(int owner, obj_id_t id, uint64_t offset) : owner_(owner), id_(id), offset_(offset) {}
  global_ptr(const this_t& p) {
    owner_  = p.owner_;
    id_     = p.id_;
    offset_ = p.offset_;
  }

  int owner() const noexcept { return owner_; }
  obj_id_t id() const noexcept { return id_; }
  uint64_t offset() const noexcept { return offset_; }

  template<class U>
  bool is_equal(const global_ptr<U>& p) const noexcept {
    return owner_  == p.owner() &&
           id_     == p.id()    &&
           offset_ == p.offset();
  }

  template<class U>
  bool belong_to_same_obj(const global_ptr<U>& p) const noexcept {
    return owner_ == p.owner() &&
           id_    == p.id();
  }

  operator bool() const noexcept { return !is_equal(this_t()); }
  bool operator!() const noexcept { return is_equal(this_t()); }

  template <class Diff>
  global_ptr_deref operator[](Diff diff) const noexcept {
    return global_ptr(owner_, id_, offset_ + diff * sizeof(T));
  }

  template <class Diff>
  this_t& operator+=(Diff diff) {
    offset_ += diff * sizeof(T);
    return *this;
  }

  template <class Diff>
  this_t& operator-=(Diff diff) {
    CHECK(offset_ >= diff * sizeof(T));
    offset_ -= diff * sizeof(T);
    return *this;
  }

  this_t& operator++() { return (*this) += 1; }
  this_t& operator--() { return (*this) -= 1; }

  this_t operator++(int) { this_t tmp(*this); ++(*this); return tmp; }
  this_t operator--(int) { this_t tmp(*this); --(*this); return tmp; }

  template <class Diff>
  this_t operator+(Diff diff) const noexcept {
    return global_ptr(owner_, id_, offset_ + diff * sizeof(T));
  }

  template <class Diff>
  this_t operator-(Diff diff) const noexcept {
    CHECK(offset_ >= diff * sizeof(T));
    return global_ptr(owner_, id_, offset_ - diff * sizeof(T));
  }

  std::ptrdiff_t operator-(const this_t& p) const noexcept {
    CHECK(belong_to_same_obj(p));
    CHECK(offset_ % sizeof(T) == 0);
    CHECK(p.offset_ % sizeof(T) == 0);
    return offset_ / sizeof(T) - p.offset_ / sizeof(T);
  }

  template <typename U>
  explicit operator global_ptr<U>() const noexcept {
    return global_ptr<U>(owner_, id_, offset_);
  }
};

template <typename T, typename U>
inline bool operator==(const global_ptr<T>& p1, const global_ptr<U>& p2) noexcept {
  return p1.is_equal(p2);
}

template <typename T, typename U>
inline bool operator!=(const global_ptr<T>& p1, const global_ptr<U>& p2) noexcept {
  return !p1.is_equal(p2);
}

TEST_CASE("global pointer manipulation") {
  global_ptr<int> p1(0, 0, 0);
  global_ptr<int> p2(1, 0, 0);
  global_ptr<int> p3(1, 1, 0);

  SUBCASE("initialization") {
    global_ptr<int> p1_(p1);
    global_ptr<int> p2_(p1.owner(), p1.id(), p1.offset());
    CHECK(p1_ == p2_);
  }

  SUBCASE("addition and subtraction") {
    auto p = p1 + 4;
    CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 4));
    CHECK(p - 2  == global_ptr<int>(0, 0, sizeof(int) * 2));
    p++;
    CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 5));
    p--;
    CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 4));
    p += 10;
    CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 14));
    p -= 5;
    CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 9));
    CHECK(p - p1 == 9);
    CHECK(p1 - p == -9);
    CHECK(p - p  == 0);
  }

  SUBCASE("equality") {
    CHECK(p1 == p1);
    CHECK(p2 == p2);
    CHECK(p3 == p3);
    CHECK(p1 != p2);
    CHECK(p2 != p3);
    CHECK(p3 != p1);
    CHECK(p1 + 1 != p1);
    CHECK((p1 + 1) - 1 == p1);
  }

  SUBCASE("boolean") {
    CHECK(p1);
    CHECK(p2);
    CHECK(p3);
    CHECK(!p1 == false);
    CHECK(!!p1);
    global_ptr<void> nullp;
    CHECK(!nullp);
    CHECK(!global_ptr<void>());
  }

  SUBCASE("dereference") {
    CHECK(&p1[0] == p1);
    CHECK(&p1[10] == p1 + 10);
  }

  SUBCASE("cast") {
    CHECK(p1 == static_cast<global_ptr<char>>(p1));
    CHECK(p1 + 4 == static_cast<global_ptr<char>>(p1) + 4 * sizeof(int));
  }
}

}
