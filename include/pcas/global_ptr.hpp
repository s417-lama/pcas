#pragma once

#include <type_traits>
#include <iterator>
#include <cstdint>
#include <unistd.h>

#include <pcas/util.hpp>

namespace pcas {

using mem_obj_id_t = uint64_t;

template <typename P, typename T>
class global_ptr_if {
  using this_t = global_ptr_if<P, T>;
  using ref_t = typename P::template global_ref<this_t>;

  int          owner_;
  mem_obj_id_t id_;
  uint64_t     offset_;

public:
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = void;
  using reference         = ref_t;
  using iterator_category = std::random_access_iterator_tag;

  global_ptr_if() : owner_(-2), id_(0), offset_(0) {}
  global_ptr_if(int owner, mem_obj_id_t id, uint64_t offset) : owner_(owner), id_(id), offset_(offset) {}

  global_ptr_if(const this_t&) = default;
  this_t& operator=(const this_t&) = default;

  int owner() const noexcept { return owner_; }
  mem_obj_id_t id() const noexcept { return id_; }
  uint64_t offset() const noexcept { return offset_; }

  template<typename U>
  bool is_equal(const global_ptr_if<P, U>& p) const noexcept {
    return owner_  == p.owner() &&
           id_     == p.id()    &&
           offset_ == p.offset();
  }

  template<typename U>
  bool belong_to_same_obj(const global_ptr_if<P, U>& p) const noexcept {
    return owner_ == p.owner() &&
           id_    == p.id();
  }

  explicit operator bool() const noexcept { return !is_equal(this_t()); }
  bool operator!() const noexcept { return is_equal(this_t()); }

  ref_t operator*() const noexcept {
    return *this;
  }

  template <typename Diff>
  ref_t operator[](Diff diff) const noexcept {
    return this_t(owner_, id_, offset_ + diff * sizeof(T));
  }

  template <typename Diff>
  this_t& operator+=(Diff diff) {
    offset_ += diff * sizeof(T);
    return *this;
  }

  template <typename Diff>
  this_t& operator-=(Diff diff) {
    PCAS_CHECK(offset_ >= diff * sizeof(T));
    offset_ -= diff * sizeof(T);
    return *this;
  }

  this_t& operator++() { return (*this) += 1; }
  this_t& operator--() { return (*this) -= 1; }

  this_t operator++(int) { this_t tmp(*this); ++(*this); return tmp; }
  this_t operator--(int) { this_t tmp(*this); --(*this); return tmp; }

  template <typename Diff>
  this_t operator+(Diff diff) const noexcept {
    return {owner_, id_, offset_ + diff * sizeof(T)};
  }

  template <typename Diff>
  this_t operator-(Diff diff) const noexcept {
    PCAS_CHECK(offset_ >= diff * sizeof(T));
    return {owner_, id_, offset_ - diff * sizeof(T)};
  }

  std::ptrdiff_t operator-(const this_t& p) const noexcept {
    PCAS_CHECK(belong_to_same_obj(p));
    PCAS_CHECK(offset_ % sizeof(T) == 0);
    PCAS_CHECK(p.offset_ % sizeof(T) == 0);
    return offset_ / sizeof(T) - p.offset_ / sizeof(T);
  }

  template <typename U>
  explicit operator global_ptr_if<P, U>() const noexcept {
    return {owner_, id_, offset_};
  }

  void swap(this_t& p) {
    this_t tmp(*this);
    *this = p;
    p = tmp;
  }
};

template <typename P, typename T, typename U>
inline bool operator==(const global_ptr_if<P, T>& p1, const global_ptr_if<P, U>& p2) noexcept {
  return p1.is_equal(p2);
}

template <typename P, typename T, typename U>
inline bool operator!=(const global_ptr_if<P, T>& p1, const global_ptr_if<P, U>& p2) noexcept {
  return !p1.is_equal(p2);
}

template <typename P, typename T, typename U>
inline bool operator<(const global_ptr_if<P, T>& p1, const global_ptr_if<P, U>& p2) noexcept {
  PCAS_CHECK(p1.belong_to_same_obj(p2));
  return p1.offset() < p2.offset();
}

template <typename P, typename T, typename U>
inline bool operator>(const global_ptr_if<P, T>& p1, const global_ptr_if<P, U>& p2) noexcept {
  PCAS_CHECK(p1.belong_to_same_obj(p2));
  return p1.offset() > p2.offset();
}

template <typename P, typename T, typename U>
inline bool operator<=(const global_ptr_if<P, T>& p1, const global_ptr_if<P, U>& p2) noexcept {
  PCAS_CHECK(p1.belong_to_same_obj(p2));
  return p1.offset() <= p2.offset();
}

template <typename P, typename T, typename U>
inline bool operator>=(const global_ptr_if<P, T>& p1, const global_ptr_if<P, U>& p2) noexcept {
  PCAS_CHECK(p1.belong_to_same_obj(p2));
  return p1.offset() >= p2.offset();
}

template <typename P, typename T>
inline void swap(global_ptr_if<P, T>& p1, global_ptr_if<P, T>& p2) {
  p1.swap(p2);
}

template <typename P, typename T, typename MemberT>
inline typename P::template global_ref<global_ptr_if<P, MemberT>>
operator->*(global_ptr_if<P, T> ptr, MemberT T::* mp) {
  static T t {};
  uint64_t offset_m = reinterpret_cast<uint64_t>(std::addressof(t.*mp))
                    - reinterpret_cast<uint64_t>(std::addressof(t));
  uint64_t offset = ptr.offset() + offset_m;
  return global_ptr_if<P, MemberT>(ptr.owner(), ptr.id(), offset);
}

template <typename>
struct is_global_ptr : public std::false_type {};

template <typename P, typename T>
struct is_global_ptr<global_ptr_if<P, T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_global_ptr_v = is_global_ptr<T>::value;

template <typename GPtrT>
class global_ref_base {
protected:
  GPtrT ptr_;
public:
  global_ref_base(const GPtrT& p) : ptr_(p) {}
  GPtrT operator&() const noexcept { return ptr_; }
};

struct global_ptr_policy_default {
  template <typename GPtrT>
  using global_ref = global_ref_base<GPtrT>;
};

namespace test {

template <typename T>
using global_ptr = global_ptr_if<global_ptr_policy_default, T>;

static_assert(is_global_ptr_v<global_ptr<int>>);
static_assert(!is_global_ptr_v<int>);

PCAS_TEST_CASE("[pcas::global_ptr] global pointer manipulation") {
  global_ptr<int> p1(0, 0, 0);
  global_ptr<int> p2(1, 0, 0);
  global_ptr<int> p3(1, 1, 0);

  PCAS_SUBCASE("initialization") {
    global_ptr<int> p1_(p1);
    global_ptr<int> p2_(p1.owner(), p1.id(), p1.offset());
    PCAS_CHECK(p1_ == p2_);
  }

  PCAS_SUBCASE("addition and subtraction") {
    auto p = p1 + 4;
    PCAS_CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 4));
    PCAS_CHECK(p - 2  == global_ptr<int>(0, 0, sizeof(int) * 2));
    p++;
    PCAS_CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 5));
    p--;
    PCAS_CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 4));
    p += 10;
    PCAS_CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 14));
    p -= 5;
    PCAS_CHECK(p      == global_ptr<int>(0, 0, sizeof(int) * 9));
    PCAS_CHECK(p - p1 == 9);
    PCAS_CHECK(p1 - p == -9);
    PCAS_CHECK(p - p  == 0);
  }

  PCAS_SUBCASE("equality") {
    PCAS_CHECK(p1 == p1);
    PCAS_CHECK(p2 == p2);
    PCAS_CHECK(p3 == p3);
    PCAS_CHECK(p1 != p2);
    PCAS_CHECK(p2 != p3);
    PCAS_CHECK(p3 != p1);
    PCAS_CHECK(p1 + 1 != p1);
    PCAS_CHECK((p1 + 1) - 1 == p1);
  }

  PCAS_SUBCASE("comparison") {
    PCAS_CHECK(p1 < p1 + 1);
    PCAS_CHECK(p1 <= p1 + 1);
    PCAS_CHECK(p1 <= p1);
    PCAS_CHECK(!(p1 < p1));
    PCAS_CHECK(!(p1 + 1 < p1));
    PCAS_CHECK(!(p1 + 1 <= p1));
    PCAS_CHECK(p1 + 1 > p1);
    PCAS_CHECK(p1 + 1 >= p1);
    PCAS_CHECK(p1 >= p1);
    PCAS_CHECK(!(p1 > p1));
    PCAS_CHECK(!(p1 > p1 + 1));
    PCAS_CHECK(!(p1 >= p1 + 1));
  }

  PCAS_SUBCASE("boolean") {
    PCAS_CHECK(p1);
    PCAS_CHECK(p2);
    PCAS_CHECK(p3);
    PCAS_CHECK(!p1 == false);
    PCAS_CHECK(!!p1);
    global_ptr<void> nullp;
    PCAS_CHECK(!nullp);
    PCAS_CHECK(!global_ptr<void>());
  }

  PCAS_SUBCASE("dereference") {
    PCAS_CHECK(&(*p1) == p1);
    PCAS_CHECK(&p1[0] == p1);
    PCAS_CHECK(&p1[10] == p1 + 10);
    struct point { int x; int y; int z; };
    global_ptr<point> px(0, 0, 0);
    PCAS_CHECK(&(px->*(&point::x)) == global_ptr<int>(0, 0, offsetof(point, x)));
    PCAS_CHECK(&(px->*(&point::y)) == global_ptr<int>(0, 0, offsetof(point, y)));
    PCAS_CHECK(&(px->*(&point::z)) == global_ptr<int>(0, 0, offsetof(point, z)));
  }

  PCAS_SUBCASE("cast") {
    PCAS_CHECK(p1 == static_cast<global_ptr<char>>(p1));
    PCAS_CHECK(p1 + 4 == static_cast<global_ptr<char>>(p1) + 4 * sizeof(int));
  }
}

}

}
