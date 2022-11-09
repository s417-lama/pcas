#pragma once

#include <type_traits>
#include <iterator>
#include <cstdint>
#include <unistd.h>

#include <pcas/util.hpp>

namespace pcas {

template <typename P>
struct global_ptr_if {
  template <typename T>
  class global_ptr {
    using this_t = global_ptr<T>;

    T* raw_ptr_ = nullptr;

  public:
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = void;
    using reference         = typename P::template global_ref<this_t>;
    using iterator_category = std::random_access_iterator_tag;

    using policy = P;
    static constexpr bool is_global_ptr_v = true;

    global_ptr() {}
    explicit global_ptr(T* ptr) : raw_ptr_(ptr) {}

    global_ptr(const this_t&) = default;
    this_t& operator=(const this_t&) = default;

    global_ptr(std::nullptr_t) {}
    this_t& operator=(std::nullptr_t) { raw_ptr_ = nullptr; }

    T* raw_ptr() const noexcept { return raw_ptr_; }

    explicit operator bool() const noexcept { return raw_ptr_ != nullptr; }
    bool operator!() const noexcept { return raw_ptr_ == nullptr; }

    reference operator*() const noexcept {
      return *this;
    }

    reference operator[](difference_type diff) const noexcept {
      return this_t(raw_ptr_ + diff);
    }

    this_t& operator+=(difference_type diff) {
      raw_ptr_ += diff;
      return *this;
    }

    this_t& operator-=(difference_type diff) {
      raw_ptr_ -= diff;
      return *this;
    }

    this_t& operator++() { return (*this) += 1; }
    this_t& operator--() { return (*this) -= 1; }

    this_t operator++(int) { this_t tmp(*this); ++(*this); return tmp; }
    this_t operator--(int) { this_t tmp(*this); --(*this); return tmp; }

    this_t operator+(difference_type diff) const noexcept {
      return this_t(raw_ptr_ + diff);
    }

    this_t operator-(difference_type diff) const noexcept {
      return this_t(raw_ptr_ - diff);
    }

    difference_type operator-(const this_t& p) const noexcept {
      return raw_ptr_ - p.raw_ptr();
    }

    template <typename U>
    explicit operator global_ptr<U>() const noexcept {
      return global_ptr<U>(reinterpret_cast<U*>(raw_ptr_));
    }

    void swap(this_t& p) {
      std::swap(raw_ptr_, p.raw_ptr_);
    }

    friend bool operator==(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
      return p1.raw_ptr() == p2.raw_ptr();
    }

    friend bool operator==(const global_ptr<T>& p, std::nullptr_t) noexcept {
      return !p;
    }

    friend inline bool operator==(std::nullptr_t, const global_ptr<T>& p) noexcept {
      return !p;
    }

    friend inline bool operator!=(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
      return p1.raw_ptr() != p2.raw_ptr();
    }

    friend inline bool operator!=(const global_ptr<T>& p, std::nullptr_t) noexcept {
      return bool(p);
    }

    friend inline bool operator!=(std::nullptr_t, const global_ptr<T>& p) noexcept {
      return bool(p);
    }

    friend inline bool operator<(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
      return p1.raw_ptr() < p2.raw_ptr();
    }

    friend inline bool operator>(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
      return p1.raw_ptr() > p2.raw_ptr();
    }

    friend inline bool operator<=(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
      return p1.raw_ptr() <= p2.raw_ptr();
    }

    friend inline bool operator>=(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
      return p1.raw_ptr() >= p2.raw_ptr();
    }

    friend void swap(global_ptr<T>& p1, global_ptr<T>& p2) {
      p1.swap(p2);
    }

  };
};

template <template <typename> typename GPtr, typename T, typename MemberT>
inline typename GPtr<T>::policy::template global_ref<GPtr<std::remove_extent_t<MemberT>>>
operator->*(GPtr<T> ptr, MemberT T::* mp) {
  using member_t = std::remove_extent_t<MemberT>;
  member_t* member_ptr = reinterpret_cast<member_t*>(std::addressof(ptr.raw_ptr()->*mp));
  return GPtr<member_t>(member_ptr);
}

template <typename, typename = void>
struct is_global_ptr : public std::false_type {};

template <typename GPtrT>
struct is_global_ptr<GPtrT, std::enable_if_t<GPtrT::is_global_ptr_v>> : public std::true_type {};

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
using global_ptr = global_ptr_if<global_ptr_policy_default>::global_ptr<T>;

static_assert(is_global_ptr_v<global_ptr<int>>);
template <typename T> struct test_template_type {};
static_assert(!is_global_ptr_v<test_template_type<int>>);
static_assert(!is_global_ptr_v<int>);

PCAS_TEST_CASE("[pcas::global_ptr] global pointer manipulation") {
  int* a1 = reinterpret_cast<int*>(0x00100000);
  int* a2 = reinterpret_cast<int*>(0x01000000);
  int* a3 = reinterpret_cast<int*>(0x10000000);
  global_ptr<int> p1(a1);
  global_ptr<int> p2(a2);
  global_ptr<int> p3(a3);

  PCAS_SUBCASE("initialization") {
    global_ptr<int> p1_(p1);
    global_ptr<int> p2_ = p1;
    PCAS_CHECK(p1_ == p2_);
    int v = 0;
    global_ptr<int> p3_(&v);
  }

  PCAS_SUBCASE("addition and subtraction") {
    auto p = p1 + 4;
    PCAS_CHECK(p      == global_ptr<int>(a1 + 4));
    PCAS_CHECK(p - 2  == global_ptr<int>(a1 + 2));
    p++;
    PCAS_CHECK(p      == global_ptr<int>(a1 + 5));
    p--;
    PCAS_CHECK(p      == global_ptr<int>(a1 + 4));
    p += 10;
    PCAS_CHECK(p      == global_ptr<int>(a1 + 14));
    p -= 5;
    PCAS_CHECK(p      == global_ptr<int>(a1 + 9));
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
    PCAS_CHECK(nullp == global_ptr<void>(nullptr));
    PCAS_CHECK(nullp == nullptr);
    PCAS_CHECK(nullptr == nullp);
    PCAS_CHECK(!(nullp != nullptr));
    PCAS_CHECK(!(nullptr != nullp));
    PCAS_CHECK(p1 != nullptr);
    PCAS_CHECK(nullptr != p1);
    PCAS_CHECK(!(p1 == nullptr));
    PCAS_CHECK(!(nullptr == p1));
  }

  PCAS_SUBCASE("dereference") {
    PCAS_CHECK(&(*p1) == p1);
    PCAS_CHECK(&p1[0] == p1);
    PCAS_CHECK(&p1[10] == p1 + 10);
    struct point1 { int x; int y; int z; };
    uintptr_t base_addr = 0x00300000;
    global_ptr<point1> px1(reinterpret_cast<point1*>(base_addr));
    PCAS_CHECK(&(px1->*(&point1::x)) == global_ptr<int>(reinterpret_cast<int*>(base_addr + offsetof(point1, x))));
    PCAS_CHECK(&(px1->*(&point1::y)) == global_ptr<int>(reinterpret_cast<int*>(base_addr + offsetof(point1, y))));
    PCAS_CHECK(&(px1->*(&point1::z)) == global_ptr<int>(reinterpret_cast<int*>(base_addr + offsetof(point1, z))));
    struct point2 { int v[3]; };
    global_ptr<point2> px2(reinterpret_cast<point2*>(base_addr));
    global_ptr<int> pv = &(px2->*(&point2::v));
    PCAS_CHECK(pv == global_ptr<int>(reinterpret_cast<int*>(base_addr)));
    PCAS_CHECK(&pv[0] == global_ptr<int>(reinterpret_cast<int*>(base_addr) + 0));
    PCAS_CHECK(&pv[1] == global_ptr<int>(reinterpret_cast<int*>(base_addr) + 1));
    PCAS_CHECK(&pv[2] == global_ptr<int>(reinterpret_cast<int*>(base_addr) + 2));
  }

  PCAS_SUBCASE("cast") {
    PCAS_CHECK(global_ptr<char>(reinterpret_cast<char*>(p1.raw_ptr())) == static_cast<global_ptr<char>>(p1));
    PCAS_CHECK(static_cast<global_ptr<char>>(p1 + 4) == static_cast<global_ptr<char>>(p1) + 4 * sizeof(int));
  }

  PCAS_SUBCASE("swap") {
    auto p1_copy = p1;
    auto p2_copy = p2;
    swap(p1, p2);
    PCAS_CHECK(p1 == p2_copy);
    PCAS_CHECK(p2 == p1_copy);
    p1.swap(p2);
    PCAS_CHECK(p1 == p1_copy);
    PCAS_CHECK(p2 == p2_copy);
  }
}

}

}
