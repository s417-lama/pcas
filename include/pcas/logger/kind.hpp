#pragma once

#include <cstdlib>

namespace pcas {
namespace logger {

class kind {
public:
  enum class value {
    Init = 0,

    Willread,
    Checkout,
    Checkin,
    Release,
    ReleaseLazy,
    Acquire,

    FlushEarly,
    FlushConflicted,

    _NKinds,
  };

  constexpr kind(value val) : val_(val) {}

  constexpr bool operator==(kind k) const { return val_ == k.val_; }
  constexpr bool operator!=(kind k) const { return val_ != k.val_; }

  constexpr bool included(kind kinds[], int n) const {
    return n > 0 && (*this == kinds[0] || included(kinds + 1, n - 1));
  }

#ifndef PCAS_LOGGER_ENABLED_KINDS
#define PCAS_LOGGER_ENABLED_KINDS
#endif

#ifndef PCAS_LOGGER_DISABLED_KINDS
#define PCAS_LOGGER_DISABLED_KINDS
#endif

  constexpr bool is_valid() const {
    kind enabled_kinds[]  = {value::_NKinds, PCAS_LOGGER_ENABLED_KINDS};
    kind disabled_kinds[] = {value::_NKinds, PCAS_LOGGER_DISABLED_KINDS};

    constexpr int n_enabled = sizeof(enabled_kinds) / sizeof(kind);
    constexpr int n_disabled = sizeof(disabled_kinds) / sizeof(kind);
    static_assert(!(n_enabled > 1 && n_disabled > 1),
                  "Enabled kinds and disabled kinds cannot be specified at the same time.");

    if (n_enabled > 1) {
      return included(enabled_kinds + 1, n_enabled - 1);
    } else if (n_disabled > 1) {
      return !included(disabled_kinds + 1, n_disabled - 1);
    } else {
      return true;
    }
  }

#undef PCAS_LOGGER_ENABLED_KINDS
#undef PCAS_LOGGER_DISABLED_KINDS

  static constexpr std::size_t size() {
    return (std::size_t)value::_NKinds;
  }

  constexpr std::size_t index() const {
    return (std::size_t)val_;
  }

  constexpr const char* str() const {
    switch (val_) {
      case value::Init:            return "";

      case value::Willread:        return "willread";
      case value::Checkout:        return "checkout";
      case value::Checkin:         return "checkin";
      case value::Release:         return "release";
      case value::ReleaseLazy:     return "release_lazy";
      case value::Acquire:         return "acquire";

      case value::FlushEarly:      return "flush_early";
      case value::FlushConflicted: return "flush_conflicted";

      default:                     return "other";
    }
  }

private:
  const value val_;
};

}
}
