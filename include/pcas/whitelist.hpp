#pragma once

#include <cstdlib>

#include "pcas/util.hpp"

namespace pcas {

class whitelist {
  using vm_section = section<uintptr_t>;
  using vm_sections = sections<uintptr_t>;

  vm_sections sections_;

public:
  void add(const void* addr, std::size_t size) {
    vm_section s {reinterpret_cast<uintptr_t>(addr), reinterpret_cast<uintptr_t>(addr) + size};
    sections_insert(sections_, s);
  }

  void merge(const whitelist& wl) {
    // TODO: efficient merging
    for (const auto& s : wl.sections_) {
      sections_insert(sections_, s);
    }
  }

  bool contain(const void* addr, std::size_t size) const {
    vm_section s {reinterpret_cast<uintptr_t>(addr), reinterpret_cast<uintptr_t>(addr) + size};
    return sections_contain(sections_, s);
  }

  bool overlap(const void* addr, std::size_t size) const {
    vm_section s {reinterpret_cast<uintptr_t>(addr), reinterpret_cast<uintptr_t>(addr) + size};
    return sections_overlap(sections_, s);
  }

  void clear() { sections_.clear(); }

};

}
